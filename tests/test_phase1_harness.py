from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from calendar_agent import (
    AgentTask,
    AgentToolPolicy,
    AuthorityFailureKind,
    CalendarAgent,
    EventAttendee,
    RiskLevel,
    ResultStatus,
    ToolCallRecord,
)
from calendar_agent.tools import ReadOnlyCalendarTool
from harness.assertions import verify_rescheduling_plan
from harness.fixtures import (
    attendee_consistency_errors,
    assert_planning_side_effect_free,
    audit_fixture_events,
    authoritative_party_matches,
    booking_consistency_errors,
    duration_satisfies_goal,
    event_attendee_fixtures,
    execution_evidence_errors,
    execution_evidence_from_trace,
    free_busy_fixtures,
    free_busy_authority_errors,
    goal1_duration_fixture,
    phase1_party_directory,
    phase1_party_records,
    Phase1PartyCalendarTools,
    phase1_tool_policy,
    provider_401_fixture,
    subscription_fixtures,
    workflow_actions,
)
from harness.phase1 import (
    HostPoolBoundary,
    MeetingBooking,
    normalize_free_busy,
    planning_side_effect_errors,
    replay_fingerprint,
    stale_subscription_reference,
    subscription_errors,
)
from tests.unit.fakes import FakeCalendarTools


def test_party_directory_has_multiple_pages_and_no_authoritative_plant_role() -> None:
    directory = phase1_party_directory()

    assert len(directory.records) == 230
    assert len(directory.page(page=1).records) == 100
    assert directory.page(page=1).has_more
    assert len(directory.all_pages()) == 3
    assert not authoritative_party_matches(directory.records, "Plant Manager")
    assert not authoritative_party_matches(directory.records, "Plant Head")
    assert not authoritative_party_matches(directory.records, "Plant")


def test_party_page_two_is_not_discarded_as_page_one_truth() -> None:
    directory = phase1_party_directory()

    pages = directory.all_pages()

    assert pages[0].records != pages[1].records
    assert sum(len(page.records) for page in pages) == 230
    assert pages[-1].has_more is False


def test_plausible_party_name_does_not_create_authority() -> None:
    records = phase1_party_directory().records

    assert all(record.name for record in records)
    assert authoritative_party_matches(records, "Aarav") == ()


def test_missing_party_identity_blocks_goal_one() -> None:
    result = CalendarAgent(FakeCalendarTools(parties=phase1_party_records())).run(
        AgentTask("calendar.find_30_minutes", "Find time with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert not any(call.mutating for call in result.tool_calls)


def test_planner_consumes_all_party_pages_before_resolving_identity() -> None:
    tools = Phase1PartyCalendarTools()

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find time with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert len(tools.trace) == 6
    assert {call.arguments["page"] for call in tools.trace} == {1, 2, 3}


def test_read_only_wrapper_preserves_page_aware_party_resolution() -> None:
    delegate = Phase1PartyCalendarTools()
    wrapped = ReadOnlyCalendarTool(delegate)

    result = CalendarAgent(wrapped).run(
        AgentTask("calendar.find_30_minutes", "Find time with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert hasattr(wrapped, "find_parties_page")
    assert len(delegate.trace) == 6
    assert {call.arguments["page"] for call in delegate.trace} == {1, 2, 3}


def test_read_only_wrapper_keeps_legacy_party_list_path() -> None:
    delegate = FakeCalendarTools(parties=phase1_party_records())
    wrapped = ReadOnlyCalendarTool(delegate)

    result = CalendarAgent(wrapped).run(
        AgentTask("calendar.find_30_minutes", "Find time with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not hasattr(wrapped, "find_parties_page")
    assert [call.tool for call in delegate.trace] == ["Party.list", "Party.list"]
    assert all(call.arguments == {"limit": 1000, "job_title": title} for call, title in zip(
        delegate.trace,
        ("Plant Manager", "Plant Head"),
    ))


def test_host_pool_is_an_inaccessible_boundary() -> None:
    boundary = HostPoolBoundary()

    assert boundary.resource == "HostPool"
    assert boundary.accessible is False
    assert "authority" in boundary.reason


def test_attendee_rows_reference_events_and_capture_roles_and_responses() -> None:
    attendees = event_attendee_fixtures()
    events = [{"id": event_id} for event_id in {attendee.event_id for attendee in attendees}]

    assert attendee_consistency_errors(events, attendees) == ()
    assert {attendee.role for attendee in attendees} == {"required", "optional", "organizer", "chair"}
    assert {attendee.response_status for attendee in attendees} == {"accepted", "declined", "tentative", "needs_action"}


def test_attendee_count_mismatch_and_missing_rows_are_visible() -> None:
    attendees = event_attendee_fixtures()
    errors = attendee_consistency_errors(
        [{"id": "audit-1", "participant_count": 9}, {"id": "empty-event", "participant_count": 2}],
        attendees,
    )

    assert any("disagrees" in error for error in errors)
    assert any("no attendee rows" in error for error in errors)


def test_declined_required_attendee_blocks_but_optional_does_not() -> None:
    declined_required = EventAttendee("event-1", "party-1", "one@example.test", "required", "declined")
    declined_optional = EventAttendee("event-1", "party-2", "two@example.test", "optional", "declined")

    from harness.phase1 import required_attendee_feasible

    assert required_attendee_feasible([declined_optional])
    assert not required_attendee_feasible([declined_required])


def test_missing_attendee_rows_are_not_equivalent_to_zero_participants() -> None:
    errors = attendee_consistency_errors([{"id": "event-1", "participant_count": 3}], [])

    assert any("no attendee rows" in error for error in errors)


def test_timezone_variants_normalize_to_the_same_utc_interval() -> None:
    fixtures = free_busy_fixtures()
    utc = normalize_free_busy(fixtures["utc"])
    kolkata = normalize_free_busy(fixtures["asia_kolkata"])
    offset = normalize_free_busy(fixtures["explicit_offset"])

    assert utc.errors == kolkata.errors == offset.errors == ()
    assert utc.intervals == kolkata.intervals == offset.intervals


def test_malformed_and_naive_free_busy_values_are_not_availability() -> None:
    malformed = normalize_free_busy(free_busy_fixtures()["malformed"])
    naive = normalize_free_busy([{"start": "2026-09-22T09:00:00", "end": "2026-09-22T09:30:00"}])

    assert malformed.errors
    assert naive.errors
    assert malformed.authoritative is False
    assert naive.authoritative is False


def test_duplicate_and_overlapping_busy_intervals_remain_observable() -> None:
    fixtures = free_busy_fixtures()

    duplicate = normalize_free_busy(fixtures["duplicate"])
    overlapping = normalize_free_busy(fixtures["overlapping"])

    assert len(duplicate.intervals) == 2
    assert len(overlapping.intervals) == 2
    assert duplicate.authoritative and overlapping.authoritative


def test_inaccessible_empty_stale_and_capped_free_busy_fixtures_are_distinct() -> None:
    fixtures = free_busy_fixtures()

    assert fixtures["inaccessible"]["authoritative"] is False
    assert fixtures["empty"] == []
    assert fixtures["stale"]["max_age_hours"] == 24
    assert fixtures["capped_paginated"]["has_more"] is True
    now = datetime(2026, 9, 22, 8, tzinfo=timezone.utc)
    assert free_busy_authority_errors(fixtures["inaccessible"], now=now, max_age=timedelta(days=1))
    assert free_busy_authority_errors(fixtures["stale"], now=now, max_age=timedelta(days=1))
    assert free_busy_authority_errors(fixtures["capped_paginated"], now=now, max_age=timedelta(days=1))


def test_free_busy_freshness_requires_valid_recent_timezone_aware_timestamp() -> None:
    now = datetime(2026, 9, 22, 8, tzinfo=timezone.utc)
    max_age = timedelta(hours=24)
    fresh = {"last_updated": "2026-09-22T07:30:00Z"}

    assert free_busy_authority_errors(fresh, now=now, max_age=max_age) == ()
    for metadata in (
        {},
        {"last_updated": "not-a-timestamp"},
        {"last_updated": "2026-09-22T07:30:00"},
    ):
        errors = free_busy_authority_errors(metadata, now=now, max_age=max_age)
        assert any("freshness timestamp" in error for error in errors)


def test_goal_one_does_not_treat_sixty_minute_link_as_thirty_minutes() -> None:
    fixture = goal1_duration_fixture()

    assert fixture["default_duration_minutes"] == 30
    assert fixture["scheduling_link"].duration_minutes == 60
    assert not duration_satisfies_goal(30, fixture["scheduling_link"])


def test_goal_one_planner_surfaces_duration_mismatch() -> None:
    tools = FakeCalendarTools(
        parties=[{"id": "plant-1", "email": "plant@example.test", "job_title": "Plant Manager"}],
        calendars=[{"id": "calendar-1", "owner_party_id": "plant-1"}],
        availability_rules=[{
            "party_id": "plant-1",
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
        }],
        scheduling_preferences=[{"default_meeting_duration": 30, "min_notice_hours": 0}],
        free_busy_result={"result": {"subjects": [{
            "subject": "plant@example.test",
            "visibility": "free_busy",
            "busy": [],
            "calendars": [{"access_level": "free_busy"}],
        }]}},
    )
    result = CalendarAgent(tools).run(AgentTask(
        "calendar.find_30_minutes",
        "Find a true 30-minute slot",
        {
            "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
            "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
            "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
            "scheduling_link_duration_minutes": 60,
        },
    ))

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "30-minute duration" in result.summary
    assert not any(call.mutating for call in result.tool_calls)


def test_audit_fixture_contains_buffers_notice_and_boundary_cases() -> None:
    events = audit_fixture_events()
    audit = events[0]

    assert audit["title"] == "Supplier quality audit"
    assert audit["minimum_notice_hours"] == 72
    assert audit["pre_buffer_minutes"] == 15
    assert audit["post_buffer_minutes"] == 30
    assert {event["calendar_id"] for event in events} == {"calendar-1", "calendar-2"}


def test_audit_plan_moves_only_strictly_after_end_and_preserves_duration_order_and_calendar() -> None:
    tools = FakeCalendarTools(events=list(audit_fixture_events()))
    before = deepcopy(tools.events)
    result = CalendarAgent(tools).run(AgentTask(
        "calendar.move_after_audit",
        "Move everything after the supplier audit",
        {"shift": timedelta(hours=1)},
    ))

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after-1", "after-2"]
    assert [move.calendar_id for move in result.rescheduling_plan] == ["calendar-1", "calendar-2"]
    assert result.rescheduling_plan[0].duration_seconds == 3600
    assert result.rescheduling_plan[1].duration_seconds == 5400
    assert tools.events == before
    assert result.claimed_outcome["moved"] is False
    assert verify_rescheduling_plan(
        result,
        expected_audit_id="audit-1",
        expected_event_ids=["after-1", "after-2"],
        audit_end=datetime(2026, 9, 22, 11, 30, tzinfo=timezone.utc),
        shift=timedelta(hours=1),
    ).passed


def test_booking_event_attendee_relationship_inconsistencies_are_not_repaired() -> None:
    bookings = (
        MeetingBooking("booking-1", "event-1", "Current", "2026-09-22T12:00:00+00:00", "2026-09-22T13:00:00+00:00", 1),
        MeetingBooking("booking-missing", "missing-event", "", "bad", "bad", 3),
    )
    events = [{"id": "event-1", "title": "Old", "start_at": "2026-09-22T12:00:00+00:00", "end_at": "2026-09-22T13:00:00+00:00"}, {"id": "orphan-event"}]
    attendees = (EventAttendee("event-1", None, "one@example.test", "required", "accepted"),)

    errors = booking_consistency_errors(bookings, events, attendees)

    assert len(errors) >= 4
    assert any("missing event" in error for error in errors)
    assert any("missing booking" in error for error in errors)
    assert any("stale" in error for error in errors)


def test_workflow_metadata_is_inspectable_but_planning_side_effects_are_forbidden() -> None:
    actions = workflow_actions()

    assert {action.action for action in actions} == {"send_email", "create_activity", "reschedule", "cancel", "booking"}
    assert planning_side_effect_errors([]) == ()
    errors = planning_side_effect_errors([action.action for action in actions])
    assert len(errors) == len(actions)
    assert all("side effect" in error for error in errors)
    assert_planning_side_effect_free([])


def test_policy_requires_explicit_risk_authority_beyond_read_only_flag() -> None:
    read_policy = phase1_tool_policy()
    write_policy = AgentToolPolicy(
        read_only=False,
        risk_mode=RiskLevel.WRITE,
        write_domains=("calendar",),
        allowed_domains=("calendar",),
        allowed_entities=("CalendarEvent",),
        allowed_actions=("CalendarEvent.update",),
    )

    assert read_policy.allows(action="CalendarEvent.list", domain="calendar", entity="CalendarEvent", risk=RiskLevel.READ)
    assert not read_policy.allows(action="CalendarEvent.update", domain="calendar", entity="CalendarEvent", risk=RiskLevel.WRITE)
    assert not read_policy.allows(action="HostPool.list", domain="host", entity="HostPool", risk=RiskLevel.READ)
    assert write_policy.allows(action="CalendarEvent.update", domain="calendar", entity="CalendarEvent", risk=RiskLevel.WRITE)
    assert not write_policy.allows(action="send_email", domain="calendar", entity="CalendarEvent", risk=RiskLevel.POSTING)


def test_execution_evidence_covers_success_failure_input_and_error() -> None:
    trace = (
        ToolCallRecord("CalendarEvent.list", {"limit": 100}),
        ToolCallRecord("CalendarEvent.update", {"id": "event-1"}, succeeded=False, error="permission_denied"),
    )

    job = execution_evidence_from_trace(trace, evaluation_decision="revise", approval_state="not_approved")

    assert len(job.steps) == 2
    assert job.steps[0].tool_input == {"limit": 100}
    assert job.steps[1].status == "failed"
    assert job.steps[1].authority_failure is not None
    assert job.steps[1].authority_failure.kind is AuthorityFailureKind.TOOL_PERMISSION_DENIED
    assert execution_evidence_errors(job, trace) == ()


def test_execution_evidence_detects_missing_invocation_result_and_replays_deterministically() -> None:
    trace = (ToolCallRecord("CalendarEvent.list", {}),)
    job = execution_evidence_from_trace(trace)
    broken = job.__class__(job_id=job.job_id, status=job.status, steps=(), messages=job.messages)

    assert execution_evidence_errors(broken, trace)
    assert replay_fingerprint(job) == replay_fingerprint(job)
    assert job.steps[0].status == "succeeded"


def test_execution_evidence_aggregate_status_matches_step_statuses() -> None:
    empty_job = execution_evidence_from_trace(())
    assert empty_job.status == "completed"
    assert execution_evidence_errors(empty_job, ()) == ()

    successful_trace = (ToolCallRecord("CalendarEvent.list", {}),)
    successful_job = execution_evidence_from_trace(successful_trace)
    assert successful_job.status == "completed"
    assert execution_evidence_errors(successful_job, successful_trace) == ()

    failed_trace = (
        ToolCallRecord("CalendarEvent.update", {"id": "event-1"}, succeeded=False, error="permission_denied"),
    )
    failed_job = execution_evidence_from_trace(failed_trace)
    assert failed_job.status == "failed"
    assert execution_evidence_errors(failed_job, failed_trace) == ()
    assert execution_evidence_errors(replace(failed_job, status="completed"), failed_trace)
    assert execution_evidence_errors(replace(successful_job, status="failed"), successful_trace)


def test_execution_evidence_rejects_unsupported_aggregate_and_step_statuses() -> None:
    trace = (ToolCallRecord("CalendarEvent.list", {}),)
    job = execution_evidence_from_trace(trace)
    skipped_step = replace(job.steps[0], status="skipped")

    assert execution_evidence_errors(replace(job, status="pending"), trace)
    assert execution_evidence_errors(replace(job, steps=(skipped_step,)), trace)


def test_provider_401_is_not_an_agent_resource_authority_failure() -> None:
    failure = provider_401_fixture()

    assert failure.status_code == 401
    assert failure.invocation_attempted
    assert not failure.completion_occurred
    assert failure.tool_calls == 0
    assert failure.resource_reads == 0
    assert failure.mutations == 0
    assert failure.token_usage is None
    assert failure.cost is None
    assert all(item.kind is not AuthorityFailureKind.AGENT_UNAUTHORIZED for item in __import__("harness.phase1", fromlist=["authority_failure_fixtures"]).authority_failure_fixtures() if item.provider_status == 401)


def test_subscription_status_errors_prevent_stale_or_failed_data_from_being_authoritative() -> None:
    now = datetime(2026, 9, 22, 8, tzinfo=timezone.utc)
    subscriptions = subscription_fixtures()

    assert subscription_errors(subscriptions[0], now=now, max_age=timedelta(days=1)) == ()
    assert subscription_errors(subscriptions[1], now=now, max_age=timedelta(days=1))
    assert subscription_errors(subscriptions[2], now=now, max_age=timedelta(days=1))
    assert stale_subscription_reference(now)


def test_phase1_fixtures_are_offline_and_have_no_external_action_records() -> None:
    assert all(action.enabled for action in workflow_actions())
    assert not planning_side_effect_errors([])
    assert provider_401_fixture().mutations == 0
