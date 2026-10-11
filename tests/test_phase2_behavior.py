from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from calendar_agent import (
    AgentPolicy,
    AgentTask,
    BoundedAgentRuntime,
    CalendarAgent,
    ResultStatus,
    SelectGoal,
    TerminationReason,
    ToolCallRecord,
)
from calendar_agent.models import MutationDisabledError
from calendar_agent.tools import ReadOnlyCalendarTool
from harness.assertions import verify_result, verify_supported_slot
from tests.unit.fakes import FakeCalendarTools


AUDIT = {
    "id": "audit-1",
    "title": "Supplier quality audit",
    "description": "Supplier review",
    "start_at": "2026-09-22T09:00:00+00:00",
    "end_at": "2026-09-22T10:00:00+00:00",
    "timezone": "UTC",
    "calendar_id": "calendar-1",
}


def goal_one_context(**overrides: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }
    context.update(overrides)
    return context


def visible_free_busy(busy: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "result": {
            "subjects": [{
                "subject": "plant@example.test",
                "visibility": "free_busy",
                "busy": busy,
                "calendars": [{"access_level": "free_busy"}],
            }],
        },
    }


def goal_one_tools(
    *,
    parties: list[dict[str, Any]] | None = None,
    calendars: list[dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    preferences: list[dict[str, Any]] | None = None,
    free_busy_result: dict[str, Any] | None = None,
) -> FakeCalendarTools:
    return FakeCalendarTools(
        parties=parties if parties is not None else [{
            "id": "plant-1",
            "email": "plant@example.test",
            "job_title": "Plant Manager",
        }],
        calendars=calendars if calendars is not None else [{
            "id": "calendar-1",
            "owner_party_id": "plant-1",
        }],
        availability_rules=rules if rules is not None else [{
            "id": "rule-1",
            "party_id": "plant-1",
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            "date_overrides": [],
        }],
        scheduling_preferences=preferences if preferences is not None else [{
            "default_meeting_duration": 30,
            "min_notice_hours": 0,
        }],
        free_busy_result=free_busy_result if free_busy_result is not None else visible_free_busy([]),
    )


def run_goal_one(tools: FakeCalendarTools, **context: Any):
    return CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", goal_one_context(**context))
    )


def event(event_id: str, start: str, end: str, *, title: str = "Work") -> dict[str, Any]:
    return {
        "id": event_id,
        "title": title,
        "start_at": start,
        "end_at": end,
        "timezone": "UTC",
        "calendar_id": "calendar-1",
    }


def run_goal_two(events: list[dict[str, Any]], *, shift: timedelta = timedelta(hours=1)):
    return CalendarAgent(FakeCalendarTools(events=events)).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", {"shift": shift})
    )


def test_goal_one_success_is_a_single_verified_plan_without_mutation() -> None:
    tools = goal_one_tools()
    before = deepcopy({
        "parties": tools.parties,
        "calendars": tools.calendars,
        "rules": tools.availability_rules,
        "preferences": tools.scheduling_preferences,
        "free_busy": tools.free_busy_result,
    })

    result = run_goal_one(tools)

    assert result.status is ResultStatus.PLANNED
    assert len(result.candidate_slots) == 1
    assert verify_supported_slot(
        result,
        expected_start="2026-09-22T09:00:00+00:00",
        expected_end="2026-09-22T09:30:00+00:00",
        expected_calendar_id="calendar-1",
    ).passed
    assert verify_result(result, expected_status=ResultStatus.PLANNED).passed
    assert not any(call.mutating for call in result.tool_calls)
    assert {
        "parties": tools.parties,
        "calendars": tools.calendars,
        "rules": tools.availability_rules,
        "preferences": tools.scheduling_preferences,
        "free_busy": tools.free_busy_result,
    } == before


def test_goal_one_zero_identity_matches_stops_before_calendar_reads() -> None:
    tools = goal_one_tools(parties=[])

    result = run_goal_one(tools)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "identified" in result.summary
    assert [call.tool for call in tools.trace] == ["Party.list", "Party.list"]


def test_goal_one_multiple_identity_matches_never_selects_arbitrarily() -> None:
    tools = goal_one_tools(parties=[
        {"id": "plant-1", "email": "one@example.test", "job_title": "Plant Manager"},
        {"id": "plant-2", "email": "two@example.test", "job_title": "Plant Head"},
    ])

    result = run_goal_one(tools)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert [call.tool for call in result.tool_calls] == ["Party.list", "Party.list"]
    assert not any(call.mutating for call in result.tool_calls)


@pytest.mark.parametrize(
    "busy",
    [
        [{"start": "2026-09-22T09:00:00+00:00", "end": "2026-09-22T17:00:00+00:00"}],
        [{"start": "2026-09-22T09:00:00+00:00", "end": "2026-09-22T09:30:00+00:00"}, {"start": "2026-09-22T09:30:00+00:00", "end": "2026-09-22T17:00:00+00:00"}],
    ],
)
def test_goal_one_busy_time_never_becomes_a_slot(busy: list[dict[str, str]]) -> None:
    result = run_goal_one(goal_one_tools(free_busy_result=visible_free_busy(busy)))

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


def test_goal_one_unavailable_rule_period_is_rejected() -> None:
    tools = goal_one_tools(rules=[{
        "party_id": "plant-1",
        "timezone": "UTC",
        "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
        "date_overrides": [{"date": "2026-09-22", "type": "unavailable"}],
    }])

    result = run_goal_one(tools)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


@pytest.mark.parametrize(
    "preferences",
    [
        [],
        [{"default_meeting_duration": "thirty", "min_notice_hours": 0}],
        [{"default_meeting_duration": 30, "min_notice_hours": "unknown"}],
        [{"default_meeting_duration": 60, "min_notice_hours": 0}],
    ],
)
def test_goal_one_invalid_scheduling_preferences_fail_closed(preferences: list[dict[str, Any]]) -> None:
    result = run_goal_one(goal_one_tools(preferences=preferences))

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


def test_goal_one_minimum_notice_is_applied_at_the_next_valid_boundary() -> None:
    result = run_goal_one(
        goal_one_tools(preferences=[{"default_meeting_duration": 30, "min_notice_hours": 1}]),
        now=datetime(2026, 9, 22, 8, 0, 1, tzinfo=timezone.utc),
        start=datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
    )

    assert result.status is ResultStatus.PLANNED
    assert result.candidate_slots[0].start_at == "2026-09-22T09:30:00+00:00"


@pytest.mark.parametrize("link_duration", [30, 45, 60])
def test_goal_one_scheduling_link_duration_is_explicit(link_duration: int) -> None:
    result = run_goal_one(goal_one_tools(), scheduling_link_duration_minutes=link_duration)

    if link_duration == 30:
        assert result.status is ResultStatus.PLANNED
    else:
        assert result.status is ResultStatus.NEEDS_CLARIFICATION
        assert not result.candidate_slots


def test_goal_one_absent_scheduling_link_duration_preserves_legacy_behavior() -> None:
    result = run_goal_one(goal_one_tools())

    assert result.status is ResultStatus.PLANNED
    assert len(result.candidate_slots) == 1


@pytest.mark.parametrize(
    "tools",
    [
        goal_one_tools(free_busy_result=visible_free_busy([{"start": "bad", "end": "2026-09-22T10:00:00Z"}])),
        goal_one_tools(rules=[{"party_id": "plant-1", "timezone": "UTC", "weekly_hours": "bad"}]),
        goal_one_tools(calendars=[{"id": "", "owner_party_id": "plant-1"}]),
        goal_one_tools(calendars=[{"id": "calendar-1", "owner_party_id": "other-party"}]),
    ],
)
def test_goal_one_malformed_authority_data_never_produces_a_confident_slot(tools: FakeCalendarTools) -> None:
    result = run_goal_one(tools)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert not any(call.mutating for call in result.tool_calls)


def test_goal_one_read_only_wrapper_blocks_mutation_attempts() -> None:
    delegate = goal_one_tools()
    wrapped = ReadOnlyCalendarTool(delegate)

    result = CalendarAgent(wrapped).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", goal_one_context())
    )

    assert result.status is ResultStatus.PLANNED
    with pytest.raises(MutationDisabledError):
        wrapped.create_event(title="must not execute")
    assert delegate.trace[-1] == ToolCallRecord(
        "CalendarEvent.create", {"title": "must not execute"}, mutating=True, succeeded=False, error="disabled"
    )


def test_goal_two_boundary_filtering_is_strict_and_preserves_source_state() -> None:
    events = [
        AUDIT,
        event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T08:30:00+00:00"),
        event("ends-at-audit", "2026-09-22T09:30:00+00:00", "2026-09-22T10:00:00+00:00"),
        event("starts-at-audit-end", "2026-09-22T10:00:00+00:00", "2026-09-22T10:30:00+00:00"),
        event("overlap", "2026-09-22T09:30:00+00:00", "2026-09-22T10:30:00+00:00"),
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:30:00+00:00"),
    ]
    tools = FakeCalendarTools(events=events)
    before = deepcopy(tools.events)

    result = CalendarAgent(tools).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", {"shift": timedelta(hours=1)})
    )

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]
    assert result.claimed_outcome["moved"] is False
    assert tools.events == before
    assert not any(call.mutating for call in result.tool_calls)


def test_goal_two_proposals_are_chronologically_ordered() -> None:
    result = run_goal_two([
        AUDIT,
        event("later", "2026-09-22T14:00:00+00:00", "2026-09-22T15:00:00+00:00"),
        event("earlier", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["earlier", "later"]


def test_goal_two_preserves_each_event_duration() -> None:
    result = run_goal_two([
        AUDIT,
        event("short", "2026-09-22T11:00:00+00:00", "2026-09-22T11:15:00+00:00"),
        event("long", "2026-09-22T12:00:00+00:00", "2026-09-22T14:30:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.duration_seconds for move in result.rescheduling_plan] == [900, 9000]


def test_goal_two_malformed_movable_event_fails_closed() -> None:
    result = run_goal_two([
        AUDIT,
        event("broken", "2026-09-22T11:00:00+00:00", "not-a-timestamp"),
        event("valid", "2026-09-22T12:00:00+00:00", "2026-09-22T13:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_goal_two_rejects_a_destination_conflict_before_proposing() -> None:
    result = run_goal_two([
        AUDIT,
        event("overlapping-audit", "2026-09-22T09:30:00+00:00", "2026-09-22T12:00:00+00:00"),
        event("move-me", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ], shift=timedelta(minutes=30))

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "conflict" in result.summary.lower()


def test_goal_two_attendee_state_is_unchanged_even_when_planning_proposes_moves() -> None:
    events = [AUDIT, event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")]
    attendees = [{"event_id": "after", "email": "person@example.test", "status": "accepted"}]
    tools = FakeCalendarTools(events=events)
    before_events = deepcopy(tools.events)
    before_attendees = deepcopy(attendees)

    result = CalendarAgent(tools).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", {"shift": timedelta(hours=1)})
    )

    assert result.status is ResultStatus.PLANNED
    assert tools.events == before_events
    assert attendees == before_attendees
    assert result.claimed_outcome["moved"] is False
    assert not any(call.mutating for call in result.tool_calls)


def test_runtime_provider_failure_has_no_tool_calls_or_authorization_conclusion() -> None:
    class ProviderFailureModel:
        def decide(self, request: Any) -> object:
            raise RuntimeError("provider unavailable")

    tools = goal_one_tools()
    raw_run = BoundedAgentRuntime(CalendarAgent(tools), ProviderFailureModel()).run("Find time")

    assert raw_run.termination_reason is TerminationReason.MODEL_FAILURE
    assert raw_run.result is None
    assert raw_run.tool_trace == ()
    assert tools.trace == []
    assert raw_run.error == "RuntimeError"


def test_runtime_tool_permission_denial_is_not_provider_failure() -> None:
    class DeniedTools(FakeCalendarTools):
        def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
            self.trace.append(ToolCallRecord("Party.list", filters, succeeded=False, error="permission_denied"))
            raise PermissionError("permission denied")

    class FixedModel:
        def decide(self, request: Any) -> SelectGoal:
            return SelectGoal("calendar.find_30_minutes", "Find time", {})

    tools = DeniedTools()
    raw_run = BoundedAgentRuntime(CalendarAgent(tools), FixedModel()).run("Find time")

    assert raw_run.termination_reason is TerminationReason.EXECUTION_FAILURE
    assert raw_run.error == "PermissionError"
    assert raw_run.result is None
    assert raw_run.tool_trace == (
        ToolCallRecord("Party.list", {"limit": 1000, "job_title": "Plant Manager"}, succeeded=False, error="permission_denied"),
    )


def test_runtime_success_preserves_trace_and_planning_status() -> None:
    tools = goal_one_tools()

    class FixedModel:
        def decide(self, request: Any) -> SelectGoal:
            return SelectGoal("calendar.find_30_minutes", "Find time", {})

    policy = AgentPolicy(model=FixedModel())
    raw_run = BoundedAgentRuntime(CalendarAgent(tools), policy.model, policy=policy).run(
        "Find time",
        goal_one_context(),
    )

    assert raw_run.termination_reason is TerminationReason.SUCCEEDED
    assert raw_run.result is not None
    assert raw_run.result.status is ResultStatus.PLANNED
    assert raw_run.result.tool_calls == list(raw_run.tool_trace)
    assert all(not call.mutating for call in raw_run.tool_trace)
