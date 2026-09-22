from datetime import datetime, timezone

from calendar_agent.agent import CalendarAgent
from calendar_agent.models import AgentTask, ResultStatus
from harness.assertions import verify_supported_slot

from .fakes import FakeCalendarTools


def test_plant_manager_ambiguity_is_explicit() -> None:
    tools = FakeCalendarTools(
        parties=[
            {"id": "one", "name": "A", "email": "a@example.test", "job_title": "Plant Manager"},
            {"id": "two", "name": "B", "email": "b@example.test", "job_title": "Plant Manager"},
        ]
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "exactly one" in " ".join(result.ambiguities)
    assert [call.tool for call in result.tool_calls] == ["Party.list", "Party.list"]
    assert not any(call.mutating for call in result.tool_calls)


def test_multiple_audits_are_not_arbitrarily_selected() -> None:
    tools = FakeCalendarTools(
        events=[
            {"id": "audit-1", "title": "Internal quality audit", "start_at": "2026-09-19T09:30", "end_at": "2026-09-19T12:30"},
            {"id": "audit-2", "title": "Supplier quality audit", "start_at": "2026-09-28T10:00", "end_at": "2026-09-28T11:00"},
        ]
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert result.audit_reference is not None
    assert len(result.audit_reference.candidates) == 2
    assert not any(call.mutating for call in result.tool_calls)


def _valid_context() -> dict[str, datetime]:
    return {
        "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }


def _tools_for_party(
    *,
    free_busy_result: dict[str, object],
    rules: list[dict[str, object]] | None = None,
    parties: list[dict[str, object]] | None = None,
    calendars: list[dict[str, object]] | None = None,
    min_notice_hours: float = 0,
) -> FakeCalendarTools:
    actual_parties = parties if parties is not None else [{
        "id": "plant-1",
        "name": "Plant Manager",
        "email": "plant@example.test",
        "job_title": "Plant Manager",
    }]
    party = actual_parties[0] if actual_parties else None
    return FakeCalendarTools(
        parties=actual_parties,
        calendars=calendars if calendars is not None else ([{"id": "calendar-1", "owner_party_id": party["id"]}] if party else []),
        availability_rules=(rules if rules is not None else ([{
            "id": "rule-1",
            "party_id": party["id"],
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            "date_overrides": [],
        }] if party else [])),
        scheduling_preferences=[{"default_meeting_duration": 30, "min_notice_hours": min_notice_hours}],
        free_busy_result=free_busy_result,
    )


def _visible_free_busy(busy: list[dict[str, str]]) -> dict[str, object]:
    return {"result": {"subjects": [{
        "subject": "plant@example.test",
        "visibility": "free_busy",
        "busy": busy,
        "calendars": [{"access_level": "free_busy"}],
    }]}}


def test_unique_plant_head_with_verified_free_slot_is_planned() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    slot_verification = verify_supported_slot(
        result,
        expected_start="2026-09-22T09:00:00+00:00",
        expected_end="2026-09-22T09:30:00+00:00",
        expected_calendar_id="calendar-1",
    )
    assert slot_verification.passed
    assert not any(call.mutating for call in result.tool_calls)


def test_minimum_notice_with_seconds_rounds_up_without_crossing_boundary() -> None:
    now = datetime(2026, 9, 22, 8, 0, 1, tzinfo=timezone.utc)
    minimum_notice_boundary = datetime(2026, 9, 22, 9, 0, 1, tzinfo=timezone.utc)
    context = {
        "now": now,
        "start": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }
    tools = _tools_for_party(
        free_busy_result=_visible_free_busy([]),
        min_notice_hours=1,
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", context)
    )

    verification = verify_supported_slot(
        result,
        expected_start="2026-09-22T09:30:00+00:00",
        expected_end="2026-09-22T10:00:00+00:00",
        minimum_notice_boundary=minimum_notice_boundary,
        expected_calendar_id="calendar-1",
    )
    assert verification.passed


def test_multiple_owned_calendars_require_clarification() -> None:
    tools = _tools_for_party(
        free_busy_result=_visible_free_busy([]),
        calendars=[
            {"id": "calendar-1", "owner_party_id": "plant-1"},
            {"id": "calendar-2", "owner_party_id": "plant-1"},
        ],
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "Multiple" in result.summary
    assert not any(call.mutating for call in result.tool_calls)


def test_not_shared_free_busy_is_not_confirmed_free() -> None:
    tools = _tools_for_party(free_busy_result={"result": {"subjects": [{
        "subject": "plant@example.test",
        "visibility": "unknown",
        "visibility_code": "freebusy.notShared",
        "busy": [],
        "calendars": [{"access_level": "none", "reason_code": "freebusy.notShared"}],
    }]}})

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "not confirmed" in " ".join(result.ambiguities)
    assert not any(call.mutating for call in result.tool_calls)


def test_no_plant_head_is_clarification() -> None:
    tools = _tools_for_party(parties=[], free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


def test_multiple_plant_heads_are_clarification() -> None:
    parties = [
        {"id": "plant-1", "name": "One", "email": "one@example.test", "job_title": "Plant Manager"},
        {"id": "plant-2", "name": "Two", "email": "two@example.test", "job_title": "Plant Manager"},
    ]
    tools = _tools_for_party(parties=parties, free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "found 2" in " ".join(result.ambiguities)


def test_missing_valid_availability_is_clarification() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([]), rules=[])

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "availability rule" in result.summary


def test_only_busy_intervals_produce_no_slot() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([{
        "start": "2026-09-22T09:00:00+00:00",
        "end": "2026-09-22T17:00:00+00:00",
    }]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


def test_valid_free_slot_after_busy_interval_is_independently_supported() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([{
        "start": "2026-09-22T09:00:00+00:00",
        "end": "2026-09-22T09:30:00+00:00",
    }]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    slot_verification = verify_supported_slot(
        result,
        expected_start="2026-09-22T09:30:00+00:00",
        expected_end="2026-09-22T10:00:00+00:00",
    )
    assert slot_verification.passed