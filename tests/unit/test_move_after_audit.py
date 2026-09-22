from __future__ import annotations

from datetime import datetime, timedelta, timezone

from calendar_agent.agent import CalendarAgent
from calendar_agent.models import AgentTask, ResultStatus
from harness.assertions import verify_rescheduling_plan

from .fakes import FakeCalendarTools


AUDIT = {
    "id": "audit-1",
    "title": "Internal quality audit",
    "description": "ISO review",
    "start_at": "2026-09-22T09:00:00+00:00",
    "end_at": "2026-09-22T10:00:00+00:00",
    "timezone": "UTC",
    "calendar_id": "calendar-1",
}
SHIFT = timedelta(hours=1)
AUDIT_END = datetime(2026, 9, 22, 10, tzinfo=timezone.utc)


def event(event_id: str, start: str, end: str, *, title: str = "Work") -> dict[str, str]:
    return {
        "id": event_id,
        "title": title,
        "start_at": start,
        "end_at": end,
        "timezone": "UTC",
        "calendar_id": "calendar-1",
    }


def run(events: list[dict[str, object]], *, shift: timedelta | None = SHIFT):
    context = {"shift": shift} if shift is not None else {}
    return CalendarAgent(FakeCalendarTools(events=events)).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", context)
    )


def test_exactly_one_audit_plans_events_strictly_after_audit_end() -> None:
    result = run([
        AUDIT,
        event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T08:30:00+00:00"),
        event("after-1", "2026-09-22T11:00:00+00:00", "2026-09-22T11:45:00+00:00"),
        event("after-2", "2026-09-22T12:00:00+00:00", "2026-09-22T13:30:00+00:00"),
    ])

    verification = verify_rescheduling_plan(
        result,
        expected_audit_id="audit-1",
        expected_event_ids=["after-1", "after-2"],
        audit_end=AUDIT_END,
        shift=SHIFT,
    )
    assert verification.passed
    assert result.claimed_outcome["moved"] is False
    assert not any(call.mutating for call in result.tool_calls)


def test_multiple_audits_do_not_produce_a_plan() -> None:
    result = run([AUDIT, {**AUDIT, "id": "audit-2", "title": "Supplier audit"}])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert len(result.audit_reference.candidates) == 2


def test_no_audit_does_not_produce_a_plan() -> None:
    result = run([event("work", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_event_overlapping_audit_is_not_selected() -> None:
    result = run([
        AUDIT,
        event("overlap", "2026-09-22T09:30:00+00:00", "2026-09-22T10:30:00+00:00"),
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


def test_malformed_event_is_ignored_without_crashing() -> None:
    result = run([
        AUDIT,
        {"id": "malformed", "title": "Broken", "start_at": "not-a-date", "end_at": "also-broken", "timezone": "UTC"},
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


def test_no_events_after_audit_is_clarification() -> None:
    result = run([
        AUDIT,
        event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T09:00:00+00:00"),
        event("at-end", "2026-09-22T10:00:00+00:00", "2026-09-22T10:30:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_timezone_null_event_is_not_silently_reinterpreted() -> None:
    ambiguous = event("ambiguous", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
    ambiguous["timezone"] = None
    result = run([AUDIT, ambiguous])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "timezone" in result.summary.lower()


def test_missing_shift_is_clarification() -> None:
    result = run([AUDIT, event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")], shift=None)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "amount" in result.summary.lower()
