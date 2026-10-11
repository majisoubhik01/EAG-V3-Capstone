from __future__ import annotations

from datetime import datetime, timedelta, timezone

from calendar_agent.agent import CalendarAgent
from calendar_agent.models import AgentTask, ResultStatus
from harness.assertions import verify_rescheduling_plan, verify_result

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


class SplitAuditListingTools(FakeCalendarTools):
    def __init__(self, full_listing: list[dict[str, object]]) -> None:
        super().__init__()
        self.full_listing = full_listing

    def list_events(self, **filters: object) -> list[dict[str, object]]:
        self._record("CalendarEvent.list", filters)
        if filters.get("search") == "audit":
            return [AUDIT]
        return self.full_listing


def run(events: list[dict[str, object]], *, shift: timedelta | None = SHIFT):
    context = {"shift": shift} if shift is not None else {}
    return CalendarAgent(FakeCalendarTools(events=events)).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", context)
    )


def run_with_split_listings(full_listing: list[dict[str, object]]):
    return CalendarAgent(SplitAuditListingTools(full_listing)).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", {"shift": SHIFT})
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
    assert verify_result(result, expected_status=ResultStatus.PLANNED).passed
    assert result.claimed_outcome["moved"] is False
    assert not any(call.mutating for call in result.tool_calls)


def test_audit_missing_from_full_listing_requires_clarification() -> None:
    result = run_with_split_listings([
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "audit-1" in " ".join(result.ambiguities)
    assert "0 matches" in " ".join(result.ambiguities)


def test_audit_duplicated_in_full_listing_requires_clarification() -> None:
    result = run_with_split_listings([
        AUDIT,
        {**AUDIT, "title": "Duplicate audit copy"},
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "audit-1" in " ".join(result.ambiguities)
    assert "2 matches" in " ".join(result.ambiguities)


def test_exactly_one_audit_in_full_listing_continues_valid_planning() -> None:
    result = run_with_split_listings([
        AUDIT,
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


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


def test_malformed_event_with_unknown_position_requires_clarification() -> None:
    result = run([
        AUDIT,
        {"id": "malformed", "title": "Broken", "start_at": "not-a-date", "end_at": "also-broken", "timezone": "UTC"},
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "malformed" in " ".join(result.ambiguities)


def test_malformed_end_before_audit_cannot_hide_a_destination_conflict() -> None:
    result = run([
        AUDIT,
        event("uncertain", "2026-09-22T08:00:00+00:00", "not-a-date"),
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("uncertain" in ambiguity for ambiguity in result.ambiguities)


def test_unrelated_pre_audit_interval_without_timezone_is_not_blanket_rejected() -> None:
    without_timezone = event("before", "2026-09-22T07:00:00+00:00", "2026-09-22T08:00:00+00:00")
    del without_timezone["timezone"]

    result = run([
        AUDIT,
        without_timezone,
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


def test_malformed_audit_candidate_requires_clarification() -> None:
    malformed_audit = {**AUDIT, "start_at": "not-a-date"}

    result = run([malformed_audit, event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_malformed_affected_event_that_starts_after_audit_requires_clarification() -> None:
    malformed = event(
        "broken",
        "2026-09-22T11:00:00+00:00",
        "2026-09-22T10:30:00+00:00",
    )

    result = run([AUDIT, malformed, event("after", "2026-09-22T12:00:00+00:00", "2026-09-22T13:00:00+00:00")])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


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


def test_missing_audit_id_does_not_produce_a_plan() -> None:
    result = run([
        {**AUDIT, "id": ""},
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_missing_affected_event_id_does_not_produce_a_plan() -> None:
    missing_id = event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
    del missing_id["id"]

    result = run([AUDIT, missing_id])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_duplicate_affected_event_ids_do_not_produce_a_plan() -> None:
    result = run([
        AUDIT,
        event("duplicate", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
        event("duplicate", "2026-09-22T13:00:00+00:00", "2026-09-22T14:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_affected_event_sharing_audit_id_does_not_produce_a_plan() -> None:
    result = run([
        AUDIT,
        event("audit-1", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
        event("after", "2026-09-22T13:00:00+00:00", "2026-09-22T14:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_successful_planning_does_not_mutate_source_events() -> None:
    from copy import deepcopy

    events = [
        AUDIT,
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ]
    tools = FakeCalendarTools(events=events)
    before = deepcopy(tools.events)

    result = CalendarAgent(tools).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", {"shift": SHIFT})
    )

    assert result.status is ResultStatus.PLANNED
    assert result.status is not ResultStatus.COMPLETED
    assert tools.events == before
    assert not any(call.mutating for call in result.tool_calls)


def test_missing_shift_is_clarification() -> None:
    result = run([AUDIT, event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")], shift=None)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "amount" in result.summary.lower()
