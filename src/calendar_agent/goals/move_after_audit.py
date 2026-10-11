"""Planning-only handler for ``calendar.move_after_audit``."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..models import (
    AgentResult,
    AgentTask,
    AuditReference,
    CalendarEventRef,
    EventMove,
    ResultStatus,
)
from ..tools import CalendarTool


GOAL = "calendar.move_after_audit"


def _event_ref(event: dict[str, object]) -> CalendarEventRef:
    return CalendarEventRef(
        id=str(event.get("id", "")),
        title=str(event.get("title", "")),
        start_at=str(event.get("start_at", "")),
        end_at=str(event.get("end_at", "")),
        calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
        status=event.get("status") if isinstance(event.get("status"), str) else None,
        timezone=event.get("timezone") if isinstance(event.get("timezone"), str) else None,
    )


def _event_id(event: dict[str, Any]) -> str | None:
    value = event.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_audit_candidate(event: dict[str, Any]) -> bool:
    text = " ".join(str(event.get(key) or "") for key in ("title", "description"))
    return "audit" in text.lower()


def _parse_event_interval(event: dict[str, Any]) -> tuple[datetime, datetime] | None:
    start_value = event.get("start_at")
    end_value = event.get("end_at")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return None
    try:
        start = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        return None
    return start, end


def _parse_event_times(event: dict[str, Any]) -> tuple[datetime, datetime] | None:
    interval = _parse_event_interval(event)
    if interval is None:
        return None
    if not isinstance(event.get("timezone"), str) or not event["timezone"].strip():
        return None
    return interval


def _parse_event_start(event: dict[str, Any]) -> datetime | None:
    value = event.get("start_at")
    if not isinstance(value, str):
        return None
    try:
        start = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return start if start.tzinfo is not None else None


def _intervals_overlap(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> bool:
    return first_start < second_end and first_end > second_start


def _clarification(
    task: AgentTask,
    audit_reference: AuditReference,
    summary: str,
    *ambiguities: str,
) -> AgentResult:
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary=summary,
        ambiguities=list(ambiguities),
        audit_reference=audit_reference,
    )


def plan(task: AgentTask, tools: CalendarTool) -> AgentResult:
    """Build a deterministic, read-only plan for events after one audit."""

    audit_records = tools.list_events(limit=1000, search="audit")
    audit_like = [event for event in audit_records if _is_audit_candidate(event)]
    candidates = tuple(_event_ref(event) for event in audit_like)
    reference = AuditReference(candidates=candidates, search_terms=("audit",))
    if not audit_like:
        return _clarification(
            task,
            reference,
            "No audit candidate was found.",
            "An event whose title or description contains 'audit' is required.",
        )

    parsed_audits = [(event, _parse_event_times(event)) for event in audit_like]
    malformed_audits = [
        event
        for event, parsed in parsed_audits
        if parsed is None or _event_id(event) is None
    ]
    valid_audits = [
        (event, parsed)
        for event, parsed in parsed_audits
        if parsed is not None and _event_id(event) is not None
    ]
    if malformed_audits:
        return _clarification(
            task,
            reference,
            "Audit resolution is ambiguous because an audit candidate has invalid or timezone-ambiguous datetimes.",
            f"{len(malformed_audits)} audit candidate(s) could not be validated.",
        )
    if len(valid_audits) != 1:
        return _clarification(
            task,
            reference,
            "The audit cannot be resolved deterministically.",
            f"Found {len(valid_audits)} valid audit candidates; exactly one is required.",
        )

    audit_event, audit_times = valid_audits[0]
    assert audit_times is not None
    _, audit_end = audit_times
    shift = task.context.get("shift")
    if not isinstance(shift, timedelta) or shift <= timedelta(0):
        return _clarification(
            task,
            reference,
            "The audit is resolved, but the amount to move events is not specified.",
            "Provide a positive timedelta shift; the planner will not invent one.",
        )

    events = tools.list_events(limit=1000)
    audit_id = _event_id(audit_event)
    assert audit_id is not None
    matching_audit_ids = sum(1 for event in events if _event_id(event) == audit_id)
    if matching_audit_ids != 1:
        occurrence = "missing from" if matching_audit_ids == 0 else "duplicated in"
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because the resolved audit is not present exactly once.",
            f"The selected audit ID {audit_id[:100]!r} was {occurrence} the full event listing "
            f"({matching_audit_ids} matches; expected exactly one).",
        )
    moves: list[EventMove] = []
    move_ids: set[str] = set()
    timezone_ambiguous: list[dict[str, Any]] = []
    malformed_events: list[dict[str, Any]] = []
    malformed_after_audit: list[dict[str, Any]] = []
    identity_ambiguous: list[dict[str, Any]] = []
    duplicate_move_ids: set[str] = set()
    for event in events:
        event_id = _event_id(event)
        if event_id == audit_id:
            continue
        parsed = _parse_event_times(event)
        if parsed is None:
            interval = _parse_event_interval(event)
            if interval is not None:
                if interval[0] > audit_end:
                    timezone_ambiguous.append(event)
            else:
                start = _parse_event_start(event)
                if start is not None and start > audit_end:
                    malformed_after_audit.append(event)
                else:
                    malformed_events.append(event)
            continue

        event_start, event_end = parsed
        if event_start <= audit_end:
            continue
        if event_id is None:
            identity_ambiguous.append(event)
            continue
        if event_id in move_ids:
            duplicate_move_ids.add(event_id)
            continue
        proposed_start = event_start + shift
        proposed_end = event_end + shift
        move_ids.add(event_id)
        moves.append(
            EventMove(
                event_id=event_id,
                title=str(event.get("title", "")),
                original_start_at=str(event["start_at"]),
                original_end_at=str(event["end_at"]),
                proposed_start_at=proposed_start.isoformat(),
                proposed_end_at=proposed_end.isoformat(),
                duration_seconds=(event_end - event_start).total_seconds(),
                calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
                timezone=event.get("timezone") if isinstance(event.get("timezone"), str) else None,
            )
        )

    moves.sort(key=lambda move: datetime.fromisoformat(move.original_start_at.replace("Z", "+00:00")))

    if timezone_ambiguous:
        return _clarification(
            task,
            reference,
            "The affected-event boundary is ambiguous because event timezone data is missing.",
            f"{len(timezone_ambiguous)} event(s) have datetime values without an explicit timezone.",
        )
    if malformed_after_audit:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because an event after the audit is malformed.",
            f"{len(malformed_after_audit)} event(s) that would be moved could not be validated.",
        )
    if malformed_events:
        sample_ids = [_event_id(event) or "<missing id>" for event in malformed_events[:5]]
        remaining = len(malformed_events) - len(sample_ids)
        sample = ", ".join(sample_ids)
        if remaining:
            sample += f", and {remaining} more"
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because event timestamps are unusable.",
            f"{len(malformed_events)} event(s) could not be classified or checked for destination conflicts; sample IDs: {sample}.",
        )
    if identity_ambiguous:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because a movable event has no valid identity.",
            f"{len(identity_ambiguous)} event(s) that would be moved have a missing or blank ID.",
        )
    if duplicate_move_ids:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because movable event IDs are duplicated.",
            f"Duplicate movable event IDs: {', '.join(sorted(duplicate_move_ids))}.",
        )

    conflicts: set[str] = set()
    for index, move in enumerate(moves):
        proposed_start = datetime.fromisoformat(move.proposed_start_at.replace("Z", "+00:00"))
        proposed_end = datetime.fromisoformat(move.proposed_end_at.replace("Z", "+00:00"))
        for other_move in moves[index + 1:]:
            other_start = datetime.fromisoformat(other_move.proposed_start_at.replace("Z", "+00:00"))
            other_end = datetime.fromisoformat(other_move.proposed_end_at.replace("Z", "+00:00"))
            if _intervals_overlap(proposed_start, proposed_end, other_start, other_end):
                conflicts.add(f"{move.event_id} with {other_move.event_id}")
        for event in events:
            event_id = _event_id(event)
            if event_id in move_ids or event_id == audit_id:
                continue
            parsed = _parse_event_interval(event)
            if parsed is not None and _intervals_overlap(proposed_start, proposed_end, *parsed):
                conflicts.add(f"{move.event_id} with {event_id or 'an unidentified event'}")
    if conflicts:
        return _clarification(
            task,
            reference,
            "The rescheduling proposal conflicts with an existing event.",
            "Conflicting proposed intervals: " + ", ".join(sorted(conflicts)) + ".",
        )

    if not moves:
        return _clarification(
            task,
            reference,
            "No valid events start strictly after the audit end.",
            "Events overlapping the audit or ending at its boundary are not affected.",
        )

    return AgentResult(
        goal=task.goal,
        status=ResultStatus.PLANNED,
        summary="A read-only rescheduling plan was prepared; no events were moved.",
        audit_reference=reference,
        rescheduling_plan=moves,
        claimed_outcome={"moved": False, "planned_event_count": len(moves), "shift": shift},
    )