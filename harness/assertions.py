"""Independent checks for planning-only scenarios."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Any

from calendar_agent.models import (
    AgentResult,
    AuditReference,
    CalendarEventRef,
    CandidateSlot,
    EventMove,
    ResultStatus,
    VerificationResult,
)


def _parse_aware_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _valid_candidate_slot(slot: object) -> bool:
    if not isinstance(slot, CandidateSlot):
        return False
    start = _parse_aware_datetime(slot.start_at)
    end = _parse_aware_datetime(slot.end_at)
    return (
        start is not None
        and end is not None
        and end - start == timedelta(minutes=30)
        and isinstance(slot.timezone, str)
        and bool(slot.timezone.strip())
        and slot.availability_confirmed is True
        and isinstance(slot.reason, str)
        and bool(slot.reason.strip())
        and isinstance(slot.party_id, str)
        and bool(slot.party_id.strip())
        and isinstance(slot.calendar_id, str)
        and bool(slot.calendar_id.strip())
    )


def _valid_event_ref(reference: object) -> bool:
    if not isinstance(reference, CalendarEventRef):
        return False
    start = _parse_aware_datetime(reference.start_at)
    end = _parse_aware_datetime(reference.end_at)
    return (
        isinstance(reference.id, str)
        and bool(reference.id.strip())
        and start is not None
        and end is not None
        and end > start
        and isinstance(reference.timezone, str)
        and bool(reference.timezone.strip())
    )


def _valid_event_move(move: object, audit_end: datetime, audit_id: str) -> bool:
    if (
        not isinstance(move, EventMove)
        or not isinstance(move.event_id, str)
        or not move.event_id.strip()
        or move.event_id == audit_id
    ):
        return False
    original_start = _parse_aware_datetime(move.original_start_at)
    original_end = _parse_aware_datetime(move.original_end_at)
    proposed_start = _parse_aware_datetime(move.proposed_start_at)
    proposed_end = _parse_aware_datetime(move.proposed_end_at)
    if None in (original_start, original_end, proposed_start, proposed_end):
        return False
    assert original_start is not None
    assert original_end is not None
    assert proposed_start is not None
    assert proposed_end is not None
    original_duration = original_end - original_start
    proposed_duration = proposed_end - proposed_start
    shift_start = proposed_start - original_start
    shift_end = proposed_end - original_end
    try:
        duration_seconds = float(move.duration_seconds)
        duration_matches = math.isfinite(duration_seconds) and timedelta(seconds=duration_seconds) == original_duration
    except (OverflowError, TypeError, ValueError):
        duration_matches = False
    return (
        original_start > audit_end
        and original_duration > timedelta(0)
        and original_duration == proposed_duration
        and shift_start == shift_end
        and shift_start > timedelta(0)
        and isinstance(move.timezone, str)
        and bool(move.timezone.strip())
        and duration_matches
    )


def _planned_state_errors(result: AgentResult) -> list[str]:
    if result.goal == "calendar.find_30_minutes":
        if (
            not isinstance(result.candidate_slots, list)
            or len(result.candidate_slots) != 1
            or not _valid_candidate_slot(result.candidate_slots[0])
        ):
            return ["A planned find-30-minutes result must contain one valid 30-minute candidate slot."]
        if (
            result.audit_reference is not None
            or not isinstance(result.rescheduling_plan, list)
            or result.rescheduling_plan
        ):
            return ["A find-30-minutes result contains contradictory rescheduling state."]
        if not isinstance(result.claimed_outcome, dict):
            return ["A planned find-30-minutes result has malformed claimed state."]
        claimed_slot = result.claimed_outcome.get("slot")
        if claimed_slot is not None and claimed_slot != result.candidate_slots[0]:
            return ["The claimed slot contradicts the returned candidate slot."]
        return []

    if result.goal == "calendar.move_after_audit":
        reference = result.audit_reference
        if (
            not isinstance(reference, AuditReference)
            or not isinstance(reference.candidates, tuple)
            or len(reference.candidates) != 1
            or not _valid_event_ref(reference.candidates[0])
        ):
            return ["A planned move result must resolve one valid audit reference."]
        audit_end = _parse_aware_datetime(reference.candidates[0].end_at)
        assert audit_end is not None
        audit_id = reference.candidates[0].id
        if not isinstance(result.rescheduling_plan, list) or not result.rescheduling_plan:
            return ["A planned move result must contain a non-empty event plan."]
        if any(not isinstance(move, EventMove) for move in result.rescheduling_plan):
            return ["A planned move result contains an invalid event move."]
        if any(not isinstance(move.event_id, str) for move in result.rescheduling_plan):
            return ["A planned move result contains an event with an invalid identity."]
        if len({move.event_id for move in result.rescheduling_plan}) != len(result.rescheduling_plan):
            return ["A planned move result contains duplicate event IDs."]
        if any(not _valid_event_move(move, audit_end, audit_id) for move in result.rescheduling_plan):
            return ["A planned move result contains an invalid or contradictory event move."]
        if not isinstance(result.candidate_slots, list) or result.candidate_slots:
            return ["A move result contains contradictory candidate-slot state."]
        claimed = result.claimed_outcome
        if not isinstance(claimed, dict):
            return ["A planned move result has malformed claimed state."]
        if "moved" in claimed and claimed["moved"] is not False:
            return ["A planned move result cannot claim that events were moved."]
        planned_count = claimed.get("planned_event_count")
        if planned_count is not None and planned_count != len(result.rescheduling_plan):
            return ["The claimed planned event count contradicts the event plan."]
        claimed_shift = claimed.get("shift")
        if claimed_shift is not None and (
            not isinstance(claimed_shift, timedelta) or claimed_shift <= timedelta(0)
        ):
            return ["The claimed shift is invalid."]
        if isinstance(claimed_shift, timedelta):
            for move in result.rescheduling_plan:
                proposed_start = _parse_aware_datetime(move.proposed_start_at)
                original_start = _parse_aware_datetime(move.original_start_at)
                assert proposed_start is not None
                assert original_start is not None
                if proposed_start - original_start != claimed_shift:
                    return ["The claimed shift contradicts the event plan."]
        return []

    return [f"No structural validator exists for planned goal {result.goal!r}."]


def verify_result(result: AgentResult, *, expected_status: ResultStatus) -> VerificationResult:
    """Verify observable result state and tool integrity, not prose claims."""

    outcome_ok = result.status == expected_status
    mutation_calls = [call.tool for call in result.tool_calls if call.mutating]
    integrity_ok = not mutation_calls and all(call.succeeded for call in result.tool_calls)
    details: list[str] = []
    if result.status is ResultStatus.PLANNED:
        structural_errors = _planned_state_errors(result)
        if structural_errors:
            outcome_ok = False
            details.extend(structural_errors)
    if not outcome_ok:
        details.append(f"Expected status {expected_status.value}, got {result.status.value}.")
    if mutation_calls:
        details.append(f"Mutation calls observed: {', '.join(mutation_calls)}.")
    if not integrity_ok and not mutation_calls:
        details.append("At least one read tool call failed.")
    return VerificationResult(
        passed=outcome_ok and integrity_ok,
        outcome_ok=outcome_ok,
        integrity_ok=integrity_ok,
        details=tuple(details),
    )


def verify_no_mutations(result: AgentResult) -> VerificationResult:
    mutation_calls = [call.tool for call in result.tool_calls if call.mutating]
    return VerificationResult(
        passed=not mutation_calls,
        outcome_ok=True,
        integrity_ok=not mutation_calls,
        details=tuple(f"Mutation observed: {name}" for name in mutation_calls),
    )


def verify_supported_slot(
    result: AgentResult,
    *,
    expected_start: str,
    expected_end: str,
    minimum_notice_boundary: datetime | None = None,
    expected_calendar_id: str | None = None,
) -> VerificationResult:
    """Verify the returned slot's observable shape independently of its prose."""

    if result.status is not ResultStatus.PLANNED or len(result.candidate_slots) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=not any(call.mutating for call in result.tool_calls),
            details=("A planned result must contain exactly one candidate slot.",),
        )
    slot = result.candidate_slots[0]
    slot_ok = (
        slot.start_at == expected_start
        and slot.end_at == expected_end
        and slot.availability_confirmed
    )
    if minimum_notice_boundary is not None:
        try:
            slot_ok = slot_ok and datetime.fromisoformat(slot.start_at) >= minimum_notice_boundary
        except ValueError:
            slot_ok = False
    if expected_calendar_id is not None:
        slot_ok = slot_ok and slot.calendar_id == expected_calendar_id
    return VerificationResult(
        passed=slot_ok and not any(call.mutating for call in result.tool_calls),
        outcome_ok=slot_ok,
        integrity_ok=not any(call.mutating for call in result.tool_calls),
        details=() if slot_ok else ("Candidate slot does not match independently expected state.",),
    )


def verify_rescheduling_plan(
    result: AgentResult,
    *,
    expected_audit_id: str,
    expected_event_ids: list[str],
    audit_end: datetime,
    shift: timedelta,
) -> VerificationResult:
    """Verify a move plan from its event state, not its summary or claimed outcome."""

    mutation_free = not any(call.mutating for call in result.tool_calls)
    reference = result.audit_reference
    if result.status is not ResultStatus.PLANNED or reference is None or len(reference.candidates) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=mutation_free,
            details=("A planned rescheduling result must resolve exactly one audit.",),
        )
    if reference.candidates[0].id != expected_audit_id:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=mutation_free,
            details=("The resolved audit does not match independent state.",),
        )

    plan = result.rescheduling_plan
    plan_ok = [move.event_id for move in plan] == expected_event_ids
    details: list[str] = []
    for move in plan:
        try:
            original_start = datetime.fromisoformat(move.original_start_at)
            original_end = datetime.fromisoformat(move.original_end_at)
            proposed_start = datetime.fromisoformat(move.proposed_start_at)
            proposed_end = datetime.fromisoformat(move.proposed_end_at)
        except ValueError:
            plan_ok = False
            details.append(f"Malformed datetime in plan for {move.event_id}.")
            continue
        if original_start <= audit_end:
            plan_ok = False
            details.append(f"Event {move.event_id} is not strictly after the audit end.")
        if original_end - original_start != proposed_end - proposed_start:
            plan_ok = False
            details.append(f"Event {move.event_id} duration changed.")
        if proposed_start - original_start != shift or proposed_end - original_end != shift:
            plan_ok = False
            details.append(f"Event {move.event_id} shift changed.")
    if not plan_ok:
        details.append("Rescheduling plan does not match independently expected ordering or state.")
    return VerificationResult(
        passed=plan_ok and mutation_free,
        outcome_ok=plan_ok,
        integrity_ok=mutation_free,
        details=tuple(details),
    )