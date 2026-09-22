"""Independent checks for planning-only scenarios."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from calendar_agent.models import AgentResult, ResultStatus, VerificationResult


def verify_result(result: AgentResult, *, expected_status: ResultStatus) -> VerificationResult:
    """Verify observable result state and tool integrity, not prose claims."""

    outcome_ok = result.status == expected_status
    mutation_calls = [call.tool for call in result.tool_calls if call.mutating]
    integrity_ok = not mutation_calls and all(call.succeeded for call in result.tool_calls)
    details: list[str] = []
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