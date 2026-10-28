"""Hosted, per-instance read-only checks for AgentSwitch Harness runs."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from calendar_agent import (
    AgentSwitchError,
    AgentSwitchCalendarTools,
    AgentSwitchClient,
    AgentSwitchConfig,
    AgentTask,
    CalendarTool,
    CalendarAgent,
    ConfigurationError,
    MutationDisabledError,
    ResultStatus,
)
from calendar_agent.models import AgentResult, EventMove, ToolCallRecord
from calendar_agent.tools import ReadOnlyCalendarTool
from harnesses.keystone_harness import (
    KeystoneCalendarAdapter,
    KeystoneCalendarTool,
    KeystoneClient,
    KeystoneClientError,
    KeystoneConfig,
    KeystoneConfigurationError,
    KeystoneGoalIntegrationError,
    run_keystone_goal,
)
from harnesses.keystone_harness.authorization import RESOURCE_READ_OPERATIONS

from .assertions import (
    BackendCalendarSnapshot,
    verify_backend_rescheduling_state,
    verify_goal_one_candidate_against_reads,
    verify_result,
)
from .journal import HostedRunJournal
from .results import MAX_EVIDENCE_LENGTH, sanitize_evidence, validate_results
from .server_runner import ServerRunnerError, write_results_atomic


SURYODAYA_INSTANCE = "suryodaya"
KEYSTONE_INSTANCE = "keystone"
SUPPORTED_INSTANCES = frozenset({SURYODAYA_INSTANCE, KEYSTONE_INSTANCE})
GOAL_ONE = "calendar.find_30_minutes"
GOAL_ONE_TITLE = "Find 30 minutes with the plant head"
# No AvailabilityRule is linked to the plant head on the hosted Suryodaya seat, so the
# check opts into this one named company-wide rule instead of failing closed.
SURYODAYA_FALLBACK_RULE_NAME = "Plant visiting hours"
GOAL_TWO = "calendar.move_after_audit"
GOAL_TWO_TITLE = "Move events after the confirmed press shop and welding audit"
GOAL_TWO_AUDIT_NUMBER = "EVT-2026-00007"
GOAL_TWO_AUDIT_TITLE = "Internal quality audit — press shop and welding"
# Keystone has no press-shop audit; its audits repeat the same title, so the check
# uses the earliest event with this title and pins it by ID for the planner.
KEYSTONE_AUDIT_TITLE = "ISO surveillance audit — day 1"
GOAL_TWO_DEFAULT_SHIFT = timedelta(minutes=30)
GOAL_TWO_MAX_SHIFT_SEARCH = timedelta(hours=8)
_SEARCH_WINDOW = timedelta(days=14)


class HostedConfigurationError(ValueError):
    """Raised when the runner environment does not identify an approved instance."""


@dataclass(frozen=True)
class HostedEnvironment:
    instance: str
    base_url: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class HostedCheck:
    task_id: str
    title: str
    passed: bool
    evidence: str
    secret_values: tuple[str, ...] = ()

    def official_record(self) -> dict[str, object]:
        evidence = sanitize_evidence(self.evidence, self.secret_values)[:MAX_EVIDENCE_LENGTH]
        return {
            "id": self.task_id,
            "title": self.title,
            "passed": self.passed,
            "score": 1.0 if self.passed else 0.0,
            "evidence": evidence,
        }


def _task(
    *,
    task_id: str,
    title: str,
    passed: bool,
    evidence: str,
    secret_values: tuple[str, ...] = (),
) -> HostedCheck:
    return HostedCheck(
        task_id=task_id,
        title=title,
        passed=passed,
        evidence=evidence,
        secret_values=secret_values,
    )


def _journaled_check(
    journal: HostedRunJournal,
    *,
    task_id: str,
    title: str,
    goal: str,
    run: Callable[[HostedRunJournal], HostedCheck],
) -> HostedCheck:
    journal.begin(task_id=task_id, title=title, goal=goal)
    check = run(journal)
    if not journal.has_execution_record(task_id):
        journal.record_observation(
            task_id,
            observation={"agent_result_available": False},
        )
    journal.finish(check)
    return check


def _record_unavailable_goal(
    journal: HostedRunJournal,
    *,
    task_id: str,
    title: str,
    goal: str,
    missing_operations: list[str],
    secret_values: tuple[str, ...],
) -> HostedCheck:
    journal.record_observation(
        task_id,
        observation={"missing_required_read_operations": missing_operations},
    )
    return _task(
        task_id=task_id,
        title=title,
        passed=False,
        evidence=(
            "UNEVALUATED: required Keystone read tools were not all discovered; "
            f"goal={goal};no goal calls were made."
        ),
        secret_values=secret_values,
    )


def check_backend_rescheduling_state(
    before: BackendCalendarSnapshot,
    after: BackendCalendarSnapshot,
    *,
    expected_moves: Sequence[EventMove],
    task_id: str,
    title: str,
    secret_values: tuple[str, ...] = (),
) -> HostedCheck:
    """Turn independent backend event readback into a persisted harness check."""

    verification = verify_backend_rescheduling_state(
        before,
        after,
        expected_moves=expected_moves,
    )
    evidence = "BACKEND_READBACK: " + "; ".join(verification.details)
    return _task(
        task_id=task_id,
        title=title,
        passed=verification.passed,
        evidence=evidence,
        secret_values=secret_values,
    )


def write_hosted_results(
    output_path: str | Path,
    checks: tuple[HostedCheck, ...],
    *,
    summary: str,
    secret_values: tuple[str, ...] = (),
) -> Path:
    """Persist custom harness checks using the same validated results contract."""

    document = {
        "tasks": [check.official_record() for check in checks],
        "summary": sanitize_evidence(summary, secret_values)[:MAX_EVIDENCE_LENGTH],
    }
    validate_results(document)
    results_json = json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False)
    return write_results_atomic(output_path, results_json)


def load_hosted_environment(values: Mapping[str, object]) -> HostedEnvironment:
    """Read the runner-provided target without substituting a local origin."""

    instance_value = values.get("AGENTSWITCH_INSTANCE")
    base_url_value = values.get("AGENTSWITCH_BASE_URL")
    token_value = values.get("AGENTSWITCH_TOKEN")
    if not isinstance(instance_value, str) or not instance_value.strip():
        raise HostedConfigurationError("AGENTSWITCH_INSTANCE must be set.")
    if not isinstance(base_url_value, str) or not base_url_value.strip():
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL must be set.")
    if not isinstance(token_value, str) or not token_value.strip():
        raise HostedConfigurationError("AGENTSWITCH_TOKEN must be set.")

    instance = instance_value.strip().casefold()
    if instance not in SUPPORTED_INSTANCES:
        raise HostedConfigurationError(
            "AGENTSWITCH_INSTANCE must select Suryodaya or Keystone."
        )

    base_url = base_url_value.strip()
    if any(character.isspace() or ord(character) < 32 for character in base_url):
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL must not contain whitespace or control characters.")
    try:
        supplied = urlsplit(base_url)
        supplied.port
    except ValueError:
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL is malformed.") from None
    if supplied.scheme.casefold() not in {"http", "https"} or not supplied.hostname:
        raise HostedConfigurationError(
            "AGENTSWITCH_BASE_URL must be an absolute HTTP(S) instance base URL."
        )
    if supplied.username is not None or supplied.password is not None:
        raise HostedConfigurationError(
            "AGENTSWITCH_BASE_URL must be a credential-free HTTP(S) instance base URL."
        )
    if supplied.query or supplied.fragment:
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL must not contain a query or fragment.")
    if supplied.path.rstrip("/").casefold().endswith("/api/mcp"):
        raise HostedConfigurationError(
            "AGENTSWITCH_BASE_URL must be the instance base URL, not the /api/mcp endpoint."
        )

    return HostedEnvironment(
        instance=instance,
        base_url=base_url.rstrip("/"),
        token=token_value.strip(),
    )


def _with_plan(prefix: str, move_summary: str, suffix: str = "") -> str:
    """Embed the full plan when it fits the evidence limit, else a fingerprint of it."""

    full = f"{prefix}plan={move_summary}{suffix}"
    if len(full) <= MAX_EVIDENCE_LENGTH:
        return full
    digest = hashlib.sha256(move_summary.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}plan_sha={digest}{suffix}"


def _read_trace_summary(calls: list[ToolCallRecord]) -> str:
    failed_reads = sum(not call.succeeded for call in calls if not call.mutating)
    mutation_attempts = sum(call.mutating for call in calls)
    return (
        f"read_calls={sum(not call.mutating for call in calls)};"
        f"failed_reads={failed_reads};mutation_attempts={mutation_attempts}"
    )


def _goal_one_reason_code(result: AgentResult) -> str:
    """Map free-text planner diagnostics to a small non-sensitive reason code."""

    diagnostic = " ".join((result.summary, *result.ambiguities)).casefold()
    if "no 30-minute slot" in diagnostic or "all candidate intervals" in diagnostic:
        return "no_candidate_satisfies_constraints"
    if any(
        marker in diagnostic
        for marker in ("free/busy", "freebusy", "not shared", "visibility", "access evidence")
    ):
        return "availability_not_verifiable"
    if any(
        marker in diagnostic
        for marker in ("scheduling preference", "availability rule", "minimum notice")
    ):
        return "scheduling_constraints_unavailable"
    if "calendar" in diagnostic:
        return "calendar_scope_ambiguous"
    if any(marker in diagnostic for marker in ("plant head", "party", "identity")):
        return "participant_identity_ambiguous"
    return "planner_needs_clarification"


def _has_valid_goal_two_proposed_shift(outcome: object) -> bool:
    if not isinstance(outcome, dict):
        return False
    proposed_shift = outcome.get("shift")
    requested_shift = outcome.get("requested_shift")
    return (
        outcome.get("shift_was_proposed") is True
        and requested_shift == GOAL_TWO_DEFAULT_SHIFT
        and isinstance(proposed_shift, timedelta)
        and GOAL_TWO_DEFAULT_SHIFT
        <= proposed_shift
        <= GOAL_TWO_MAX_SHIFT_SEARCH
        and (proposed_shift - GOAL_TWO_DEFAULT_SHIFT)
        % GOAL_TWO_DEFAULT_SHIFT
        == timedelta(0)
        and outcome.get("alternative_shift_proposed")
        is (proposed_shift != requested_shift)
    )


def check_suryodaya_goal_one(
    tools: CalendarTool,
    *,
    now: datetime | None = None,
    secret_values: tuple[str, ...] = (),
    journal: HostedRunJournal | None = None,
) -> HostedCheck:
    """Verify the harness-owned, read-only candidate predicate."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc).replace(microsecond=0)
    search_end = current + _SEARCH_WINDOW
    task = AgentTask(
        GOAL_ONE,
        GOAL_ONE_TITLE,
        {
            "now": current,
            "start": current,
            "end": search_end,
            "fallback_availability_rule_name": SURYODAYA_FALLBACK_RULE_NAME,
        },
    )
    read_only_tools = ReadOnlyCalendarTool(tools)
    try:
        result = CalendarAgent(read_only_tools).run(task)
    except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
        if journal is not None:
            journal.record_execution(
                "suryodaya.goal1_read_only_candidate",
                result=None,
                tool_calls=read_only_tools.trace,
                error_category=type(exc).__name__,
            )
        trace = _read_trace_summary(tools.trace)
        return _task(
            task_id="suryodaya.goal1_read_only_candidate",
            title="Suryodaya read-only Goal 1 candidate check",
            passed=False,
            evidence=f"UNEVALUATED: planner stopped ({type(exc).__name__}); {trace}.",
            secret_values=secret_values,
        )

    if journal is not None:
        journal.record_execution(
            "suryodaya.goal1_read_only_candidate",
            result=result,
            tool_calls=read_only_tools.trace,
        )
    try:
        verification = verify_goal_one_candidate_against_reads(
            result,
            read_only_tools,
            now=current,
            search_start=current,
            search_end=search_end,
            prior_tool_calls=result.tool_calls,
        )
    except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
        if journal is not None:
            journal.record_observation(
                "suryodaya.goal1_read_only_candidate",
                observation={"verification_error_category": type(exc).__name__},
                tool_calls=tuple(read_only_tools.trace),
            )
        return _task(
            task_id="suryodaya.goal1_read_only_candidate",
            title="Suryodaya read-only Goal 1 candidate check",
            passed=False,
            evidence=(
                "UNEVALUATED: candidate state re-verification stopped "
                f"({type(exc).__name__}); {_read_trace_summary(read_only_tools.trace)}."
            ),
            secret_values=secret_values,
        )
    if verification.passed:
        evidence = (
            "HARNESS_PREDICATE: one independently re-verified, available "
            f"30-minute candidate; {_read_trace_summary(read_only_tools.trace)}."
        )
        return _task(
            task_id="suryodaya.goal1_read_only_candidate",
            title="Suryodaya read-only Goal 1 candidate check",
            passed=True,
            evidence=evidence,
            secret_values=secret_values,
        )

    evidence = (
        f"UNEVALUATED: planner status={result.status.value}; "
        f"candidate_slots={len(result.candidate_slots)}; "
        f"clarifications={len(result.ambiguities)}; "
        f"reason_code={_goal_one_reason_code(result) if result.status is not ResultStatus.PLANNED else 'candidate_reverification_failed'}; "
        f"verification_details={'; '.join(verification.details) or 'no valid candidate'}; "
        f"{_read_trace_summary(read_only_tools.trace)}."
    )
    return _task(
        task_id="suryodaya.goal1_read_only_candidate",
        title="Suryodaya read-only Goal 1 candidate check",
        passed=False,
        evidence=evidence,
        secret_values=secret_values,
    )


def check_suryodaya_goal_two(
    tools: CalendarTool,
    *,
    secret_values: tuple[str, ...] = (),
    journal: HostedRunJournal | None = None,
) -> HostedCheck:
    """Verify an exact-reference, read-only move plan that awaits confirmation."""

    task = AgentTask(
        GOAL_TWO,
        GOAL_TWO_TITLE,
        {
            "audit_number": GOAL_TWO_AUDIT_NUMBER,
            "audit_title": GOAL_TWO_AUDIT_TITLE,
            "max_shift_search": GOAL_TWO_MAX_SHIFT_SEARCH,
            "use_calendar_timezone_fallback": True,
        },
    )
    try:
        result = CalendarAgent(ReadOnlyCalendarTool(tools)).run(task)
    except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
        if journal is not None:
            journal.record_execution(
                "suryodaya.goal2_read_only_plan",
                result=None,
                tool_calls=tools.trace,
                error_category=type(exc).__name__,
            )
        return _task(
            task_id="suryodaya.goal2_read_only_plan",
            title="Suryodaya read-only Goal 2 plan",
            passed=False,
            evidence=(
                "UNEVALUATED: Goal 2 read-only planning stopped "
                f"({type(exc).__name__}); {_read_trace_summary(tools.trace)}."
            ),
            secret_values=secret_values,
        )

    if journal is not None:
        journal.record_execution(
            "suryodaya.goal2_read_only_plan",
            result=result,
            tool_calls=result.tool_calls,
        )
    trace = _read_trace_summary(result.tool_calls)
    if result.status is not ResultStatus.PLANNED:
        details = " ".join((result.summary, *result.ambiguities))
        return _task(
            task_id="suryodaya.goal2_read_only_plan",
            title="Suryodaya read-only Goal 2 plan",
            passed=False,
            evidence=(
                f"Goal 2 did not produce a plan ({result.status.value}); "
                f"moves=0; {details}; {trace}."
            ),
            secret_values=secret_values,
        )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)
    reference = result.audit_reference
    exact_reference = (
        reference is not None
        and len(reference.candidates) == 1
        and reference.candidates[0].id == GOAL_TWO_AUDIT_NUMBER
        and reference.candidates[0].title == GOAL_TWO_AUDIT_TITLE
        and isinstance(reference.candidates[0].status, str)
        and reference.candidates[0].status.casefold() == "confirmed"
        and isinstance(reference.candidates[0].calendar_id, str)
        and bool(reference.candidates[0].calendar_id.strip())
    )
    outcome = result.claimed_outcome
    confirmation_required = (
        isinstance(outcome, dict)
        and outcome.get("confirmation_required") is True
        and isinstance(outcome.get("confirmation_prompt"), str)
        and bool(outcome["confirmation_prompt"].strip())
    )
    proposed_shift = outcome.get("shift") if isinstance(outcome, dict) else None
    proposed_shift_minutes = (
        int(proposed_shift.total_seconds() // 60)
        if isinstance(proposed_shift, timedelta)
        else 0
    )
    mutation_free = not any(call.mutating for call in result.tool_calls)
    move_summary = ";".join(
        f"{move.event_id}:{move.proposed_start_at}>{move.proposed_end_at}"
        for move in result.rescheduling_plan
    )
    evidence = _with_plan(
        "HARNESS_PREDICATE: "
        f"audit={GOAL_TWO_AUDIT_NUMBER};calendar="
        f"{reference.candidates[0].calendar_id if exact_reference else 'unverified'};"
        f"moves={len(result.rescheduling_plan)};requested_shift=30m;"
        f"proposed_shift={proposed_shift_minutes}m;"
        f"confirmation_required={str(confirmation_required).lower()};"
        "confirm_or_change_shift=true;"
        "events_moved=false;"
        f"mutation_attempts={sum(call.mutating for call in result.tool_calls)};",
        move_summary,
    )
    passed = (
        verification.passed
        and exact_reference
        and confirmation_required
        and _has_valid_goal_two_proposed_shift(outcome)
        and outcome.get("moved") is False
        and mutation_free
        and bool(result.rescheduling_plan)
    )
    if len(evidence) > MAX_EVIDENCE_LENGTH:
        passed = False
        evidence = (
            "UNEVALUATED: Goal 2 plan is too large to fit the results evidence "
            "limit; a complete plan was not serialized."
        )
    return _task(
        task_id="suryodaya.goal2_read_only_plan",
        title="Suryodaya read-only Goal 2 plan",
        passed=passed,
        evidence=evidence,
        secret_values=secret_values,
    )


def check_keystone_goal_one(
    adapter: KeystoneCalendarAdapter,
    *,
    now: datetime | None = None,
    secret_values: tuple[str, ...] = (),
    journal: HostedRunJournal | None = None,
) -> HostedCheck:
    """Apply the same harness-owned candidate predicate to Keystone."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc).replace(microsecond=0)
    search_end = current + _SEARCH_WINDOW
    result = run_keystone_goal(
        AgentTask(
            GOAL_ONE,
            GOAL_ONE_TITLE,
            {
                "now": current,
                "start": current,
                "end": search_end,
            },
        ),
        adapter,
    )
    if journal is not None:
        journal.record_execution(
            "keystone.goal1_read_only_candidate",
            result=result,
            tool_calls=result.tool_calls,
        )
    verification_tools = KeystoneCalendarTool(adapter)
    try:
        verification = verify_goal_one_candidate_against_reads(
            result,
            verification_tools,
            now=current,
            search_start=current,
            search_end=search_end,
            prior_tool_calls=result.tool_calls,
        )
    except KeystoneGoalIntegrationError as exc:
        if journal is not None:
            journal.record_observation(
                "keystone.goal1_read_only_candidate",
                observation={"verification_error_category": exc.category},
                tool_calls=tuple(result.tool_calls),
            )
        return _task(
            task_id="keystone.goal1_read_only_candidate",
            title="Keystone read-only Goal 1 candidate check",
            passed=False,
            evidence=(
                "UNEVALUATED: candidate state re-verification stopped "
                f"({exc.category}); "
                f"{_read_trace_summary([*result.tool_calls, *verification_tools.trace])}."
            ),
            secret_values=secret_values,
        )
    if verification.passed:
        evidence = (
            "HARNESS_PREDICATE: one independently re-verified, available "
            f"30-minute candidate; "
            f"{_read_trace_summary([*result.tool_calls, *verification_tools.trace])}."
        )
    else:
        evidence = (
            "UNEVALUATED: Keystone Goal 1 planner "
            f"status={result.status.value}; candidate_slots={len(result.candidate_slots)}; "
            f"clarifications={len(result.ambiguities)}; "
            f"reason_code={_goal_one_reason_code(result) if result.status is not ResultStatus.PLANNED else 'candidate_reverification_failed'}; "
            f"verification_details={'; '.join(verification.details) or 'no valid candidate'}; "
            f"{_read_trace_summary([*result.tool_calls, *verification_tools.trace])}; "
            "candidate availability requires complete verified identity, preference, "
            "availability-rule, and visible free/busy evidence."
        )
    return _task(
        task_id="keystone.goal1_read_only_candidate",
        title="Keystone read-only Goal 1 candidate check",
        passed=verification.passed,
        evidence=evidence,
        secret_values=secret_values,
    )


def check_keystone_goal_two(
    adapter: KeystoneCalendarAdapter,
    *,
    secret_values: tuple[str, ...] = (),
    journal: HostedRunJournal | None = None,
) -> HostedCheck:
    """Verify one uniquely discoverable audit plan without executing mutations."""

    discovery_tools = KeystoneCalendarTool(adapter)
    try:
        audit_events = discovery_tools.list_events(
            filters={"search": KEYSTONE_AUDIT_TITLE},
            limit=1000,
        )
    except KeystoneGoalIntegrationError as exc:
        if journal is not None:
            journal.record_execution(
                "keystone.goal2_read_only_plan",
                result=None,
                tool_calls=discovery_tools.trace,
                error_category=exc.category,
            )
        return _task(
            task_id="keystone.goal2_read_only_plan",
            title="Keystone read-only Goal 2 plan",
            passed=False,
            evidence=(
                "UNEVALUATED: audit discovery stopped "
                f"({exc.category}); {_read_trace_summary(discovery_tools.trace)}."
            ),
            secret_values=secret_values,
        )

    audit_candidates = [
        event
        for event in audit_events
        if isinstance(event.get("title"), str)
        and event["title"].strip().casefold()
        == KEYSTONE_AUDIT_TITLE.casefold()
    ]
    audit_candidates.sort(key=lambda event: str(event.get("start_at")))
    audit_candidates = audit_candidates[:1]
    if journal is not None:
        journal.record_observation(
            "keystone.goal2_read_only_plan",
            observation={
                "audit_search_result_count": len(audit_events),
                "exact_title_match_count": len(audit_candidates),
            },
            tool_calls=tuple(discovery_tools.trace),
        )
    if len(audit_candidates) != 1:
        return _task(
            task_id="keystone.goal2_read_only_plan",
            title="Keystone read-only Goal 2 plan",
            passed=False,
            evidence=(
                "UNEVALUATED: Keystone audit search did not resolve exactly one "
                f"candidate (found={len(audit_candidates)}); "
                f"{_read_trace_summary(discovery_tools.trace)}."
            ),
            secret_values=secret_values,
        )

    audit_id = audit_candidates[0].get("id")
    calendar_id = audit_candidates[0].get("calendar_id")
    if (
        not isinstance(audit_id, str)
        or not audit_id.strip()
        or not isinstance(calendar_id, str)
        or not calendar_id.strip()
    ):
        return _task(
            task_id="keystone.goal2_read_only_plan",
            title="Keystone read-only Goal 2 plan",
            passed=False,
            evidence=(
                "UNEVALUATED: the unique Keystone audit candidate has no validated "
                f"identity or owning calendar; {_read_trace_summary(discovery_tools.trace)}."
            ),
            secret_values=secret_values,
        )
    audit_status = audit_candidates[0].get("status")
    if not isinstance(audit_status, str) or audit_status.strip().casefold() != "confirmed":
        return _task(
            task_id="keystone.goal2_read_only_plan",
            title="Keystone read-only Goal 2 plan",
            passed=False,
            evidence=(
                "UNEVALUATED: the unique Keystone audit candidate is not "
                f"verified as confirmed; {_read_trace_summary(discovery_tools.trace)}."
            ),
            secret_values=secret_values,
        )

    result = run_keystone_goal(
        AgentTask(
            GOAL_TWO,
            "Move everything after the audit",
            {
                "calendar_id": calendar_id,
                "audit_event_id": audit_id,
                "audit_title": KEYSTONE_AUDIT_TITLE,
                "max_shift_search": GOAL_TWO_MAX_SHIFT_SEARCH,
            },
        ),
        adapter,
    )
    trace = [*discovery_tools.trace, *result.tool_calls]
    if journal is not None:
        journal.record_execution(
            "keystone.goal2_read_only_plan",
            result=result,
            tool_calls=trace,
        )
    trace_summary = _read_trace_summary(trace)
    if result.status is not ResultStatus.PLANNED:
        return _task(
            task_id="keystone.goal2_read_only_plan",
            title="Keystone read-only Goal 2 plan",
            passed=False,
            evidence=(
                f"Keystone Goal 2 did not produce a plan ({result.status.value}); "
                f"moves=0; clarifications={len(result.ambiguities)}; {trace_summary}."
            ),
            secret_values=secret_values,
        )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)
    reference = result.audit_reference
    exact_calendar = (
        reference is not None
        and len(reference.candidates) == 1
        and reference.candidates[0].id == audit_id
        and reference.candidates[0].title == KEYSTONE_AUDIT_TITLE
        and reference.candidates[0].calendar_id == calendar_id
        and isinstance(reference.candidates[0].status, str)
        and reference.candidates[0].status.casefold() == "confirmed"
    )
    outcome = result.claimed_outcome
    confirmation_required = (
        isinstance(outcome, dict)
        and outcome.get("confirmation_required") is True
        and isinstance(outcome.get("confirmation_prompt"), str)
        and bool(outcome["confirmation_prompt"].strip())
    )
    proposed_shift = outcome.get("shift") if isinstance(outcome, dict) else None
    proposed_shift_minutes = (
        int(proposed_shift.total_seconds() // 60)
        if isinstance(proposed_shift, timedelta)
        else 0
    )
    mutation_free = not any(call.mutating for call in result.tool_calls)
    move_summary = ";".join(
        f"{move.event_id}:{move.proposed_start_at}>{move.proposed_end_at}"
        for move in result.rescheduling_plan
    )
    evidence = _with_plan(
        "HARNESS_PREDICATE: "
        f"calendar_resolved={str(exact_calendar).lower()};"
        f"moves={len(result.rescheduling_plan)};"
        f"requested_shift=30m;proposed_shift={proposed_shift_minutes}m;"
        f"confirmation_required={str(confirmation_required).lower()};"
        "events_moved=false;"
        f"mutation_attempts={sum(call.mutating for call in trace)};",
        move_summary,
        f";{trace_summary}",
    )
    passed = (
        verification.passed
        and exact_calendar
        and confirmation_required
        and _has_valid_goal_two_proposed_shift(outcome)
        and outcome.get("moved") is False
        and mutation_free
        and bool(result.rescheduling_plan)
    )
    if len(evidence) > MAX_EVIDENCE_LENGTH:
        passed = False
        evidence = (
            "UNEVALUATED: Keystone Goal 2 plan is too large to fit the results "
            "evidence limit; a complete plan was not serialized."
        )
    return _task(
        task_id="keystone.goal2_read_only_plan",
        title="Keystone read-only Goal 2 plan",
        passed=passed,
        evidence=evidence,
        secret_values=secret_values,
    )


def run_hosted(
    environment: Mapping[str, object],
    *,
    output_path: str | Path,
    now: datetime | None = None,
) -> tuple[Path, tuple[HostedCheck, ...]]:
    """Run the platform-selected instance and write its result document."""

    raw_token = environment.get("AGENTSWITCH_TOKEN")
    secret_values = (raw_token.strip(),) if isinstance(raw_token, str) and raw_token.strip() else ()
    try:
        config = load_hosted_environment(environment)
    except HostedConfigurationError as exc:
        tasks = (
            _task(
                task_id="harness.configuration",
                title="Hosted harness configuration",
                passed=False,
                evidence=f"UNEVALUATED: {type(exc).__name__}: {exc}",
                secret_values=secret_values,
            ),
        )
        summary = (
            "Hosted harness configuration could not be validated; "
            "goal_tasks_not_run=true; official_evaluator=not_invoked."
        )
    else:
        journal = HostedRunJournal(
            Path(output_path).parent / ".harness-artifacts" / "runs",
            instance=config.instance,
            secret_values=secret_values,
        )
        if config.instance == SURYODAYA_INSTANCE:
            try:
                client_config = AgentSwitchConfig(base_url=config.base_url, token=config.token)
                tools = AgentSwitchCalendarTools(AgentSwitchClient(client_config))
                tasks = (
                    _journaled_check(
                        journal,
                        task_id="suryodaya.goal1_read_only_candidate",
                        title="Suryodaya read-only Goal 1 candidate check",
                        goal=GOAL_ONE,
                        run=lambda task_journal: check_suryodaya_goal_one(
                            tools,
                            now=now,
                            secret_values=secret_values,
                            journal=task_journal,
                        ),
                    ),
                    _journaled_check(
                        journal,
                        task_id="suryodaya.goal2_read_only_plan",
                        title="Suryodaya read-only Goal 2 plan",
                        goal=GOAL_TWO,
                        run=lambda task_journal: check_suryodaya_goal_two(
                            tools,
                            secret_values=secret_values,
                            journal=task_journal,
                        ),
                    ),
                )
            except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
                tasks = (
                    _task(
                        task_id="harness.execution",
                        title="Suryodaya harness execution",
                        passed=False,
                        evidence=(
                            "UNEVALUATED: hosted harness initialization or evaluation "
                            f"stopped ({type(exc).__name__})."
                        ),
                        secret_values=secret_values,
                    ),
                )
        elif config.instance == KEYSTONE_INSTANCE:
            try:
                keystone_config = KeystoneConfig.for_hosted_runner(
                    base_url=config.base_url,
                    token=config.token,
                )
                client = KeystoneClient(
                    keystone_config,
                    enable_hosted_read_transport=True,
                )
                journal.begin(
                    task_id="keystone.mcp_tools_list",
                    title="Keystone MCP tools/list and read tools",
                    goal="mcp.tools.list",
                )
                try:
                    discovery = client.list_tools()
                except (KeystoneClientError, KeystoneConfigurationError) as exc:
                    journal.record_execution(
                        "keystone.mcp_tools_list",
                        result=None,
                        tool_calls=(),
                        error_category=type(exc).__name__,
                    )
                    raise
                available_operations = {tool.name for tool in discovery.tools}
                missing_operations = sorted(
                    RESOURCE_READ_OPERATIONS - available_operations
                )
                journal.record_observation(
                    "keystone.mcp_tools_list",
                    observation={
                        "discovered_tool_count": len(discovery.tools),
                        "required_read_tools_missing": missing_operations,
                    },
                )
                connection_check = _task(
                    task_id="keystone.mcp_tools_list",
                    title="Keystone MCP tools/list and read tools",
                    passed=not missing_operations,
                    evidence=(
                        f"discovered_tools={len(discovery.tools)};"
                        f"required_read_tools_missing={','.join(missing_operations) or 'none'}."
                    ),
                    secret_values=secret_values,
                )
                journal.finish(connection_check)
                adapter = KeystoneCalendarAdapter(client)
                if missing_operations:
                    goal_checks = (
                        _journaled_check(
                            journal,
                            task_id="keystone.goal1_read_only_candidate",
                            title="Keystone read-only Goal 1 candidate check",
                            goal=GOAL_ONE,
                            run=lambda task_journal: _record_unavailable_goal(
                                task_journal,
                                task_id="keystone.goal1_read_only_candidate",
                                title="Keystone read-only Goal 1 candidate check",
                                goal=GOAL_ONE,
                                missing_operations=missing_operations,
                                secret_values=secret_values,
                            ),
                        ),
                        _journaled_check(
                            journal,
                            task_id="keystone.goal2_read_only_plan",
                            title="Keystone read-only Goal 2 plan",
                            goal=GOAL_TWO,
                            run=lambda task_journal: _record_unavailable_goal(
                                task_journal,
                                task_id="keystone.goal2_read_only_plan",
                                title="Keystone read-only Goal 2 plan",
                                goal=GOAL_TWO,
                                missing_operations=missing_operations,
                                secret_values=secret_values,
                            ),
                        ),
                    )
                else:
                    goal_checks = (
                        _journaled_check(
                            journal,
                            task_id="keystone.goal1_read_only_candidate",
                            title="Keystone read-only Goal 1 candidate check",
                            goal=GOAL_ONE,
                            run=lambda task_journal: check_keystone_goal_one(
                                adapter,
                                now=now,
                                secret_values=secret_values,
                                journal=task_journal,
                            ),
                        ),
                        _journaled_check(
                            journal,
                            task_id="keystone.goal2_read_only_plan",
                            title="Keystone read-only Goal 2 plan",
                            goal=GOAL_TWO,
                            run=lambda task_journal: check_keystone_goal_two(
                                adapter,
                                secret_values=secret_values,
                                journal=task_journal,
                            ),
                        ),
                    )
                tasks = (connection_check, *goal_checks)
            except (KeystoneClientError, KeystoneConfigurationError) as exc:
                tasks = (
                    _task(
                        task_id="harness.execution",
                        title="Keystone harness execution",
                        passed=False,
                        evidence=(
                            "UNEVALUATED: Keystone harness initialization or evaluation "
                            f"stopped ({type(exc).__name__})."
                        ),
                        secret_values=secret_values,
                    ),
                )
        else:
            raise AssertionError("load_hosted_environment must reject unsupported instances")
        passed_count = sum(task.passed for task in tasks)
        summary = (
            f"{config.instance}: {passed_count}/{len(tasks)} harness checks passed; "
            f"run_id={journal.run_id};official_evaluator=not_invoked."
        )

    written_path = write_hosted_results(
        output_path,
        tasks,
        summary=summary,
        secret_values=secret_values,
    )
    return written_path, tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harnesses.suryodaya_harness.hosted_runner")
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    args = parser.parse_args(argv)
    try:
        output_path, tasks = run_hosted(os.environ, output_path=args.results)
    except (OSError, ServerRunnerError, TypeError, ValueError) as exc:
        print(f"Hosted harness error: {type(exc).__name__}.")
        return 1

    passed_count = sum(task.passed for task in tasks)
    print(
        f"Hosted harness wrote {output_path}; harness_checks_passed="
        f"{passed_count}/{len(tasks)}; official_evaluator=not_invoked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
