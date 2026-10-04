"""Provider-neutral deterministic evaluation contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib

from .models import AgentResult, ResultStatus, VerificationResult
from .persistence import EvaluationEvidence, PersistedRun, RawRunJournal
from .runtime import RawRun


class Verdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    UNEVALUATED = "unevaluated"


@dataclass(frozen=True)
class EvaluationState:
    """Explicit evaluator evidence supplied independently of the agent run."""

    predicate_available: bool = False
    predicate_passed: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_available, bool):
            raise ValueError("predicate_available must be boolean")
        if self.predicate_passed is not None and not isinstance(self.predicate_passed, bool):
            raise ValueError("predicate_passed must be boolean or None")
        if not self.predicate_available and self.predicate_passed is not None:
            raise ValueError("predicate_passed requires an available predicate")


@dataclass(frozen=True)
class EvaluationResult:
    verdict: Verdict
    scorer_version: str
    run_id: str
    reason: str

    @property
    def passed(self) -> bool:
        """Only approve is a passing evaluation."""

        return self.verdict is Verdict.APPROVE


def _raw_run_fingerprint(raw_run: RawRun) -> str:
    serialized = RawRunJournal.dumps(raw_run)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _candidate_slot_present(result: AgentResult | None) -> bool:
    return bool(
        result is not None
        and result.goal == "calendar.find_30_minutes"
        and len(result.candidate_slots) == 1
    )


def build_evaluation_evidence(
    raw_run: RawRun,
    verification: VerificationResult,
    *,
    expected_status: ResultStatus,
) -> EvaluationEvidence:
    """Convert Harness verification into explicit, JSON-safe persisted evidence."""

    result = raw_run.result
    return EvaluationEvidence(
        expected_status=expected_status.value,
        run_completed=(
            raw_run.termination_reason.value == "succeeded"
            and raw_run.error is None
            and result is not None
            and result.status is ResultStatus.PLANNED
        ),
        verification_passed=verification.passed,
        outcome_ok=verification.outcome_ok,
        integrity_ok=verification.integrity_ok,
        tool_trace_consistent=(result is not None and result.tool_calls == list(raw_run.tool_trace)),
        candidate_slot_present=_candidate_slot_present(result),
        raw_run_fingerprint=_raw_run_fingerprint(raw_run),
    )


def _approval_errors(persisted: PersistedRun) -> list[str]:
    raw_run = persisted.raw_run
    evidence = persisted.evaluation_evidence
    if evidence is None:
        return ["No persisted evaluation evidence is available."]

    errors: list[str] = []
    if evidence.raw_run_fingerprint != _raw_run_fingerprint(raw_run):
        errors.append("Persisted evaluation evidence does not match the raw run.")
    if evidence.expected_status != ResultStatus.PLANNED.value:
        errors.append("Approval evidence must expect a planned result.")
    if not evidence.run_completed:
        errors.append("The run did not complete with PLANNED/SUCCEEDED semantics.")
    if not evidence.verification_passed or not evidence.outcome_ok:
        errors.append("Persisted result verification did not pass.")
    if not evidence.integrity_ok:
        errors.append("Persisted tool evidence is not mutation-free and successful.")
    if not evidence.tool_trace_consistent:
        errors.append("Persisted result and raw tool traces do not agree.")
    if evidence.candidate_slot_present != _candidate_slot_present(raw_run.result):
        errors.append("Persisted candidate-slot evidence is inconsistent with the result.")

    result = raw_run.result
    if raw_run.termination_reason.value != "succeeded" or raw_run.error is not None:
        errors.append("The raw run does not record a successful termination.")
    if result is None or result.status is not ResultStatus.PLANNED:
        errors.append("The raw run does not contain a planned result.")
    elif result.tool_calls != list(raw_run.tool_trace):
        errors.append("The result tool calls do not match the raw tool trace.")
    if any(call.mutating or not call.succeeded for call in raw_run.tool_trace):
        errors.append("The raw tool trace contains a mutation or failed call.")
    if result is not None and result.goal == "calendar.find_30_minutes" and not _candidate_slot_present(result):
        errors.append("The planned find-30-minutes result lacks candidate-slot evidence.")
    return errors


class DeterministicScorer:
    """Score persisted evidence without invoking an agent, model, or tools."""

    def __init__(self, version: str = "1") -> None:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("scorer version must be a non-empty string")
        self.version = version

    def score(
        self,
        evidence: PersistedRun | RawRun,
        *,
        state: EvaluationState | None = None,
    ) -> EvaluationResult:
        if isinstance(evidence, PersistedRun):
            raw_run = evidence.raw_run
        elif isinstance(evidence, RawRun):
            raw_run = evidence
        else:
            raise TypeError("scorer evidence must be a PersistedRun or RawRun")

        if state is None or not state.predicate_available:
            return EvaluationResult(
                verdict=Verdict.UNEVALUATED,
                scorer_version=self.version,
                run_id=raw_run.run_id,
                reason="The required live evaluation predicate is unavailable.",
            )
        if state.predicate_passed is None:
            return EvaluationResult(
                verdict=Verdict.UNEVALUATED,
                scorer_version=self.version,
                run_id=raw_run.run_id,
                reason="The required evaluation predicate has no result.",
            )
        if state.predicate_passed and isinstance(evidence, PersistedRun):
            approval_errors = _approval_errors(evidence)
            if approval_errors:
                return EvaluationResult(
                    verdict=Verdict.REVISE,
                    scorer_version=self.version,
                    run_id=raw_run.run_id,
                    reason="Persisted evidence failed approval checks: " + " ".join(approval_errors),
                )
            return EvaluationResult(
                verdict=Verdict.APPROVE,
                scorer_version=self.version,
                run_id=raw_run.run_id,
                reason="The supplied evaluation predicate passed.",
            )
        return EvaluationResult(
            verdict=Verdict.REVISE,
            scorer_version=self.version,
            run_id=raw_run.run_id,
            reason="The supplied evaluation predicate failed.",
        )


def score_persisted_run(
    evidence: PersistedRun,
    *,
    state: EvaluationState | None = None,
    scorer_version: str = "1",
) -> EvaluationResult:
    return DeterministicScorer(scorer_version).score(evidence, state=state)