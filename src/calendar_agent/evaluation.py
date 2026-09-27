"""Provider-neutral deterministic evaluation contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .persistence import PersistedRun
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
        if state.predicate_passed:
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