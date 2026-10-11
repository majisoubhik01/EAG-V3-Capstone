"""Offline Release 8.1 outcome and results.json contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from collections.abc import Iterable
from typing import Any


MAX_TASKS = 200
MAX_EVIDENCE_LENGTH = 500
SUPPORTED_GOALS = frozenset({
    "calendar.find_30_minutes",
    "calendar.move_after_audit",
})

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?P<prefix>\b(?:authorization|api[_ -]?key|token|password|passwd|cookie|secret|credential)\s*[:=]\s*(?:bearer\s+)?)"
    r"(?P<value>[^,;\r\n]+)",
    re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"(?P<prefix>\bbearer\s+)(?P<value>[^\s,;]+)", re.IGNORECASE)
_API_KEY_VALUE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b")


class ResultSchemaError(ValueError):
    """Raised when an official results document is malformed."""


class OutcomeStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNEVALUATED = "unevaluated"


@dataclass(frozen=True)
class PredicateEvaluation:
    """Typed result from a future authoritative goal evaluator."""

    goal: str
    passed: bool

    def __post_init__(self) -> None:
        if self.goal not in SUPPORTED_GOALS:
            raise ValueError("predicate evaluation goal is unsupported")
        if type(self.passed) is not bool:
            raise TypeError("predicate evaluation result must be a boolean")


@dataclass(frozen=True)
class GoalOutcome:
    """An internal outcome that cannot pass without typed predicate evidence."""

    status: OutcomeStatus
    reason: str
    goal: str | None = None
    predicate_evaluation: PredicateEvaluation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, OutcomeStatus):
            raise TypeError("status must be an OutcomeStatus")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.goal is not None and self.goal not in SUPPORTED_GOALS:
            raise ValueError("goal is unsupported")
        if self.predicate_evaluation is not None and not isinstance(
            self.predicate_evaluation, PredicateEvaluation
        ):
            raise TypeError("predicate_evaluation must be a PredicateEvaluation or None")

        if self.status is OutcomeStatus.PASSED:
            if (
                self.goal is None
                or self.predicate_evaluation is None
                or self.predicate_evaluation.goal != self.goal
                or self.predicate_evaluation.passed is not True
            ):
                raise ValueError("PASSED requires a confirmed predicate evaluation for its goal")
        elif self.status is OutcomeStatus.FAILED:
            if (
                self.goal is None
                or self.predicate_evaluation is None
                or self.predicate_evaluation.goal != self.goal
                or self.predicate_evaluation.passed is not False
            ):
                raise ValueError("FAILED requires a confirmed predicate evaluation for its goal")
        elif self.predicate_evaluation is not None:
            raise ValueError("UNEVALUATED cannot contain a predicate evaluation")


@dataclass(frozen=True)
class ResultTask:
    """Internal task evidence plus its official-schema projection."""

    task_id: str
    title: str
    outcome: GoalOutcome
    evidence: str
    score: float | None = None
    secret_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.outcome, GoalOutcome):
            raise TypeError("outcome must be a GoalOutcome")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("evidence must be a non-empty string")
        if any(not isinstance(value, str) or not value for value in self.secret_values):
            raise ValueError("secret_values must contain non-empty strings")
        _validate_score(self.score, "score")

    def official_record(self) -> dict[str, Any]:
        evidence = self.evidence.strip()
        if self.outcome.status is OutcomeStatus.UNEVALUATED and not evidence.startswith("UNEVALUATED:"):
            evidence = f"UNEVALUATED: {evidence}"
        evidence = sanitize_evidence(evidence, self.secret_values)[:MAX_EVIDENCE_LENGTH]
        return {
            "id": self.task_id.strip(),
            "title": self.title.strip(),
            "passed": self.outcome.status is OutcomeStatus.PASSED,
            "score": self.score if self.score is not None else (
                1.0 if self.outcome.status is OutcomeStatus.PASSED else 0.0
            ),
            "evidence": evidence,
        }


def _validate_score(value: object, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between 0 and 1")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")


def sanitize_evidence(evidence: str, secret_values: Iterable[str] = ()) -> str:
    """Redact known secrets and common credential-shaped values.

    This is defensive filtering, not a guarantee that arbitrary secret formats
    can be detected. Callers should pass known runtime secret values whenever
    they are available, and evidence should contain only necessary diagnostics.
    """

    sanitized = evidence
    for secret in sorted({value for value in secret_values if value}, key=len, reverse=True):
        sanitized = sanitized.replace(secret, "[REDACTED]")

    def redact_assignment(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}[REDACTED]"

    def redact_bearer(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}[REDACTED]"

    sanitized = _CREDENTIAL_ASSIGNMENT.sub(redact_assignment, sanitized)
    sanitized = _BEARER_VALUE.sub(redact_bearer, sanitized)
    return _API_KEY_VALUE.sub("[REDACTED]", sanitized)


def validate_results(document: object) -> None:
    """Validate the documented Release 8.1 results shape."""

    if not isinstance(document, dict):
        raise ResultSchemaError("results must be an object")
    if set(document) != {"tasks", "summary"}:
        raise ResultSchemaError("results must contain only tasks and summary")

    tasks = document["tasks"]
    summary = document["summary"]
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
        raise ResultSchemaError("tasks must contain between 1 and 200 items")
    if not isinstance(summary, str) or not summary.strip():
        raise ResultSchemaError("summary must be a non-empty string")
    if "\n" in summary or "\r" in summary:
        raise ResultSchemaError("summary must be a single line")

    task_ids: set[str] = set()
    allowed_task_keys = {"id", "title", "passed", "score", "evidence"}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ResultSchemaError(f"tasks[{index}] must be an object")
        if set(task) - allowed_task_keys:
            raise ResultSchemaError(f"tasks[{index}] contains unsupported fields")
        for key in ("id", "title", "passed", "evidence"):
            if key not in task:
                raise ResultSchemaError(f"tasks[{index}] is missing {key}")

        task_id = task["id"]
        if not isinstance(task_id, str) or not task_id.strip():
            raise ResultSchemaError(f"tasks[{index}].id must be non-empty")
        normalized_id = task_id.strip()
        if normalized_id in task_ids:
            raise ResultSchemaError(f"duplicate task id: {normalized_id}")
        task_ids.add(normalized_id)

        if not isinstance(task["title"], str) or not task["title"].strip():
            raise ResultSchemaError(f"tasks[{index}].title must be non-empty")
        if type(task["passed"]) is not bool:
            raise ResultSchemaError(f"tasks[{index}].passed must be a boolean")
        if "score" in task:
            try:
                _validate_score(task["score"], f"tasks[{index}].score")
            except ValueError as exc:
                raise ResultSchemaError(str(exc)) from exc
        evidence = task["evidence"]
        if not isinstance(evidence, str) or not evidence.strip():
            raise ResultSchemaError(f"tasks[{index}].evidence must be non-empty")
        if len(evidence) > MAX_EVIDENCE_LENGTH:
            raise ResultSchemaError(f"tasks[{index}].evidence is too long")


def serialize_results(tasks: Iterable[ResultTask], summary: str) -> str:
    """Serialize internal outcomes into the official JSON document."""

    task_list = list(tasks)
    if any(not isinstance(task, ResultTask) for task in task_list):
        raise TypeError("tasks must contain ResultTask values")
    document = {
        "tasks": [task.official_record() for task in task_list],
        "summary": summary,
    }
    validate_results(document)
    return json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False)


def parse_results(serialized: str) -> dict[str, Any]:
    """Parse and validate an official results document without side effects."""

    if not isinstance(serialized, str):
        raise TypeError("serialized results must be a string")
    try:
        document = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ResultSchemaError("results are not valid JSON") from exc
    validate_results(document)
    return document