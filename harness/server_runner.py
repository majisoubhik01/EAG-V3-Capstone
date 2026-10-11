"""Offline Release 8.1 server-runner shell.

This module deliberately stops before Calendar execution, model calls, and MCP
requests. A future authoritative evaluator is the only dependency that may
produce a goal predicate result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping, Sequence
from typing import Protocol
from urllib.parse import urlsplit

from .results import (
    GoalOutcome,
    OutcomeStatus,
    PredicateEvaluation,
    ResultTask,
    SUPPORTED_GOALS,
    serialize_results,
)


GOAL_TITLES = {
    "calendar.find_30_minutes": "Find 30 minutes with the plant head",
    "calendar.move_after_audit": "Move everything after the audit",
}
_REQUIRED_ENVIRONMENT = (
    "AGENTSWITCH_BASE_URL",
    "AGENTSWITCH_TOKEN",
    "AGENTSWITCH_INSTANCE",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
)
_SECRET_ENVIRONMENT_KEYS = {
    "AGENTSWITCH_TOKEN",
    "OPENAI_API_KEY",
}


class ServerRunnerError(RuntimeError):
    """Raised when the offline runner cannot produce a safe result artifact."""


class ServerConfigurationError(ValueError):
    """Raised when injected Release 8.1 environment data is invalid."""


class EvaluatorContractError(ValueError):
    """Raised when an evaluator violates the typed goal-evaluation boundary."""


class GoalEvaluator(Protocol):
    """Future authoritative evaluator boundary; absent by default in Phase 2."""

    def evaluate(self, goal: str) -> PredicateEvaluation | None: ...


@dataclass(frozen=True)
class ServerEnvironment:
    """Validated server settings with secret fields excluded from repr output."""

    agentswitch_base_url: str
    agentswitch_token: str = field(repr=False)
    agentswitch_instance: str
    openai_base_url: str
    openai_api_key: str = field(repr=False)
    openai_model: str

    def secret_values(self) -> tuple[str, ...]:
        return (self.agentswitch_token, self.openai_api_key)


def _require_text(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ServerConfigurationError(f"{key} must be set")
    return value.strip()


def _validate_url(values: Mapping[str, object], key: str) -> str:
    value = _require_text(values, key).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ServerConfigurationError(f"{key} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ServerConfigurationError(f"{key} must not contain URL credentials")
    return value


def _known_secret_values(values: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(values, Mapping):
        return ()
    return tuple(
        value.strip()
        for key, value in values.items()
        if key in _SECRET_ENVIRONMENT_KEYS
        and isinstance(value, str)
        and value.strip()
    )


def load_environment(values: Mapping[str, object]) -> ServerEnvironment:
    """Validate an injected environment mapping without reading files or globals."""

    if not isinstance(values, Mapping):
        raise TypeError("environment must be a mapping")
    for key in _REQUIRED_ENVIRONMENT:
        if key not in values:
            raise ServerConfigurationError(f"{key} must be set")

    return ServerEnvironment(
        agentswitch_base_url=_validate_url(values, "AGENTSWITCH_BASE_URL"),
        agentswitch_token=_require_text(values, "AGENTSWITCH_TOKEN"),
        agentswitch_instance=_require_text(values, "AGENTSWITCH_INSTANCE"),
        openai_base_url=_validate_url(values, "OPENAI_BASE_URL"),
        openai_api_key=_require_text(values, "OPENAI_API_KEY"),
        openai_model=_require_text(values, "OPENAI_MODEL"),
    )


def _unevaluated_outcome(goal: str, reason: str) -> GoalOutcome:
    return GoalOutcome(OutcomeStatus.UNEVALUATED, reason, goal=goal)


def evaluate_goal(
    goal: str,
    *,
    evaluator: GoalEvaluator | None = None,
) -> GoalOutcome:
    """Convert only a typed, goal-matching evaluator result into an outcome."""

    if goal not in SUPPORTED_GOALS:
        raise EvaluatorContractError("unsupported Calendar goal")
    if evaluator is None:
        return _unevaluated_outcome(
            goal,
            "No authoritative evaluator is configured; the Calendar goal was not evaluated.",
        )

    try:
        evaluation = evaluator.evaluate(goal)
    except Exception as exc:
        return _unevaluated_outcome(
            goal,
            f"The authoritative evaluator was unavailable ({type(exc).__name__}).",
        )
    if evaluation is None:
        return _unevaluated_outcome(
            goal,
            "The authoritative evaluator returned no predicate result.",
        )
    if not isinstance(evaluation, PredicateEvaluation):
        raise EvaluatorContractError("evaluator must return PredicateEvaluation or None")
    if evaluation.goal != goal:
        raise EvaluatorContractError("evaluator returned a result for a different goal")
    return GoalOutcome(
        OutcomeStatus.PASSED if evaluation.passed else OutcomeStatus.FAILED,
        "The authoritative predicate was evaluated.",
        goal=goal,
        predicate_evaluation=evaluation,
    )


def build_goal_tasks(
    *,
    evaluator: GoalEvaluator | None = None,
    secret_values: Sequence[str] = (),
) -> tuple[ResultTask, ...]:
    """Build safe Calendar task records without executing Calendar work."""

    tasks: list[ResultTask] = []
    for goal, title in GOAL_TITLES.items():
        try:
            outcome = evaluate_goal(goal, evaluator=evaluator)
        except EvaluatorContractError:
            outcome = _unevaluated_outcome(
                goal,
                "The evaluator violated the typed contract; the Calendar goal was not evaluated.",
            )
        tasks.append(
            ResultTask(
                task_id=goal,
                title=title,
                outcome=outcome,
                evidence=outcome.reason,
                secret_values=tuple(secret_values),
            )
        )
    return tuple(tasks)


@dataclass(frozen=True)
class ServerRunReport:
    """Safe metadata about one explicit offline-shell invocation."""

    results_json: str
    output_path: Path
    configuration_valid: bool
    tasks: tuple[ResultTask, ...]



def write_results_atomic(output_path: str | Path, results_json: str) -> Path:
    """Atomically write already-validated results during explicit execution."""

    target = Path(output_path)
    if not target.name:
        raise ServerRunnerError("results output path must name a file")
    temporary_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(results_json)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ServerRunnerError("could not write results safely") from exc
    return target


class ServerRunner:
    """Explicit Phase 2 shell with no default evaluator or external clients."""

    def __init__(
        self,
        evaluator: GoalEvaluator | None = None,
        *,
        mcp_client: object | None = None,
        model_provider: object | None = None,
    ) -> None:
        self.evaluator = evaluator
        # These are reserved injection points for later phases. Phase 2 never invokes them.
        self.mcp_client = mcp_client
        self.model_provider = model_provider

    def run(
        self,
        *,
        environment: Mapping[str, object],
        output_path: str | Path,
    ) -> ServerRunReport:
        """Validate configuration, build unevaluated tasks, and write results."""

        secret_values = _known_secret_values(environment)
        try:
            validated_environment = load_environment(environment)
        except (ServerConfigurationError, TypeError) as exc:
            tasks = build_goal_tasks(secret_values=secret_values)
            safe_reason = f"UNEVALUATED: server configuration was unavailable ({type(exc).__name__})."
            tasks = tuple(
                ResultTask(
                    task_id=task.task_id,
                    title=task.title,
                    outcome=task.outcome,
                    evidence=safe_reason,
                    secret_values=secret_values,
                )
                for task in tasks
            )
            configuration_valid = False
        else:
            tasks = build_goal_tasks(
                evaluator=self.evaluator,
                secret_values=validated_environment.secret_values(),
            )
            configuration_valid = True

        results_json = serialize_results(tasks, "; ".join(
            f"{task.task_id}={task.outcome.status.value}" for task in tasks
        ))
        written_path = write_results_atomic(output_path, results_json)
        return ServerRunReport(
            results_json=results_json,
            output_path=written_path,
            configuration_valid=configuration_valid,
            tasks=tasks,
        )


def main(argv: list[str] | None = None) -> int:
    """Run the shell explicitly with the real process environment."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m harness.server_runner")
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    args = parser.parse_args(argv)
    try:
        report = ServerRunner().run(environment=os.environ, output_path=args.results)
    except ServerRunnerError as exc:
        print(f"Server runner error: {exc}")
        return 1
    return 0 if report.configuration_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
