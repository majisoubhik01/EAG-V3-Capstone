"""Executable deterministic Harness demo and persisted-run replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from calendar_agent.evaluation import (
    DeterministicScorer,
    EvaluationResult,
    EvaluationState,
    Verdict,
    build_evaluation_evidence,
)
from calendar_agent.persistence import load_raw_run, persist_raw_run

from .fixtures import deterministic_scenario
from .runner import Harness, HarnessReport


DEFAULT_ARTIFACT = Path(".harness-artifacts") / "demo.json"


def run_demo(artifact: str | Path = DEFAULT_ARTIFACT) -> HarnessReport:
    scenario = deterministic_scenario()
    report = Harness(scenario.tools).run(
        scenario.task,
        expected_status=scenario.expected_status,
        model=scenario.model,
    )
    if report.raw_run is None:
        raise RuntimeError("Harness runtime did not produce a RawRun")

    artifact_path = Path(artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    persist_raw_run(
        artifact_path,
        report.raw_run,
        evaluation_evidence=build_evaluation_evidence(
            report.raw_run,
            report.verification,
            expected_status=scenario.expected_status,
        ),
    )
    persisted = load_raw_run(artifact_path)
    evaluation = DeterministicScorer().score(
        persisted,
        state=EvaluationState(
            predicate_available=True,
            predicate_passed=report.verification.passed,
        ),
    )
    return replace(report, evaluation=evaluation)


def replay_artifact(
    artifact: str | Path,
    *,
    predicate_state: EvaluationState | None = None,
) -> EvaluationResult:
    """Load and score an artifact without constructing or invoking runtime objects."""

    persisted = load_raw_run(artifact)
    return DeterministicScorer().score(persisted, state=predicate_state)


def _predicate_state(value: str) -> EvaluationState | None:
    if value == "passed":
        return EvaluationState(predicate_available=True, predicate_passed=True)
    if value == "failed":
        return EvaluationState(predicate_available=True, predicate_passed=False)
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    subparsers = parser.add_subparsers(dest="command")
    replay = subparsers.add_parser("replay", help="load and rescore an existing artifact")
    replay.add_argument("artifact", type=Path)
    replay.add_argument(
        "--predicate",
        choices=("unavailable", "passed", "failed"),
        default="unavailable",
        help="explicit predicate state supplied to the deterministic scorer",
    )
    return parser


def _print_demo(report: HarnessReport, artifact: Path) -> None:
    assert report.raw_run is not None
    assert report.evaluation is not None
    print("Harness demo")
    print(f"scenario: {report.task.goal}")
    print(f"status: {report.result.status.value if report.result is not None else 'none'}")
    print(f"termination: {report.raw_run.termination_reason.value}")
    print(f"verified: {'pass' if report.verification.passed else 'fail'}")
    print(f"artifact: {artifact}")
    print("reloaded: yes")
    print(f"evaluation: {report.evaluation.verdict.value}")


def _print_replay(artifact: Path, evaluation: EvaluationResult) -> None:
    print("Harness replay")
    print(f"artifact: {artifact}")
    print(f"run_id: {evaluation.run_id}")
    print(f"evaluation: {evaluation.verdict.value}")
    print(f"reason: {evaluation.reason}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "replay":
            evaluation = replay_artifact(args.artifact, predicate_state=_predicate_state(args.predicate))
            _print_replay(args.artifact, evaluation)
            return 0

        report = run_demo(args.artifact)
        _print_demo(report, args.artifact)
        return 0 if report.evaluation is not None and report.evaluation.verdict is Verdict.APPROVE else 1
    except Exception as exc:
        print(f"Harness error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())