"""Scenario runner with independent verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from calendar_agent.agent import CalendarAgent
from calendar_agent.evaluation import DeterministicScorer, EvaluationResult, EvaluationState
from calendar_agent.models import AgentResult, AgentTask, ResultStatus, VerificationResult
from calendar_agent.persistence import load_raw_run, persist_raw_run
from calendar_agent.runtime import AgentPolicy, BoundedAgentRuntime, Model, RawRun
from calendar_agent.tools import CalendarTool

from .assertions import verify_result


@dataclass(frozen=True)
class HarnessReport:
    task: AgentTask
    result: AgentResult | None
    verification: VerificationResult
    raw_run: RawRun | None = None
    evaluation: EvaluationResult | None = None


class Harness:
    def __init__(self, tools: CalendarTool) -> None:
        self.tools = tools

    def run(
        self,
        task: AgentTask,
        *,
        expected_status: ResultStatus,
        runtime: BoundedAgentRuntime | None = None,
        model: Model | None = None,
        policy: AgentPolicy | None = None,
    ) -> HarnessReport:
        if runtime is not None and (model is not None or policy is not None):
            raise ValueError("runtime cannot be combined with model or policy")
        if runtime is None and model is not None:
            runtime = BoundedAgentRuntime(CalendarAgent(self.tools), model, policy=policy)

        if runtime is None:
            result = CalendarAgent(self.tools).run(task)
            verification = verify_result(result, expected_status=expected_status)
            return HarnessReport(task=task, result=result, verification=verification)

        raw_run = runtime.run(task.instruction, task.context)
        if raw_run.result is not None:
            verification = verify_result(raw_run.result, expected_status=expected_status)
        else:
            verification = VerificationResult(
                passed=False,
                outcome_ok=False,
                integrity_ok=not any(call.mutating for call in raw_run.tool_trace),
                details=("The runtime produced no AgentResult to verify independently.",),
            )
        return HarnessReport(
            task=task,
            result=raw_run.result,
            verification=verification,
            raw_run=raw_run,
        )

    def run_persisted(
        self,
        task: AgentTask,
        *,
        expected_status: ResultStatus,
        journal_path: str | Path,
        runtime: BoundedAgentRuntime | None = None,
        model: Model | None = None,
        policy: AgentPolicy | None = None,
        predicate_state: EvaluationState | None = None,
        scorer: DeterministicScorer | None = None,
    ) -> HarnessReport:
        """Run through the runtime, persist evidence, reload it, and score it."""

        if runtime is None and model is None:
            raise ValueError("run_persisted requires a runtime or model")

        report = self.run(
            task,
            expected_status=expected_status,
            runtime=runtime,
            model=model,
            policy=policy,
        )
        if report.raw_run is None:
            raise ValueError("runtime execution did not produce a RawRun")

        persist_raw_run(journal_path, report.raw_run)
        persisted = load_raw_run(journal_path)
        evaluation = (scorer or DeterministicScorer()).score(
            persisted,
            state=predicate_state,
        )
        return replace(report, evaluation=evaluation)