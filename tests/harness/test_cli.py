from __future__ import annotations

import json

from calendar_agent import CalendarAgent, DeterministicScorer, Verdict
from calendar_agent.evaluation import EvaluationState
from calendar_agent.persistence import load_raw_run, persist_raw_run
from calendar_agent.runtime import BoundedAgentRuntime
from harness.__main__ import main, replay_artifact, run_demo
from harness.fixtures import deterministic_scenario
from harness.runner import Harness


class CountingCalendarAgent(CalendarAgent):
    def __init__(self, tools) -> None:
        super().__init__(tools)
        self.calls = 0

    def run(self, task):
        self.calls += 1
        return super().run(task)


def test_demo_returns_verified_report_and_persists_reloadable_artifact(tmp_path) -> None:
    artifact = tmp_path / "demo.json"

    report = run_demo(artifact)
    persisted = load_raw_run(artifact)

    assert report.raw_run is not None
    assert persisted.raw_run.run_id == report.raw_run.run_id
    assert persisted.raw_run.task == report.raw_run.task
    assert persisted.raw_run.result is not None
    assert report.raw_run.result is not None
    assert persisted.raw_run.result.status is report.raw_run.result.status
    assert persisted.raw_run.result.candidate_slots == report.raw_run.result.candidate_slots
    assert persisted.raw_run.tool_trace == report.raw_run.tool_trace
    assert report.verification.passed
    assert report.evaluation is not None
    assert report.evaluation.verdict is Verdict.APPROVE


def test_module_style_demo_execution_prints_summary(tmp_path, capsys) -> None:
    artifact = tmp_path / "demo.json"

    assert main(["--artifact", str(artifact)]) == 0

    output = capsys.readouterr().out
    assert "Harness demo" in output
    assert "reloaded: yes" in output
    assert "evaluation: approve" in output
    assert artifact.is_file()


def test_replay_scores_loaded_artifact_without_agent_model_or_tools(tmp_path) -> None:
    scenario = deterministic_scenario()
    agent = CountingCalendarAgent(scenario.tools)
    runtime = BoundedAgentRuntime(agent, scenario.model)
    report = Harness(scenario.tools).run(
        scenario.task,
        expected_status=scenario.expected_status,
        runtime=runtime,
    )
    assert report.raw_run is not None
    artifact = tmp_path / "run.json"
    persist_raw_run(artifact, report.raw_run)
    model_calls = scenario.model.calls
    agent_calls = agent.calls
    tool_calls = len(scenario.tools.trace)

    evaluation = replay_artifact(artifact)

    assert evaluation.verdict is Verdict.UNEVALUATED
    assert not evaluation.passed
    assert scenario.model.calls == model_calls
    assert agent.calls == agent_calls
    assert len(scenario.tools.trace) == tool_calls
    assert all(not call.mutating for call in scenario.tools.trace)


def test_replay_cli_accepts_explicit_predicate_without_rerunning(tmp_path, capsys) -> None:
    report = run_demo(tmp_path / "demo.json")

    assert report.raw_run is not None
    assert main(["replay", str(tmp_path / "demo.json"), "--predicate", "passed"]) == 0

    output = capsys.readouterr().out
    assert "Harness replay" in output
    assert "evaluation: approve" in output


def test_tampered_persisted_result_cannot_be_approved_by_predicate(tmp_path) -> None:
    artifact = tmp_path / "demo.json"
    report = run_demo(artifact)
    assert report.evaluation is not None
    assert report.evaluation.verdict is Verdict.APPROVE

    document = json.loads(artifact.read_text(encoding="utf-8"))
    document["raw_run"]["result"]["status"] = "failed"
    artifact.write_text(json.dumps(document), encoding="utf-8")

    persisted = load_raw_run(artifact)
    evaluation = DeterministicScorer().score(
        persisted,
        state=EvaluationState(predicate_available=True, predicate_passed=True),
    )

    assert evaluation.verdict is not Verdict.APPROVE


def test_valid_replay_without_predicate_remains_unevaluated(tmp_path) -> None:
    artifact = tmp_path / "demo.json"
    run_demo(artifact)

    evaluation = replay_artifact(artifact)

    assert evaluation.verdict is Verdict.UNEVALUATED


def test_replay_does_not_construct_runtime_or_provider_components(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "demo.json"
    run_demo(artifact)

    def forbidden(*args, **kwargs):
        raise AssertionError("replay constructed an execution component")

    monkeypatch.setattr("calendar_agent.agent.CalendarAgent", forbidden)
    monkeypatch.setattr("calendar_agent.agentswitch_client.AgentSwitchClient", forbidden)
    monkeypatch.setattr("harness.fixtures.DeterministicCalendarTools", forbidden)
    monkeypatch.setattr("harness.fixtures.FixedGoalModel", forbidden)

    evaluation = replay_artifact(
        artifact,
        predicate_state=EvaluationState(predicate_available=True, predicate_passed=True),
    )

    assert evaluation.verdict is Verdict.APPROVE


def test_unevaluated_is_not_approval() -> None:
    scenario = deterministic_scenario()
    report = Harness(scenario.tools).run(
        scenario.task,
        expected_status=scenario.expected_status,
        model=scenario.model,
    )
    assert report.raw_run is not None

    evaluation = DeterministicScorer().score(report.raw_run)

    assert evaluation.verdict is Verdict.UNEVALUATED
    assert not evaluation.passed