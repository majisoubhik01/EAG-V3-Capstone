from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from typing import Any

import pytest

from calendar_agent import (
    AgentPolicy,
    BoundedAgentRuntime,
    CalendarAgent,
    DeterministicScorer,
    EvaluationManifest,
    EvaluationState,
    IncompatibleArtifactVersionError,
    InvalidArtifactValueError,
    MalformedArtifactError,
    MissingArtifactFieldError,
    RawRunJournal,
    SelectGoal,
    Verdict,
    load_raw_run,
    persist_raw_run,
)
from tests.unit.fakes import FakeCalendarTools


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, request: Any) -> SelectGoal:
        self.calls += 1
        return SelectGoal(
            goal="calendar.find_30_minutes",
            instruction="Find time",
            context={},
        )


def make_raw_run() -> tuple[Any, CountingModel, FakeCalendarTools]:
    model = CountingModel()
    tools = FakeCalendarTools()
    policy = AgentPolicy(model=model, prompt_ref="policy://calendar/read-only/v1")
    raw_run = BoundedAgentRuntime(
        CalendarAgent(tools),
        model,
        policy=policy,
    ).run(
        "Find a slot",
        {"now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc)},
    )
    return raw_run, model, tools


def test_raw_run_round_trip_preserves_evidence_and_manifest(tmp_path) -> None:
    raw_run, _, _ = make_raw_run()
    manifest = EvaluationManifest.for_run(raw_run)

    persisted = persist_raw_run(tmp_path / "run.json", raw_run, manifest)
    loaded = load_raw_run(tmp_path / "run.json")

    assert loaded == persisted
    assert loaded.raw_run == raw_run
    assert loaded.raw_run.run_id == raw_run.run_id
    assert loaded.raw_run.model_decisions == raw_run.model_decisions
    assert loaded.raw_run.task == raw_run.task
    assert loaded.raw_run.result == raw_run.result
    assert loaded.raw_run.tool_trace == raw_run.tool_trace
    assert loaded.manifest.task_identifier == "calendar.find_30_minutes"
    assert loaded.manifest.prompt_ref == "policy://calendar/read-only/v1"
    assert loaded.manifest.model_identity is None
    assert loaded.manifest.environment is None


def test_run_persist_load_score_does_not_rerun_model_agent_or_tools(tmp_path) -> None:
    raw_run, model, tools = make_raw_run()
    persist_raw_run(tmp_path / "run.json", raw_run)
    loaded = load_raw_run(tmp_path / "run.json")
    model_calls = model.calls
    tool_calls = len(tools.trace)

    evaluation = DeterministicScorer().score(loaded)

    assert evaluation.verdict is Verdict.UNEVALUATED
    assert model.calls == model_calls
    assert len(tools.trace) == tool_calls


def test_scorer_version_change_rescores_same_persisted_run(tmp_path) -> None:
    raw_run, model, tools = make_raw_run()
    persist_raw_run(tmp_path / "run.json", raw_run)
    loaded = load_raw_run(tmp_path / "run.json")
    state = EvaluationState(predicate_available=True, predicate_passed=True)

    first = DeterministicScorer("scorer-v1").score(loaded, state=state)
    second = DeterministicScorer("scorer-v2").score(loaded, state=state)

    assert first.verdict is Verdict.APPROVE
    assert second.verdict is Verdict.APPROVE
    assert first.run_id == second.run_id == raw_run.run_id
    assert first.scorer_version == "scorer-v1"
    assert second.scorer_version == "scorer-v2"
    assert model.calls == 1
    assert tools.trace


def test_secret_values_are_redacted_from_artifact(tmp_path) -> None:
    raw_run, _, _ = make_raw_run()
    raw_run = replace(
        raw_run,
        supplied_context={
            "api_key": "api-key-secret",
            "nested": {"Authorization": "Bearer bearer-secret"},
            "ordinary": "Bearer another-secret",
        },
        policy_metadata={"password": "password-secret", "read_only": True},
    )

    path = tmp_path / "secret-safe.json"
    persist_raw_run(path, raw_run)
    artifact = path.read_text(encoding="utf-8")

    assert "api-key-secret" not in artifact
    assert "bearer-secret" not in artifact
    assert "another-secret" not in artifact
    assert "password-secret" not in artifact
    assert "[REDACTED]" in artifact


def test_missing_predicate_is_unevaluated_and_never_passes(tmp_path) -> None:
    raw_run, _, _ = make_raw_run()
    persist_raw_run(tmp_path / "run.json", raw_run)
    loaded = load_raw_run(tmp_path / "run.json")

    result = DeterministicScorer().score(loaded)

    assert result.verdict is Verdict.UNEVALUATED
    assert not result.passed


@pytest.mark.parametrize(
    ("state", "verdict"),
    [
        (EvaluationState(predicate_available=True, predicate_passed=True), Verdict.APPROVE),
        (EvaluationState(predicate_available=True, predicate_passed=False), Verdict.REVISE),
        (EvaluationState(), Verdict.UNEVALUATED),
    ],
)
def test_all_verdicts_are_deterministic(tmp_path, state, verdict) -> None:
    raw_run, _, _ = make_raw_run()
    persist_raw_run(tmp_path / "run.json", raw_run)
    loaded = load_raw_run(tmp_path / "run.json")

    first = DeterministicScorer("same-version").score(loaded, state=state)
    second = DeterministicScorer("same-version").score(loaded, state=state)

    assert first == second
    assert first.verdict is verdict
    assert first.passed is (verdict is Verdict.APPROVE)


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(MalformedArtifactError):
        RawRunJournal.loads("{not-json")


def test_missing_required_field_is_rejected() -> None:
    raw_run, _, _ = make_raw_run()
    document = json.loads(RawRunJournal.dumps(raw_run))
    del document["raw_run"]["termination_reason"]

    with pytest.raises(MissingArtifactFieldError):
        RawRunJournal.loads(json.dumps(document))


def test_invalid_status_is_rejected() -> None:
    raw_run, _, _ = make_raw_run()
    document = json.loads(RawRunJournal.dumps(raw_run))
    document["raw_run"]["result"]["status"] = "not-a-status"

    with pytest.raises(InvalidArtifactValueError):
        RawRunJournal.loads(json.dumps(document))


def test_incompatible_artifact_version_is_rejected() -> None:
    raw_run, _, _ = make_raw_run()
    document = json.loads(RawRunJournal.dumps(raw_run))
    document["artifact_version"] = 999

    with pytest.raises(IncompatibleArtifactVersionError):
        RawRunJournal.loads(json.dumps(document))


def test_incompatible_manifest_version_is_rejected() -> None:
    raw_run, _, _ = make_raw_run()
    document = json.loads(RawRunJournal.dumps(raw_run))
    document["evaluation_manifest"]["manifest_version"] = 999

    with pytest.raises(IncompatibleArtifactVersionError):
        RawRunJournal.loads(json.dumps(document))


def test_manifest_does_not_fabricate_unavailable_values() -> None:
    raw_run, _, _ = make_raw_run()

    manifest = EvaluationManifest.for_run(raw_run)

    assert manifest.task_version is None
    assert manifest.model_identity is None
    assert manifest.model_configuration is None
    assert manifest.prompt_version is None
    assert manifest.policy_version is None
    assert manifest.seed is None
    assert manifest.timeout_seconds is None
    assert manifest.tool_set_identity is None
    assert manifest.scorer_version is None