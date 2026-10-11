from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness.__main__ import main as local_harness_main
from harness.results import OutcomeStatus, PredicateEvaluation, parse_results
from harness.runner import Harness, main as runner_main
from harness.server_runner import (
    GOAL_TITLES,
    EvaluatorContractError,
    ServerConfigurationError,
    ServerRunner,
    ServerRunnerError,
    evaluate_goal,
    load_environment,
    write_results_atomic,
)


VALID_ENV = {
    "AGENTSWITCH_BASE_URL": "https://agentswitch.example.test",
    "AGENTSWITCH_TOKEN": "seat-token-for-tests",
    "AGENTSWITCH_INSTANCE": "unverified-test-instance",
    "OPENAI_BASE_URL": "https://model.example.test/v1",
    "OPENAI_API_KEY": "sk-test-key-1234567890",
    "OPENAI_MODEL": "test-model",
}
ROOT = Path(__file__).parents[2]


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, *args, **kwargs) -> None:
        self.calls += 1


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *args, **kwargs) -> None:
        self.calls += 1


class MatchingEvaluator:
    def __init__(self, values: dict[str, bool]) -> None:
        self.values = values
        self.calls: list[str] = []

    def evaluate(self, goal: str) -> PredicateEvaluation:
        self.calls.append(goal)
        return PredicateEvaluation(goal, self.values[goal])


class WrongGoalEvaluator:
    def evaluate(self, goal: str) -> PredicateEvaluation:
        return "not-a-predicate-evaluation"  # type: ignore[return-value]


class RaisingEvaluator:
    def evaluate(self, goal: str) -> PredicateEvaluation:
        raise TimeoutError("provider details must not be exposed")


def test_valid_environment_is_loaded_without_secret_repr() -> None:
    environment = load_environment(VALID_ENV)

    assert environment.agentswitch_base_url == "https://agentswitch.example.test"
    assert environment.openai_model == "test-model"
    representation = repr(environment)
    assert VALID_ENV["AGENTSWITCH_TOKEN"] not in representation
    assert VALID_ENV["OPENAI_API_KEY"] not in representation


@pytest.mark.parametrize(
    "key",
    [
        "AGENTSWITCH_BASE_URL",
        "AGENTSWITCH_TOKEN",
        "AGENTSWITCH_INSTANCE",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ],
)
def test_missing_environment_value_is_rejected_without_leaking_secrets(key: str) -> None:
    values = dict(VALID_ENV)
    values.pop(key)

    with pytest.raises(ServerConfigurationError) as error:
        load_environment(values)

    assert key in str(error.value)
    assert VALID_ENV["AGENTSWITCH_TOKEN"] not in str(error.value)
    assert VALID_ENV["OPENAI_API_KEY"] not in str(error.value)


def test_invalid_urls_are_rejected_without_reading_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "AGENTSWITCH_TOKEN=dotenv-token\nOPENAI_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    values = dict(VALID_ENV)
    values["AGENTSWITCH_BASE_URL"] = "not-a-url"

    with pytest.raises(ServerConfigurationError, match="AGENTSWITCH_BASE_URL"):
        load_environment(values)

    assert load_environment(VALID_ENV).agentswitch_token == VALID_ENV["AGENTSWITCH_TOKEN"]


def test_import_and_default_runner_do_not_call_mcp_or_model(tmp_path: Path) -> None:
    client = CountingClient()
    provider = CountingProvider()
    output = tmp_path / "results.json"

    assert not output.exists()
    report = ServerRunner(mcp_client=client, model_provider=provider).run(
        environment=VALID_ENV,
        output_path=output,
    )

    assert report.configuration_valid
    assert client.calls == 0
    assert provider.calls == 0
    assert output.is_file()
    parsed = parse_results(output.read_text(encoding="utf-8"))
    assert all(task["passed"] is False for task in parsed["tasks"])
    assert all(task["evidence"].startswith("UNEVALUATED:") for task in parsed["tasks"])


def _subprocess_environment(**updates: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "AGENTSWITCH_BASE_URL",
            "AGENTSWITCH_TOKEN",
            "AGENTSWITCH_INSTANCE",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        }
    }
    environment.update({key: value for key, value in VALID_ENV.items()})
    environment.update(updates)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_module_import_does_not_execute_or_create_results(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import harness.runner"],
        cwd=tmp_path,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert not (tmp_path / "results.json").exists()
    assert result.stdout == ""
    assert result.stderr == ""


def test_configured_module_entry_point_writes_unevaluated_results(tmp_path: Path) -> None:
    output = tmp_path / "results.json"

    result = subprocess.run(
        [sys.executable, "-m", "harness.runner", "--results", str(output)],
        cwd=tmp_path,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    document = parse_results(output.read_text(encoding="utf-8"))
    assert len(document["tasks"]) == 2
    assert all(task["passed"] is False for task in document["tasks"])
    assert all(type(task["passed"]) is bool for task in document["tasks"])
    assert VALID_ENV["AGENTSWITCH_TOKEN"] not in result.stdout + result.stderr
    assert VALID_ENV["OPENAI_API_KEY"] not in result.stdout + result.stderr


def test_invalid_configuration_writes_nonpassing_results_and_returns_failure(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    environment = _subprocess_environment()
    environment.pop("AGENTSWITCH_INSTANCE")

    result = subprocess.run(
        [sys.executable, "-m", "harness.runner", "--results", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    document = parse_results(output.read_text(encoding="utf-8"))
    assert all(task["passed"] is False for task in document["tasks"])


def test_local_demo_entry_point_remains_unchanged(tmp_path: Path) -> None:
    artifact = tmp_path / "demo.json"

    result = subprocess.run(
        [sys.executable, "-m", "harness", "--artifact", str(artifact)],
        cwd=tmp_path,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Harness demo" in result.stdout
    assert artifact.is_file()


def test_no_evaluator_keeps_both_calendar_goals_unevaluated(tmp_path: Path) -> None:
    report = ServerRunner().run(environment=VALID_ENV, output_path=tmp_path / "results.json")

    assert {task.task_id for task in report.tasks} == set(GOAL_TITLES)
    assert all(task.outcome.status is OutcomeStatus.UNEVALUATED for task in report.tasks)
    assert all(task.official_record()["passed"] is False for task in report.tasks)


def test_injected_evaluator_can_produce_only_matching_typed_goal_results(tmp_path: Path) -> None:
    evaluator = MatchingEvaluator({
        "calendar.find_30_minutes": True,
        "calendar.move_after_audit": False,
    })

    report = ServerRunner(evaluator=evaluator).run(
        environment=VALID_ENV,
        output_path=tmp_path / "results.json",
    )

    by_id = {task.task_id: task for task in report.tasks}
    assert by_id["calendar.find_30_minutes"].outcome.status is OutcomeStatus.PASSED
    assert by_id["calendar.move_after_audit"].outcome.status is OutcomeStatus.FAILED
    assert evaluator.calls == list(GOAL_TITLES)


def test_wrong_evaluator_goal_fails_closed(tmp_path: Path) -> None:
    report = ServerRunner(evaluator=WrongGoalEvaluator()).run(
        environment=VALID_ENV,
        output_path=tmp_path / "results.json",
    )

    assert all(task.outcome.status is OutcomeStatus.UNEVALUATED for task in report.tasks)


def test_evaluator_contract_rejects_unsupported_direct_goal() -> None:
    with pytest.raises(EvaluatorContractError):
        evaluate_goal("calendar.unknown", evaluator=MatchingEvaluator({}))


def test_provider_auth_permission_and_mcp_failures_cannot_pass(tmp_path: Path) -> None:
    report = ServerRunner(evaluator=RaisingEvaluator()).run(
        environment=VALID_ENV,
        output_path=tmp_path / "results.json",
    )

    assert all(task.outcome.status is OutcomeStatus.UNEVALUATED for task in report.tasks)
    assert all(task.official_record()["passed"] is False for task in report.tasks)
    serialized = report.results_json
    assert "provider details" not in serialized
    assert VALID_ENV["AGENTSWITCH_TOKEN"] not in serialized
    assert VALID_ENV["OPENAI_API_KEY"] not in serialized


def test_configuration_failure_still_writes_safe_nonempty_results(tmp_path: Path) -> None:
    values = dict(VALID_ENV)
    values.pop("AGENTSWITCH_INSTANCE")
    output = tmp_path / "results.json"

    report = ServerRunner().run(environment=values, output_path=output)

    assert not report.configuration_valid
    parsed = parse_results(output.read_text(encoding="utf-8"))
    assert len(parsed["tasks"]) == 2
    assert all(task["passed"] is False for task in parsed["tasks"])
    assert VALID_ENV["AGENTSWITCH_TOKEN"] not in report.results_json
    assert VALID_ENV["OPENAI_API_KEY"] not in report.results_json


def test_atomic_write_replaces_target_and_cleans_up_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    output.write_text("old", encoding="utf-8")
    results_json = '{"tasks": [{"id": "t1", "title": "T", "passed": false, "evidence": "E"}], "summary": "0/1"}'

    written = write_results_atomic(output, results_json)

    assert written == output
    assert output.read_text(encoding="utf-8") == results_json
    assert parse_results(output.read_text(encoding="utf-8"))["tasks"][0]["id"] == "t1"
    assert list(tmp_path.glob(".results.json.*.tmp")) == []


def test_atomic_write_failure_preserves_target_and_cleans_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("contains a secret token")

    monkeypatch.setattr("harness.server_runner.os.replace", fail_replace)
    output = tmp_path / "results.json"
    original = b'{"existing result": true}\n'
    output.write_bytes(original)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"keep this file")

    with pytest.raises(ServerRunnerError, match="could not write results safely") as error:
        write_results_atomic(output, "safe")

    assert "secret token" not in str(error.value)
    assert output.read_bytes() == original
    assert unrelated.read_bytes() == b"keep this file"
    assert list(tmp_path.glob(".results.json.*.tmp")) == []


def test_atomic_write_parent_creation_failure_is_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"keep this file")

    def fail_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("contains a secret token")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with pytest.raises(ServerRunnerError, match="could not write results safely") as error:
        write_results_atomic(tmp_path / "missing" / "results.json", "safe")

    assert "secret token" not in str(error.value)
    assert unrelated.read_bytes() == b"keep this file"


def test_atomic_write_temporary_file_creation_failure_preserves_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "results.json"
    original = b"old result bytes"
    output.write_bytes(original)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"keep this file")

    def fail_temporary_file(*args: object, **kwargs: object) -> None:
        raise OSError("contains a secret token")

    monkeypatch.setattr("harness.server_runner.tempfile.NamedTemporaryFile", fail_temporary_file)
    with pytest.raises(ServerRunnerError, match="could not write results safely") as error:
        write_results_atomic(output, "safe")

    assert "secret token" not in str(error.value)
    assert output.read_bytes() == original
    assert unrelated.read_bytes() == b"keep this file"


def test_cli_returns_failure_when_atomic_result_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    for key, value in VALID_ENV.items():
        monkeypatch.setenv(key, value)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("harness.server_runner.os.replace", fail_replace)
    output = tmp_path / "results.json"
    original = b"previous run result"
    output.write_bytes(original)

    exit_code = runner_main(["--results", str(output)])

    assert exit_code != 0
    assert "Server runner error: could not write results safely" in capsys.readouterr().out
    assert output.read_bytes() == original
    assert list(tmp_path.glob(".results.json.*.tmp")) == []


def test_existing_local_runner_api_remains_importable() -> None:
    assert callable(local_harness_main)
    assert callable(Harness.run)
    assert callable(Harness.run_persisted)
