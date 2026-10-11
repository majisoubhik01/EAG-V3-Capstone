from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.results import (
    GoalOutcome,
    OutcomeStatus,
    PredicateEvaluation,
    ResultSchemaError,
    ResultTask,
    parse_results,
    sanitize_evidence,
    serialize_results,
    validate_results,
)


ROOT = Path(__file__).parents[2]


def confirmed_outcome() -> GoalOutcome:
    return GoalOutcome(
        OutcomeStatus.PASSED,
        "The authoritative predicate passed.",
        goal="calendar.find_30_minutes",
        predicate_evaluation=PredicateEvaluation("calendar.find_30_minutes", True),
    )


def failed_outcome() -> GoalOutcome:
    return GoalOutcome(
        OutcomeStatus.FAILED,
        "The authoritative predicate failed.",
        goal="calendar.find_30_minutes",
        predicate_evaluation=PredicateEvaluation("calendar.find_30_minutes", False),
    )


def result_task(
    task_id: str = "find-30",
    *,
    outcome: GoalOutcome | None = None,
    evidence: str = "The predicate result is recorded.",
) -> ResultTask:
    return ResultTask(
        task_id,
        "Calendar goal",
        outcome or confirmed_outcome(),
        evidence,
    )


def valid_document() -> dict[str, object]:
    return {
        "tasks": [{
            "id": "find-30",
            "title": "Calendar goal",
            "passed": True,
            "score": 1.0,
            "evidence": "The predicate result is recorded.",
        }],
        "summary": "1/1 checks passed",
    }


def test_official_configuration_contains_verified_field_names_without_inventing_instance() -> None:
    config = (ROOT / "agentswitch-harness.toml").read_text(encoding="utf-8")

    assert 'install = "pip install -e ."' in config
    assert 'run = "python -m harness.runner"' in config
    assert 'results = "results.json"' in config
    assert "instances = []" in config
    assert "timeout_minutes = 10" in config


def test_valid_results_round_trip() -> None:
    serialized = serialize_results([result_task()], "1/1 checks passed")

    assert parse_results(serialized) == valid_document()


def test_single_line_summary_is_valid() -> None:
    validate_results(valid_document())


@pytest.mark.parametrize("line_break", ["\n", "\r", "\r\n"])
def test_results_reject_multiline_summary(line_break: str) -> None:
    document = valid_document()
    document["summary"] = f"checks started{line_break}checks ended"

    with pytest.raises(ResultSchemaError, match="single line"):
        validate_results(document)


def test_failed_predicate_serializes_as_failed() -> None:
    document = parse_results(
        serialize_results(
            [result_task(outcome=failed_outcome(), evidence="The predicate was false.")],
            "0/1 checks passed",
        )
    )

    assert document["tasks"][0]["passed"] is False
    assert document["tasks"][0]["score"] == 0.0


def test_unevaluated_outcome_is_explicit_and_never_passes() -> None:
    outcome = GoalOutcome(OutcomeStatus.UNEVALUATED, "The evaluator was unavailable.")

    document = parse_results(
        serialize_results(
            [result_task(outcome=outcome, evidence="No authoritative predicate was available.")],
            "0/1 checks evaluated",
        )
    )

    task = document["tasks"][0]
    assert task["passed"] is False
    assert task["score"] == 0.0
    assert task["evidence"].startswith("UNEVALUATED:")


def test_planned_without_authoritative_predicate_cannot_be_passed() -> None:
    with pytest.raises(ValueError, match="confirmed predicate"):
        GoalOutcome(
            OutcomeStatus.PASSED,
            "A local planner returned PLANNED.",
            goal="calendar.find_30_minutes",
        )


def test_provider_or_mcp_success_alone_remains_unevaluated() -> None:
    for reason in ("MCP tools/list succeeded.", "The model provider returned a response."):
        outcome = GoalOutcome(OutcomeStatus.UNEVALUATED, reason)
        task = result_task(outcome=outcome)
        assert task.outcome.status is OutcomeStatus.UNEVALUATED
        assert task.official_record()["passed"] is False


def test_confirmed_true_predicate_is_required_for_passed() -> None:
    with pytest.raises(ValueError, match="confirmed predicate"):
        GoalOutcome(
            OutcomeStatus.PASSED,
            "The predicate was not checked.",
            goal="calendar.find_30_minutes",
        )

    assert result_task().outcome.status is OutcomeStatus.PASSED


def test_failed_requires_false_evaluation_for_the_same_goal() -> None:
    with pytest.raises(ValueError, match="for its goal"):
        GoalOutcome(
            OutcomeStatus.FAILED,
            "The wrong goal was evaluated.",
            goal="calendar.find_30_minutes",
            predicate_evaluation=PredicateEvaluation("calendar.move_after_audit", False),
        )


def test_source_labels_cannot_be_used_as_predicate_evidence() -> None:
    with pytest.raises(TypeError, match="PredicateEvaluation"):
        GoalOutcome(
            OutcomeStatus.PASSED,
            "MCP succeeded.",
            goal="calendar.find_30_minutes",
            predicate_evaluation="MCP",  # type: ignore[arg-type]
        )


def test_unevaluated_cannot_contain_completed_predicate() -> None:
    with pytest.raises(ValueError, match="predicate evaluation"):
        GoalOutcome(
            OutcomeStatus.UNEVALUATED,
            "The outcome was unavailable.",
            goal="calendar.find_30_minutes",
            predicate_evaluation=PredicateEvaluation("calendar.find_30_minutes", False),
        )


@pytest.mark.parametrize(
    "document",
    [
        {"tasks": [], "summary": "none"},
        {"tasks": [{"id": str(index), "title": "T", "passed": False, "evidence": "E"} for index in range(201)], "summary": "too many"},
        {"tasks": [{"id": "same", "title": "T", "passed": False, "evidence": "E"}, {"id": "same", "title": "T", "passed": False, "evidence": "E"}], "summary": "duplicate"},
        {"tasks": [{"id": "t", "title": "T", "passed": 1, "evidence": "E"}], "summary": "not boolean"},
        {"tasks": [{"id": "t", "title": "T", "passed": False, "score": -0.1, "evidence": "E"}], "summary": "bad score"},
        {"tasks": [{"id": "t", "title": "T", "passed": False, "score": 1.1, "evidence": "E"}], "summary": "bad score"},
        {"tasks": [{"id": "t", "title": "T", "passed": False, "score": "0", "evidence": "E"}], "summary": "bad score"},
        {"tasks": [{"id": "t", "title": "T", "passed": False, "evidence": ""}], "summary": "bad evidence"},
    ],
)
def test_invalid_results_are_rejected(document: dict[str, object]) -> None:
    with pytest.raises(ResultSchemaError):
        validate_results(document)


def test_missing_and_unsupported_fields_are_rejected() -> None:
    missing_summary = {"tasks": valid_document()["tasks"]}
    unsupported = {**valid_document(), "internal_outcome": "passed"}

    with pytest.raises(ResultSchemaError):
        validate_results(missing_summary)
    with pytest.raises(ResultSchemaError):
        validate_results(unsupported)


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(ResultSchemaError, match="valid JSON"):
        parse_results("{not-json")


def test_serializer_rejects_empty_tasks() -> None:
    with pytest.raises(ResultSchemaError, match="between 1 and 200"):
        serialize_results([], "no tasks")


def test_long_evidence_is_kept_concise() -> None:
    evidence = "x" * 1000

    document = parse_results(serialize_results([result_task(evidence=evidence)], "summary"))

    assert len(document["tasks"][0]["evidence"]) == 500


def test_known_and_credential_shaped_secrets_are_sanitized_before_truncation() -> None:
    evidence = (
        "token=known-token; password=hunter2; Authorization: Bearer header-token; "
        "OPENAI_API_KEY=sk-test-key-1234567890; "
        + ("x" * 600)
    )

    document = parse_results(
        serialize_results(
            [result_task(evidence=evidence,)],
            "summary",
        )
    )
    serialized_evidence = document["tasks"][0]["evidence"]

    assert "known-token" not in serialized_evidence
    assert "hunter2" not in serialized_evidence
    assert "header-token" not in serialized_evidence
    assert "sk-test-key-1234567890" not in serialized_evidence
    assert len(serialized_evidence) == 500


def test_explicit_secret_values_are_sanitized_even_without_a_known_pattern() -> None:
    task = ResultTask(
        "find-30",
        "Calendar goal",
        confirmed_outcome(),
        "diagnostic secret-value=" + "opaque-secret" + " end",
        secret_values=("opaque-secret",),
    )

    record = task.official_record()

    assert "opaque-secret" not in record["evidence"]
    assert "[REDACTED]" in record["evidence"]


def test_sanitize_evidence_documents_defensive_scope() -> None:
    sanitized = sanitize_evidence("The opaque value is opaque-secret.", ("opaque-secret",))

    assert sanitized == "The opaque value is [REDACTED]."


def test_json_is_object_not_a_list() -> None:
    with pytest.raises(ResultSchemaError, match="object"):
        parse_results(json.dumps([]))
