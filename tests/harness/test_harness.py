from datetime import datetime, timedelta, timezone

from calendar_agent.agent import CalendarAgent
from calendar_agent.evaluation import DeterministicScorer, EvaluationState
from calendar_agent.models import (
    AgentResult,
    AgentTask,
    AuditReference,
    CalendarEventRef,
    CandidateSlot,
    EventMove,
    ResultStatus,
    ToolCallRecord,
)
from calendar_agent.runtime import BoundedAgentRuntime
from harness.assertions import verify_result
from harness.runner import Harness

from tests.unit.fakes import FakeCalendarTools


class CountingModel:
    def __init__(self, decision: object) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, request: object) -> object:
        self.calls += 1
        return self.decision


class CountingCalendarAgent(CalendarAgent):
    def __init__(self, tools) -> None:
        super().__init__(tools)
        self.calls = 0

    def run(self, task):
        self.calls += 1
        return super().run(task)


def goal_one_model() -> CountingModel:
    from calendar_agent.runtime import SelectGoal

    return CountingModel(SelectGoal("calendar.find_30_minutes", "Find time", {}))


def test_harness_verifies_state_independently_of_claimed_outcome() -> None:
    result = AgentResult(
        goal="calendar.find_30_minutes",
        status=ResultStatus.COMPLETED,
        summary="Done",
        claimed_outcome={"success": True},
        tool_calls=[ToolCallRecord("Party.list", {})],
    )

    verification = verify_result(result, expected_status=ResultStatus.NEEDS_CLARIFICATION)

    assert not verification.passed
    assert not verification.outcome_ok
    assert verification.integrity_ok


def test_harness_captures_result_and_trace() -> None:
    tools = FakeCalendarTools()
    report = Harness(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head"),
        expected_status=ResultStatus.NEEDS_CLARIFICATION,
    )

    assert report.verification.passed
    assert report.result.tool_calls
    assert all(not call.mutating for call in report.result.tool_calls)


def test_harness_direct_execution_remains_without_evaluation() -> None:
    report = Harness(FakeCalendarTools()).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head"),
        expected_status=ResultStatus.NEEDS_CLARIFICATION,
    )

    assert report.evaluation is None


def test_harness_persisted_runtime_execution_scores_reloaded_artifact(tmp_path) -> None:
    model = goal_one_model()
    tools = FakeCalendarTools()
    journal_path = tmp_path / "run.json"

    report = Harness(tools).run_persisted(
        AgentTask(
            "calendar.find_30_minutes",
            "Find time",
            {
                "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
                "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
                "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
            },
        ),
        expected_status=ResultStatus.NEEDS_CLARIFICATION,
        model=model,
        journal_path=journal_path,
    )

    from calendar_agent.persistence import load_raw_run

    persisted = load_raw_run(journal_path)
    assert report.raw_run is not None
    assert persisted.raw_run == report.raw_run
    assert report.evaluation is not None
    assert report.evaluation.verdict.value == "unevaluated"
    assert report.evaluation.run_id == persisted.raw_run.run_id
    assert model.calls == 1
    assert all(not call.mutating for call in tools.trace)


def test_harness_persisted_scoring_consumes_loaded_artifact_without_rerunning(tmp_path) -> None:
    model = goal_one_model()
    tools = FakeCalendarTools()
    agent = CountingCalendarAgent(tools)
    runtime = BoundedAgentRuntime(agent, model)
    scorer = RecordingScorer()

    report = Harness(tools).run_persisted(
        AgentTask("calendar.find_30_minutes", "Find time"),
        expected_status=ResultStatus.NEEDS_CLARIFICATION,
        runtime=runtime,
        journal_path=tmp_path / "run.json",
        scorer=scorer,
    )

    assert scorer.loaded_types == ["PersistedRun"]
    assert report.evaluation is not None
    assert model.calls == 1
    assert agent.calls == 1
    assert len(tools.trace) == 2
    assert all(not call.mutating for call in tools.trace)


class RecordingScorer(DeterministicScorer):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_types: list[str] = []

    def score(self, evidence, *, state=None):
        self.loaded_types.append(type(evidence).__name__)
        return super().score(evidence, state=state)


def test_harness_persisted_scoring_preserves_unevaluated_without_predicate(tmp_path) -> None:
    model = goal_one_model()
    report = Harness(FakeCalendarTools()).run_persisted(
        AgentTask("calendar.find_30_minutes", "Find time"),
        expected_status=ResultStatus.NEEDS_CLARIFICATION,
        model=model,
        journal_path=tmp_path / "run.json",
        predicate_state=EvaluationState(),
    )

    assert report.evaluation is not None
    assert report.evaluation.verdict.value == "unevaluated"
    assert not report.evaluation.passed


def test_harness_rejects_planned_goal_one_without_valid_candidate_slot() -> None:
    result = AgentResult(
        goal="calendar.find_30_minutes",
        status=ResultStatus.PLANNED,
        summary="Done",
        candidate_slots=[],
        tool_calls=[ToolCallRecord("Party.list", {})],
    )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)

    assert not verification.passed
    assert not verification.outcome_ok


def test_harness_rejects_malformed_goal_one_candidate_slot() -> None:
    result = AgentResult(
        goal="calendar.find_30_minutes",
        status=ResultStatus.PLANNED,
        summary="Done",
        candidate_slots=[
            CandidateSlot(
                start_at="2026-09-22T09:00:00",
                end_at="2026-09-22T09:31:00",
                timezone=None,
                availability_confirmed=False,
                reason="",
            )
        ],
        tool_calls=[ToolCallRecord("Party.list", {})],
    )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)

    assert not verification.passed
    assert not verification.outcome_ok


def test_harness_rejects_empty_goal_two_plan() -> None:
    result = AgentResult(
        goal="calendar.move_after_audit",
        status=ResultStatus.PLANNED,
        summary="Done",
        audit_reference=AuditReference(
            candidates=(
                CalendarEventRef(
                    id="audit-1",
                    title="Audit",
                    start_at="2026-09-22T09:00:00+00:00",
                    end_at="2026-09-22T10:00:00+00:00",
                    timezone="UTC",
                ),
            ),
            search_terms=("audit",),
        ),
        tool_calls=[ToolCallRecord("CalendarEvent.list", {})],
    )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)

    assert not verification.passed
    assert not verification.outcome_ok


def test_harness_rejects_contradictory_goal_two_plan() -> None:
    result = AgentResult(
        goal="calendar.move_after_audit",
        status=ResultStatus.PLANNED,
        summary="Done",
        audit_reference=AuditReference(
            candidates=(
                CalendarEventRef(
                    id="audit-1",
                    title="Audit",
                    start_at="2026-09-22T09:00:00+00:00",
                    end_at="2026-09-22T10:00:00+00:00",
                    timezone="UTC",
                ),
            ),
            search_terms=("audit",),
        ),
        rescheduling_plan=[
            EventMove(
                event_id="work-1",
                title="Work",
                original_start_at="2026-09-22T11:00:00+00:00",
                original_end_at="2026-09-22T12:00:00+00:00",
                proposed_start_at="2026-09-22T12:00:00+00:00",
                proposed_end_at="2026-09-22T13:00:00+00:00",
                duration_seconds=3600,
                timezone="UTC",
            )
        ],
        claimed_outcome={
            "moved": True,
            "planned_event_count": 1,
            "shift": timedelta(hours=1),
        },
        tool_calls=[ToolCallRecord("CalendarEvent.list", {})],
    )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)

    assert not verification.passed
    assert not verification.outcome_ok


def test_planned_result_is_not_treated_as_completed() -> None:
    result = AgentResult(
        goal="calendar.find_30_minutes",
        status=ResultStatus.PLANNED,
        summary="A slot was planned",
        tool_calls=[ToolCallRecord("Party.list", {})],
    )

    verification = verify_result(result, expected_status=ResultStatus.COMPLETED)

    assert not verification.passed
    assert not verification.outcome_ok