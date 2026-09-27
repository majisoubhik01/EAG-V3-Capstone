from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from calendar_agent import (
    AgentPolicy,
    BoundedAgentRuntime,
    CalendarAgent,
    Clarify,
    Fail,
    ResultStatus,
    SelectGoal,
    TerminationReason,
)
from calendar_agent.models import AgentTask
from harness.assertions import verify_result
from harness.runner import Harness
from tests.unit.fakes import FakeCalendarTools


class FakeModel:
    def __init__(self, decision: object = None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.requests: list[Any] = []
        self.calls = 0

    def decide(self, request: Any) -> object:
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.decision


def goal_one_tools() -> FakeCalendarTools:
    return FakeCalendarTools(
        parties=[{
            "id": "plant-1",
            "email": "plant@example.test",
            "job_title": "Plant Manager",
        }],
        calendars=[{"id": "calendar-1", "owner_party_id": "plant-1"}],
        availability_rules=[{
            "id": "rule-1",
            "party_id": "plant-1",
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            "date_overrides": [],
        }],
        scheduling_preferences=[{"default_meeting_duration": 30, "min_notice_hours": 0}],
        free_busy_result={"result": {"subjects": [{
            "subject": "plant@example.test",
            "visibility": "free_busy",
            "busy": [],
            "calendars": [{"access_level": "free_busy"}],
        }]}},
    )


def goal_two_tools() -> FakeCalendarTools:
    return FakeCalendarTools(events=[
        {
            "id": "audit-1",
            "title": "Internal quality audit",
            "description": "ISO review",
            "start_at": "2026-09-22T09:00:00+00:00",
            "end_at": "2026-09-22T10:00:00+00:00",
            "timezone": "UTC",
            "calendar_id": "calendar-1",
        },
        {
            "id": "work-1",
            "title": "Work",
            "start_at": "2026-09-22T11:00:00+00:00",
            "end_at": "2026-09-22T12:00:00+00:00",
            "timezone": "UTC",
            "calendar_id": "calendar-1",
        },
    ])


def test_runtime_selects_goal_one_and_builds_agent_task() -> None:
    model = FakeModel(SelectGoal(
        goal="calendar.find_30_minutes",
        instruction="Find 30 minutes with the plant head",
        context={},
    ))
    agent = CalendarAgent(goal_one_tools())
    runtime = BoundedAgentRuntime(agent, model)

    raw_run = runtime.run(
        "Find 30 minutes with the plant head.",
        {"now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc), "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc), "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc)},
    )

    assert raw_run.termination_reason is TerminationReason.SUCCEEDED
    assert raw_run.task is not None
    assert raw_run.task.goal == "calendar.find_30_minutes"
    assert raw_run.result is not None
    assert raw_run.result.status is ResultStatus.PLANNED
    assert model.requests[0].user_instruction == "Find 30 minutes with the plant head."
    assert model.requests[0].available_goal_ids == (
        "calendar.find_30_minutes",
        "calendar.move_after_audit",
    )
    assert model.requests[0].policy_metadata["read_only"] is True


def test_runtime_selects_goal_two_and_builds_agent_task() -> None:
    model = FakeModel(SelectGoal(
        goal="calendar.move_after_audit",
        instruction="Move everything after the audit",
        context={"shift": timedelta(hours=1)},
    ))
    agent = CalendarAgent(goal_two_tools())

    raw_run = BoundedAgentRuntime(agent, model).run("Move everything after the audit")

    assert raw_run.termination_reason is TerminationReason.SUCCEEDED
    assert raw_run.task is not None
    assert raw_run.task.goal == "calendar.move_after_audit"
    assert raw_run.task.context["shift"] == timedelta(hours=1)
    assert raw_run.result is not None
    assert raw_run.result.status is ResultStatus.PLANNED
    assert [move.event_id for move in raw_run.result.rescheduling_plan] == ["work-1"]


def test_runtime_rejects_unknown_goal_without_running_calendar_agent() -> None:
    model = FakeModel(SelectGoal("calendar.unknown", "Unknown", {}))
    tools = FakeCalendarTools()

    raw_run = BoundedAgentRuntime(CalendarAgent(tools), model).run("Do something")

    assert raw_run.termination_reason is TerminationReason.UNKNOWN_GOAL
    assert raw_run.result is not None
    assert raw_run.result.status is ResultStatus.FAILED
    assert raw_run.task is None
    assert tools.trace == []


def test_runtime_rejects_malformed_model_decision_safely() -> None:
    model = FakeModel({"goal": "calendar.find_30_minutes"})
    tools = FakeCalendarTools()

    raw_run = BoundedAgentRuntime(CalendarAgent(tools), model).run("Find time")

    assert raw_run.termination_reason is TerminationReason.INVALID_MODEL_DECISION
    assert raw_run.result is None
    assert raw_run.error == "model returned an invalid decision"
    assert tools.trace == []


def test_runtime_rejects_malformed_typed_decision_safely() -> None:
    model = FakeModel(SelectGoal("calendar.find_30_minutes", None, {}))
    tools = FakeCalendarTools()

    raw_run = BoundedAgentRuntime(CalendarAgent(tools), model).run("Find time")

    assert raw_run.termination_reason is TerminationReason.INVALID_MODEL_DECISION
    assert raw_run.result is None
    assert tools.trace == []


def test_runtime_clarification_does_not_run_calendar_agent() -> None:
    model = FakeModel(Clarify("Which audit do you mean?"))
    tools = FakeCalendarTools()

    raw_run = BoundedAgentRuntime(CalendarAgent(tools), model).run("Move after the audit")

    assert raw_run.termination_reason is TerminationReason.CLARIFICATION
    assert raw_run.result is None
    assert raw_run.error == "Which audit do you mean?"
    assert tools.trace == []


def test_runtime_fail_decision_terminates_without_running_calendar_agent() -> None:
    model = FakeModel(Fail("The request cannot be planned."))
    tools = FakeCalendarTools()

    raw_run = BoundedAgentRuntime(CalendarAgent(tools), model).run("Do the impossible")

    assert raw_run.termination_reason is TerminationReason.MODEL_FAILURE
    assert raw_run.result is None
    assert raw_run.error == "The request cannot be planned."
    assert tools.trace == []


def test_runtime_captures_model_failure() -> None:
    model = FakeModel(error=RuntimeError("model unavailable"))

    raw_run = BoundedAgentRuntime(CalendarAgent(FakeCalendarTools()), model).run("Find time")

    assert raw_run.termination_reason is TerminationReason.MODEL_FAILURE
    assert raw_run.error == "RuntimeError"
    assert raw_run.steps_used == 1
    assert raw_run.retries_used == 0


def test_runtime_enforces_retry_ceiling() -> None:
    model = FakeModel(error=RuntimeError("model unavailable"))
    policy = AgentPolicy(model=model, max_model_steps=5, retry_ceiling=2)

    raw_run = BoundedAgentRuntime(CalendarAgent(FakeCalendarTools()), model, policy=policy).run("Find time")

    assert raw_run.termination_reason is TerminationReason.MODEL_FAILURE
    assert raw_run.steps_used == 3
    assert raw_run.retries_used == 2
    assert model.calls == 3


def test_runtime_stops_at_zero_model_step_budget() -> None:
    model = FakeModel(SelectGoal("calendar.find_30_minutes", "Find time", {}))
    policy = AgentPolicy(model=model, max_model_steps=0)

    raw_run = BoundedAgentRuntime(CalendarAgent(FakeCalendarTools()), model, policy=policy).run("Find time")

    assert raw_run.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert raw_run.steps_used == 0
    assert model.calls == 0


def test_runtime_preserves_agent_result_and_tool_trace_in_raw_run() -> None:
    model = FakeModel(SelectGoal("calendar.find_30_minutes", "Find time", {}))
    tools = goal_one_tools()
    context = {
        "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }

    raw_run = BoundedAgentRuntime(CalendarAgent(tools), model).run("Find time", context)

    assert raw_run.result is not None
    assert raw_run.result.tool_calls == list(raw_run.tool_trace)
    assert raw_run.tool_trace == tuple(tools.trace)
    assert raw_run.supplied_context == context
    assert raw_run.task is not None
    assert raw_run.task.context == context


def test_model_success_language_does_not_override_structured_verification() -> None:
    model = FakeModel(SelectGoal(
        "calendar.find_30_minutes",
        "Success: the meeting is booked.",
        {},
    ))
    raw_run = BoundedAgentRuntime(CalendarAgent(FakeCalendarTools()), model).run("Book it")

    assert raw_run.result is not None
    assert raw_run.result.status is ResultStatus.NEEDS_CLARIFICATION
    verification = verify_result(raw_run.result, expected_status=ResultStatus.NEEDS_CLARIFICATION)
    assert verification.passed
    assert raw_run.termination_reason is TerminationReason.CLARIFICATION


def test_harness_runs_goal_one_through_bounded_runtime_and_verifies_result() -> None:
    model = FakeModel(SelectGoal("calendar.find_30_minutes", "Find time", {}))
    tools = goal_one_tools()
    context = {
        "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }

    report = Harness(tools).run(
        AgentTask("calendar.find_30_minutes", "Find time", context),
        expected_status=ResultStatus.PLANNED,
        model=model,
    )

    assert report.verification.passed
    assert report.result is not None
    assert report.raw_run is not None
    assert report.raw_run.task == AgentTask("calendar.find_30_minutes", "Find time", context)


def test_harness_runs_goal_two_through_bounded_runtime_and_verifies_result() -> None:
    model = FakeModel(SelectGoal(
        "calendar.move_after_audit",
        "Move everything after the audit",
        {"shift": timedelta(hours=1)},
    ))

    report = Harness(goal_two_tools()).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit"),
        expected_status=ResultStatus.PLANNED,
        model=model,
    )

    assert report.verification.passed
    assert report.result is not None
    assert [move.event_id for move in report.result.rescheduling_plan] == ["work-1"]


def test_harness_surfaces_model_clarification_without_claiming_success() -> None:
    report = Harness(FakeCalendarTools()).run(
        AgentTask("calendar.move_after_audit", "Move after the audit"),
        expected_status=ResultStatus.NEEDS_CLARIFICATION,
        model=FakeModel(Clarify("Which audit do you mean?")),
    )

    assert report.raw_run is not None
    assert report.raw_run.termination_reason is TerminationReason.CLARIFICATION
    assert report.result is None
    assert not report.verification.passed
    assert not report.verification.outcome_ok


def test_harness_surfaces_model_failure_without_claiming_success() -> None:
    report = Harness(FakeCalendarTools()).run(
        AgentTask("calendar.find_30_minutes", "Find time"),
        expected_status=ResultStatus.FAILED,
        model=FakeModel(Fail("Cannot plan this request.")),
    )

    assert report.raw_run is not None
    assert report.raw_run.termination_reason is TerminationReason.MODEL_FAILURE
    assert report.result is None
    assert not report.verification.passed


def test_harness_rejects_unknown_runtime_goal_before_planner() -> None:
    tools = FakeCalendarTools()
    report = Harness(tools).run(
        AgentTask("calendar.unknown", "Do something"),
        expected_status=ResultStatus.FAILED,
        model=FakeModel(SelectGoal("calendar.unknown", "Do something", {})),
    )

    assert report.raw_run is not None
    assert report.raw_run.termination_reason is TerminationReason.UNKNOWN_GOAL
    assert report.raw_run.task is None
    assert report.result is not None
    assert report.result.status is ResultStatus.FAILED
    assert report.verification.passed
    assert tools.trace == []


def test_harness_does_not_convert_runtime_error_into_success() -> None:
    report = Harness(FakeCalendarTools()).run(
        AgentTask("calendar.find_30_minutes", "Find time"),
        expected_status=ResultStatus.PLANNED,
        model=FakeModel(error=RuntimeError("model unavailable")),
    )

    assert report.raw_run is not None
    assert report.raw_run.termination_reason is TerminationReason.MODEL_FAILURE
    assert report.result is None
    assert not report.verification.passed
    assert not report.verification.outcome_ok


def test_harness_preserves_runtime_trace_for_independent_verification() -> None:
    tools = goal_one_tools()
    context = {
        "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }
    report = Harness(tools).run(
        AgentTask("calendar.find_30_minutes", "Find time", context),
        expected_status=ResultStatus.PLANNED,
        model=FakeModel(SelectGoal("calendar.find_30_minutes", "Find time", {})),
    )

    assert report.raw_run is not None
    assert report.result is not None
    assert report.raw_run.tool_trace
    assert report.result.tool_calls == list(report.raw_run.tool_trace)
    assert all(not call.mutating for call in report.raw_run.tool_trace)


def test_harness_does_not_treat_planned_as_completed_in_runtime_path() -> None:
    report = Harness(goal_one_tools()).run(
        AgentTask("calendar.find_30_minutes", "Success: booked", {
            "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
            "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
            "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
        }),
        expected_status=ResultStatus.COMPLETED,
        model=FakeModel(SelectGoal("calendar.find_30_minutes", "Success: booked", {})),
    )

    assert report.raw_run is not None
    assert report.raw_run.termination_reason is TerminationReason.SUCCEEDED
    assert report.result is not None
    assert report.result.status is ResultStatus.PLANNED
    assert not report.verification.passed
    assert not report.verification.outcome_ok