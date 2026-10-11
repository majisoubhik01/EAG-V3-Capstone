"""Provider-neutral, bounded runtime for selecting an existing Calendar goal."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .agent import CalendarAgent
from .models import AgentResult, AgentTask, ResultStatus, ToolCallRecord
from .planner import HANDLERS
from .tools import ReadOnlyCalendarTool


@dataclass(frozen=True)
class ModelRequest:
    """Input supplied to a model without exposing runtime objects or secrets."""

    user_instruction: str
    available_goal_ids: tuple[str, ...]
    context: Mapping[str, Any]
    policy_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SelectGoal:
    """A model decision to invoke one registered goal handler."""

    goal: str
    instruction: str
    context: Mapping[str, Any]


@dataclass(frozen=True)
class Clarify:
    """A model decision that needs user clarification before goal execution."""

    reason: str


@dataclass(frozen=True)
class Fail:
    """A model decision that cannot produce an executable goal."""

    reason: str


ModelDecision = SelectGoal | Clarify | Fail


def _context_value_error(value: Any, path: str = "$", active: set[int] | None = None) -> str | None:
    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int, datetime, timedelta)):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"{path} contains a non-finite number"
    if isinstance(value, (Mapping, list, tuple)):
        value_id = id(value)
        if value_id in active:
            return f"{path} contains a cycle"
        active.add(value_id)
        try:
            items = value.items() if isinstance(value, Mapping) else enumerate(value)
            for key, item in items:
                if isinstance(value, Mapping) and not isinstance(key, str):
                    return f"{path} has a non-string key"
                item_error = _context_value_error(item, f"{path}.{key}", active)
                if item_error is not None:
                    return item_error
        except Exception as exc:
            return f"{path} could not be inspected: {type(exc).__name__}"
        finally:
            active.remove(value_id)
        return None
    return f"{path} contains unsupported type {type(value).__name__}"


def _decision_context_error(context: Any) -> str | None:
    if not isinstance(context, Mapping):
        return "context must be a Mapping"
    return _context_value_error(context, "context")


class Model(Protocol):
    """Minimal provider-neutral model contract."""

    def decide(self, request: ModelRequest) -> ModelDecision: ...


class TerminationReason(str, Enum):
    SUCCEEDED = "succeeded"
    CLARIFICATION = "clarification"
    MODEL_FAILURE = "model_failure"
    INVALID_MODEL_DECISION = "invalid_model_decision"
    UNKNOWN_GOAL = "unknown_goal"
    EXECUTION_FAILURE = "execution_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class AgentPolicy:
    """Boundaries and injected model configuration for one runtime."""

    model: Model
    allowed_goals: tuple[str, ...] | None = None
    max_model_steps: int = 1
    retry_ceiling: int = 0
    read_only: bool = True
    prompt_ref: str | None = None

    def __post_init__(self) -> None:
        if self.max_model_steps < 0:
            raise ValueError("max_model_steps cannot be negative")
        if self.retry_ceiling < 0:
            raise ValueError("retry_ceiling cannot be negative")
        if not self.read_only:
            raise ValueError("the Calendar runtime is read-only")

    @property
    def goal_ids(self) -> tuple[str, ...]:
        if self.allowed_goals is None:
            return tuple(HANDLERS)
        return self.allowed_goals

    def metadata(self) -> dict[str, Any]:
        """Return replay metadata without the injected model or credentials."""

        return {
            "allowed_goals": self.goal_ids,
            "max_model_steps": self.max_model_steps,
            "retry_ceiling": self.retry_ceiling,
            "read_only": self.read_only,
            "prompt_ref": self.prompt_ref,
        }


@dataclass(frozen=True)
class RawRun:
    """In-memory evidence sufficient for later scoring without another model call."""

    user_instruction: str
    supplied_context: dict[str, Any]
    policy_metadata: dict[str, Any]
    model_decisions: tuple[object, ...]
    task: AgentTask | None
    result: AgentResult | None
    tool_trace: tuple[ToolCallRecord, ...]
    error: str | None
    steps_used: int
    retries_used: int
    termination_reason: TerminationReason
    run_id: str = field(default_factory=lambda: str(uuid4()))


class BoundedAgentRuntime:
    """Select and execute one existing goal within explicit model boundaries."""

    def __init__(
        self,
        calendar_agent: CalendarAgent,
        model: Model,
        *,
        policy: AgentPolicy | None = None,
    ) -> None:
        self.calendar_agent = calendar_agent
        self.model = model
        self.policy = policy or AgentPolicy(model=model)
        if self.policy.model is not model:
            raise ValueError("policy.model must be the injected model")
        if self.policy.read_only:
            self.calendar_agent.tools = ReadOnlyCalendarTool(self.calendar_agent.tools)

    def run(self, user_instruction: str, context: Mapping[str, Any] | None = None) -> RawRun:
        supplied_context = dict(context or {})
        decisions: list[object] = []
        task: AgentTask | None = None
        result: AgentResult | None = None
        tool_trace: tuple[ToolCallRecord, ...] = ()
        steps_used = 0
        retries_used = 0

        if self.policy.max_model_steps == 0:
            return self._raw_run(
                user_instruction,
                supplied_context,
                decisions,
                task,
                result,
                tool_trace,
                "model step budget exhausted",
                steps_used,
                retries_used,
                TerminationReason.BUDGET_EXHAUSTED,
            )

        request = ModelRequest(
            user_instruction=user_instruction,
            available_goal_ids=self.policy.goal_ids,
            context=dict(supplied_context),
            policy_metadata=self.policy.metadata(),
        )

        while steps_used < self.policy.max_model_steps:
            steps_used += 1
            try:
                decision = self.model.decide(request)
            except Exception as exc:
                if retries_used < self.policy.retry_ceiling and steps_used < self.policy.max_model_steps:
                    retries_used += 1
                    continue
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    type(exc).__name__,
                    steps_used,
                    retries_used,
                    TerminationReason.MODEL_FAILURE,
                )

            decisions.append(decision)
            if isinstance(decision, Clarify):
                if not isinstance(decision.reason, str) or not decision.reason.strip():
                    return self._raw_run(
                        user_instruction,
                        supplied_context,
                        decisions,
                        task,
                        result,
                        tool_trace,
                        "Clarify.reason must be non-empty",
                        steps_used,
                        retries_used,
                        TerminationReason.INVALID_MODEL_DECISION,
                    )
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    decision.reason,
                    steps_used,
                    retries_used,
                    TerminationReason.CLARIFICATION,
                )
            if isinstance(decision, Fail):
                if not isinstance(decision.reason, str) or not decision.reason.strip():
                    return self._raw_run(
                        user_instruction,
                        supplied_context,
                        decisions,
                        task,
                        result,
                        tool_trace,
                        "Fail.reason must be non-empty",
                        steps_used,
                        retries_used,
                        TerminationReason.INVALID_MODEL_DECISION,
                    )
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    decision.reason,
                    steps_used,
                    retries_used,
                    TerminationReason.MODEL_FAILURE,
                )
            if not isinstance(decision, SelectGoal):
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    "model returned an invalid decision",
                    steps_used,
                    retries_used,
                    TerminationReason.INVALID_MODEL_DECISION,
                )
            if not isinstance(decision.goal, str) or not decision.goal.strip():
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    "SelectGoal.goal must be a non-empty string",
                    steps_used,
                    retries_used,
                    TerminationReason.INVALID_MODEL_DECISION,
                )
            if decision.goal not in self.policy.goal_ids or decision.goal not in HANDLERS:
                failed = AgentResult(
                    goal=decision.goal,
                    status=ResultStatus.FAILED,
                    summary=f"Unknown goal: {decision.goal}",
                    ambiguities=["The model selected an unregistered goal."],
                )
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    failed,
                    tool_trace,
                    "unknown goal",
                    steps_used,
                    retries_used,
                    TerminationReason.UNKNOWN_GOAL,
                )
            if not isinstance(decision.instruction, str) or not decision.instruction.strip():
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    "SelectGoal contains invalid instruction or context",
                    steps_used,
                    retries_used,
                    TerminationReason.INVALID_MODEL_DECISION,
                )

            try:
                decision_context = dict(decision.context)
            except Exception as exc:
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    f"SelectGoal.context could not be copied: {type(exc).__name__}",
                    steps_used,
                    retries_used,
                    TerminationReason.INVALID_MODEL_DECISION,
                )
            context_error = _decision_context_error(decision_context)
            if context_error is not None:
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    f"Invalid SelectGoal.context: {context_error}",
                    steps_used,
                    retries_used,
                    TerminationReason.INVALID_MODEL_DECISION,
                )

            task = AgentTask(
                goal=decision.goal,
                instruction=decision.instruction,
                context={**decision_context, **supplied_context},
            )
            trace_start = len(self.calendar_agent.tools.trace)
            try:
                result = self.calendar_agent.run(task)
            except Exception as exc:
                tool_trace = tuple(self.calendar_agent.tools.trace[trace_start:])
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    type(exc).__name__,
                    steps_used,
                    retries_used,
                    TerminationReason.EXECUTION_FAILURE,
                )
            tool_trace = tuple(self.calendar_agent.tools.trace[trace_start:])
            denied_mutation = next((call for call in tool_trace if call.mutating), None)
            if denied_mutation is not None:
                result.status = ResultStatus.FAILED
                result.summary = "A Calendar mutation was denied by the read-only runtime policy."
                result.ambiguities.append("The runtime blocked a Calendar mutation attempt.")
                result.claimed_outcome.clear()
                result.tool_calls = list(tool_trace)
                return self._raw_run(
                    user_instruction,
                    supplied_context,
                    decisions,
                    task,
                    result,
                    tool_trace,
                    f"read-only policy denied mutation: {denied_mutation.tool}",
                    steps_used,
                    retries_used,
                    TerminationReason.EXECUTION_FAILURE,
                )
            if result.status is ResultStatus.PLANNED:
                termination = TerminationReason.SUCCEEDED
                result_error = None
            elif result.status is ResultStatus.NEEDS_CLARIFICATION:
                termination = TerminationReason.CLARIFICATION
                result_error = None
            elif result.status is ResultStatus.FAILED:
                termination = TerminationReason.EXECUTION_FAILURE
                result_error = None
            else:
                termination = TerminationReason.EXECUTION_FAILURE
                result_error = f"unsupported result status: {result.status!r}"
            return self._raw_run(
                user_instruction,
                supplied_context,
                decisions,
                task,
                result,
                tool_trace,
                result_error,
                steps_used,
                retries_used,
                termination,
            )

        return self._raw_run(
            user_instruction,
            supplied_context,
            decisions,
            task,
            result,
            tool_trace,
            "model step budget exhausted",
            steps_used,
            retries_used,
            TerminationReason.BUDGET_EXHAUSTED,
        )

    def _raw_run(
        self,
        user_instruction: str,
        supplied_context: dict[str, Any],
        decisions: list[object],
        task: AgentTask | None,
        result: AgentResult | None,
        tool_trace: tuple[ToolCallRecord, ...],
        error: str | None,
        steps_used: int,
        retries_used: int,
        termination_reason: TerminationReason,
    ) -> RawRun:
        return RawRun(
            user_instruction=user_instruction,
            supplied_context=supplied_context,
            policy_metadata=self.policy.metadata(),
            model_decisions=tuple(decisions),
            task=task,
            result=result,
            tool_trace=tool_trace,
            error=error,
            steps_used=steps_used,
            retries_used=retries_used,
            termination_reason=termination_reason,
        )