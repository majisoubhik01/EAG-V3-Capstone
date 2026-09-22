"""Goal dispatch kept intentionally small for the first implementation pass."""

from __future__ import annotations

from .goals.find_30_minutes import plan as plan_find_30_minutes
from .goals.move_after_audit import plan as plan_move_after_audit
from .models import AgentResult, AgentTask, ResultStatus
from .tools import CalendarTool


HANDLERS = {
    "calendar.find_30_minutes": plan_find_30_minutes,
    "calendar.move_after_audit": plan_move_after_audit,
}


def plan_task(task: AgentTask, tools: CalendarTool) -> AgentResult:
    handler = HANDLERS.get(task.goal)
    if handler is None:
        return AgentResult(
            goal=task.goal,
            status=ResultStatus.FAILED,
            summary=f"Unsupported goal: {task.goal}",
            ambiguities=["No handler is registered for this goal."],
        )
    return handler(task, tools)