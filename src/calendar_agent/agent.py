"""Calendar Agent orchestration without HTTP or mutation knowledge."""

from __future__ import annotations

from .models import AgentResult, AgentTask
from .planner import plan_task
from .tools import CalendarTool


class CalendarAgent:
    def __init__(self, tools: CalendarTool) -> None:
        self.tools = tools

    def run(self, task: AgentTask) -> AgentResult:
        trace_start = len(self.tools.trace)
        result = plan_task(task, self.tools)
        result.tool_calls.extend(self.tools.trace[trace_start:])
        return result