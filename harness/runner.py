"""Scenario runner with independent verification."""

from __future__ import annotations

from dataclasses import dataclass

from calendar_agent.agent import CalendarAgent
from calendar_agent.models import AgentResult, AgentTask, ResultStatus, VerificationResult
from calendar_agent.tools import CalendarTool

from .assertions import verify_result


@dataclass(frozen=True)
class HarnessReport:
    task: AgentTask
    result: AgentResult
    verification: VerificationResult


class Harness:
    def __init__(self, tools: CalendarTool) -> None:
        self.tools = tools

    def run(self, task: AgentTask, *, expected_status: ResultStatus) -> HarnessReport:
        result = CalendarAgent(self.tools).run(task)
        verification = verify_result(result, expected_status=expected_status)
        return HarnessReport(task=task, result=result, verification=verification)