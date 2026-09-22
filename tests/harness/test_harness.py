from calendar_agent.models import AgentResult, AgentTask, ResultStatus, ToolCallRecord
from harness.assertions import verify_result
from harness.runner import Harness

from tests.unit.fakes import FakeCalendarTools


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