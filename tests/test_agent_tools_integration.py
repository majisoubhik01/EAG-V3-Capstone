from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from calendar_agent import (
    AgentSwitchCalendarTools,
    AgentSwitchToolError,
    CalendarAgent,
    MutationDisabledError,
    ResultStatus,
)
from calendar_agent.models import AgentTask


MUTATION_TOOLS = {
    "CalendarEvent.create",
    "CalendarEvent.update",
    "CalendarEvent.delete",
    "CalendarEvent.confirm",
    "CalendarEvent.mark_tentative",
    "CalendarEvent.cancel",
}


class FakeAgentSwitchClient:
    def __init__(
        self,
        responses: dict[str, list[dict[str, Any]]],
        *,
        failure: tuple[str, Exception] | None = None,
    ) -> None:
        self.responses = {name: list(values) for name, values in responses.items()}
        self.failure = failure
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if self.failure is not None and name == self.failure[0]:
            raise self.failure[1]
        values = self.responses[name]
        if not values:
            raise AssertionError(f"No fake MCP response remains for {name}")
        return values.pop(0)


def records(*items: dict[str, Any]) -> dict[str, Any]:
    return {"structuredContent": {"data": list(items)}}


def free_busy_result() -> dict[str, Any]:
    return {
        "structuredContent": {
            "result": {
                "subjects": [{
                    "subject": "plant@example.test",
                    "visibility": "free_busy",
                    "busy": [],
                    "calendars": [{"access_level": "free_busy"}],
                }]
            }
        }
    }


def test_goal_one_runs_through_mcp_adapter_to_agent_result() -> None:
    start = datetime(2026, 9, 22, 9, tzinfo=timezone.utc)
    end = datetime(2026, 9, 22, 17, tzinfo=timezone.utc)
    client = FakeAgentSwitchClient({
        "Party.list": [
            records({
                "id": "plant-1",
                "name": "Plant Manager",
                "email": "plant@example.test",
                "job_title": "Plant Manager",
            }),
            records({
                "id": "plant-1",
                "name": "Plant Manager",
                "email": "plant@example.test",
                "job_title": "Plant Manager",
            }),
        ],
        "Calendar.list": [records({"id": "calendar-1", "owner_party_id": "plant-1"})],
        "SchedulingPreferences.list": [records({"default_meeting_duration": 30, "min_notice_hours": 0})],
        "AvailabilityRule.list": [records({
            "id": "rule-1",
            "party_id": "plant-1",
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            "date_overrides": [],
        })],
        "endpoint.calendar.free_busy": [free_busy_result()],
    })
    tools = AgentSwitchCalendarTools(client)

    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.find_30_minutes",
            "Find 30 minutes with the plant head",
            {"now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc), "start": start, "end": end},
        )
    )

    assert result.status is ResultStatus.PLANNED
    assert len(result.candidate_slots) == 1
    assert result.candidate_slots[0].start_at == "2026-09-22T09:00:00+00:00"
    assert result.candidate_slots[0].end_at == "2026-09-22T09:30:00+00:00"
    assert [name for name, _ in client.calls] == [
        "Party.list",
        "Party.list",
        "Calendar.list",
        "SchedulingPreferences.list",
        "AvailabilityRule.list",
        "endpoint.calendar.free_busy",
    ]
    assert client.calls[0][1] == {"limit": 1000, "job_title": "Plant Manager"}
    assert client.calls[1][1] == {"limit": 1000, "job_title": "Plant Head"}
    assert client.calls[2][1] == {"owner_party_id": "plant-1", "limit": 1000}
    assert client.calls[3][1] == {"limit": 1000}
    assert client.calls[4][1] == {"limit": 1000}
    assert client.calls[5][1] == {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "emails": ["plant@example.test"],
        "calendar_ids": ["calendar-1"],
    }
    assert [call.tool for call in result.tool_calls] == [name for name, _ in client.calls]
    assert not any(name in MUTATION_TOOLS for name, _ in client.calls)


def test_goal_two_runs_through_mcp_adapter_to_agent_result() -> None:
    audit = {
        "id": "audit-1",
        "title": "Internal quality audit",
        "description": "ISO review",
        "start_at": "2026-09-22T09:00:00+00:00",
        "end_at": "2026-09-22T10:00:00+00:00",
        "timezone": "UTC",
        "calendar_id": "calendar-1",
    }
    after = {
        "id": "work-1",
        "title": "Work",
        "start_at": "2026-09-22T11:00:00+00:00",
        "end_at": "2026-09-22T12:00:00+00:00",
        "timezone": "UTC",
        "calendar_id": "calendar-1",
    }
    client = FakeAgentSwitchClient({
        "CalendarEvent.list": [records(audit), records(audit, after)],
    })
    tools = AgentSwitchCalendarTools(client)

    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": timedelta(hours=1)},
        )
    )

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["work-1"]
    assert result.rescheduling_plan[0].proposed_start_at == "2026-09-22T12:00:00+00:00"
    assert result.rescheduling_plan[0].proposed_end_at == "2026-09-22T13:00:00+00:00"
    assert result.claimed_outcome["moved"] is False
    assert client.calls == [
        ("CalendarEvent.list", {"limit": 1000, "search": "audit"}),
        ("CalendarEvent.list", {"limit": 1000}),
    ]
    assert [call.tool for call in result.tool_calls] == [name for name, _ in client.calls]
    assert not any(name in MUTATION_TOOLS for name, _ in client.calls)


def test_adapter_read_failure_is_recorded_and_propagates() -> None:
    client = FakeAgentSwitchClient(
        {"Party.list": []},
        failure=("Party.list", AgentSwitchToolError("fake MCP failure")),
    )
    tools = AgentSwitchCalendarTools(client)

    with pytest.raises(AgentSwitchToolError, match="fake MCP failure"):
        CalendarAgent(tools).run(
            AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
        )

    assert tools.trace[0].tool == "Party.list"
    assert not tools.trace[0].succeeded
    assert tools.trace[0].error == "AgentSwitchToolError"


def test_adapter_mutation_boundary_fails_closed() -> None:
    client = FakeAgentSwitchClient({})
    tools = AgentSwitchCalendarTools(client)

    with pytest.raises(MutationDisabledError):
        tools.update_event(id="work-1")

    assert client.calls == []
    assert tools.trace[-1].mutating
    assert not tools.trace[-1].succeeded