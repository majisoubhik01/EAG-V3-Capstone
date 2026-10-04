"""Explicit Calendar tool boundary.

The adapter exposes read operations and records every call. Mutation methods are
present as a visible boundary, but fail closed and are never used by the first
planning-only goal handlers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence

from .agentswitch_client import AgentSwitchClient
from .models import MutationDisabledError, ToolCallRecord


class CalendarTool(Protocol):
    """Operations the agent may use without knowing about HTTP or MCP."""

    trace: list[ToolCallRecord]

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]: ...

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]: ...

    def get_calendar(self, calendar_id: str) -> dict[str, Any]: ...

    def list_events(self, **filters: Any) -> list[dict[str, Any]]: ...

    def get_event(self, event_id: str) -> dict[str, Any]: ...

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]: ...

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]: ...

    def create_event(self, **arguments: Any) -> dict[str, Any]: ...

    def update_event(self, **arguments: Any) -> dict[str, Any]: ...

    def delete_event(self, event_id: str) -> dict[str, Any]: ...

    def confirm_event(self, event_id: str) -> dict[str, Any]: ...

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]: ...

    def cancel_event(self, event_id: str) -> dict[str, Any]: ...


class ReadOnlyCalendarTool:
    """Delegate reads while preventing mutations from reaching the wrapped tool."""

    def __init__(self, delegate: CalendarTool) -> None:
        self._delegate = delegate

    @property
    def trace(self) -> list[ToolCallRecord]:
        return self._delegate.trace

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.find_parties(**filters)

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_calendars(**filters)

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        return self._delegate.get_calendar(calendar_id)

    def list_events(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_events(**filters)

    def get_event(self, event_id: str) -> dict[str, Any]:
        return self._delegate.get_event(event_id)

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self._delegate.free_busy(
            start=start,
            end=end,
            emails=emails,
            calendar_ids=calendar_ids,
        )

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_availability_rules(**filters)

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_scheduling_preferences(**filters)

    def _mutation_disabled(self, tool: str, arguments: dict[str, Any]) -> None:
        self.trace.append(ToolCallRecord(tool, arguments, mutating=True, succeeded=False, error="disabled"))
        raise MutationDisabledError(f"Mutation tool disabled: {tool}")

    def create_event(self, **arguments: Any) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.create", arguments)
        return {}

    def update_event(self, **arguments: Any) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.update", arguments)
        return {}

    def delete_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.delete", {"id": event_id})
        return {}

    def confirm_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.confirm", {"id": event_id})
        return {}

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.mark_tentative", {"id": event_id})
        return {}

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.cancel", {"id": event_id})
        return {}


class AgentSwitchCalendarTools:
    """Translate typed calendar operations into MCP tool calls."""

    def __init__(self, client: AgentSwitchClient) -> None:
        self._client = client
        self.trace: list[ToolCallRecord] = []

    def _read(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._client.call_tool(tool, arguments)
        except Exception as exc:
            self.trace.append(ToolCallRecord(tool, arguments, error=type(exc).__name__, succeeded=False))
            raise
        self.trace.append(ToolCallRecord(tool, arguments))
        return result

    @staticmethod
    def _records(result: dict[str, Any]) -> list[dict[str, Any]]:
        structured = result.get("structuredContent", {})
        records = structured.get("data", []) if isinstance(structured, dict) else []
        return records if isinstance(records, list) else []

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("Party.list", filters))

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("Calendar.list", filters))

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        return self._read("Calendar.get", {"id": calendar_id}).get("structuredContent", {})

    def list_events(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("CalendarEvent.list", filters))

    def get_event(self, event_id: str) -> dict[str, Any]:
        return self._read("CalendarEvent.get", {"id": event_id}).get("structuredContent", {})

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        if emails is not None:
            arguments["emails"] = list(emails)
        if calendar_ids is not None:
            arguments["calendar_ids"] = list(calendar_ids)
        return self._read("endpoint.calendar.free_busy", arguments).get("structuredContent", {})

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("AvailabilityRule.list", filters))

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("SchedulingPreferences.list", filters))

    def _mutation_disabled(self, tool: str, arguments: dict[str, Any]) -> None:
        self.trace.append(ToolCallRecord(tool, arguments, mutating=True, succeeded=False, error="disabled"))
        raise MutationDisabledError(f"Mutation tool disabled: {tool}")

    def create_event(self, **arguments: Any) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.create", arguments)
        return {}

    def update_event(self, **arguments: Any) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.update", arguments)
        return {}

    def delete_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.delete", {"id": event_id})
        return {}

    def confirm_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.confirm", {"id": event_id})
        return {}

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.mark_tentative", {"id": event_id})
        return {}

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.cancel", {"id": event_id})
        return {}