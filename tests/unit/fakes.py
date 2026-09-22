from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from calendar_agent.models import MutationDisabledError, ToolCallRecord


class FakeCalendarTools:
    def __init__(
        self,
        *,
        parties: list[dict[str, Any]] | None = None,
        events: list[dict[str, Any]] | None = None,
        free_busy_result: dict[str, Any] | None = None,
        calendars: list[dict[str, Any]] | None = None,
        availability_rules: list[dict[str, Any]] | None = None,
        scheduling_preferences: list[dict[str, Any]] | None = None,
    ) -> None:
        self.parties = [] if parties is None else parties
        self.events = events or []
        self.free_busy_result = free_busy_result or {}
        self.calendars = calendars or []
        self.availability_rules = availability_rules or []
        self.scheduling_preferences = scheduling_preferences or []
        self.trace: list[ToolCallRecord] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> None:
        self.trace.append(ToolCallRecord(name, arguments))

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        self._record("Party.list", filters)
        job_title = filters.get("job_title")
        if job_title is None:
            return self.parties
        return [
            party
            for party in self.parties
            if str(party.get("job_title") or "").strip().lower() == str(job_title).strip().lower()
        ]

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]:
        self._record("Calendar.list", filters)
        owner_party_id = filters.get("owner_party_id")
        if owner_party_id is None:
            return self.calendars
        return [calendar for calendar in self.calendars if calendar.get("owner_party_id") == owner_party_id]

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        self._record("Calendar.get", {"id": calendar_id})
        return {}

    def list_events(self, **filters: Any) -> list[dict[str, Any]]:
        self._record("CalendarEvent.list", filters)
        return self.events

    def get_event(self, event_id: str) -> dict[str, Any]:
        self._record("CalendarEvent.get", {"id": event_id})
        return {}

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        self._record(
            "endpoint.calendar.free_busy",
            {"start": start.isoformat(), "end": end.isoformat(), "emails": list(emails or [])},
        )
        return self.free_busy_result

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]:
        self._record("AvailabilityRule.list", filters)
        return self.availability_rules

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]:
        self._record("SchedulingPreferences.list", filters)
        return self.scheduling_preferences

    def _mutation(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.trace.append(ToolCallRecord(name, arguments, mutating=True, succeeded=False, error="disabled"))
        raise MutationDisabledError(name)

    def create_event(self, **arguments: Any) -> dict[str, Any]:
        return self._mutation("CalendarEvent.create", arguments)

    def update_event(self, **arguments: Any) -> dict[str, Any]:
        return self._mutation("CalendarEvent.update", arguments)

    def delete_event(self, event_id: str) -> dict[str, Any]:
        return self._mutation("CalendarEvent.delete", {"id": event_id})

    def confirm_event(self, event_id: str) -> dict[str, Any]:
        return self._mutation("CalendarEvent.confirm", {"id": event_id})

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]:
        return self._mutation("CalendarEvent.mark_tentative", {"id": event_id})

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        return self._mutation("CalendarEvent.cancel", {"id": event_id})