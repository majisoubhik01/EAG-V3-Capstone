"""Deterministic fixtures for executable Harness demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Sequence

from calendar_agent.models import AgentTask, MutationDisabledError, ResultStatus, ToolCallRecord
from calendar_agent.runtime import ModelRequest, SelectGoal

from .scenarios.find_30_minutes import task as find_30_minutes_task
from .phase1 import (
    Phase1PartyCalendarTools,
    PartyDirectory,
    attendee_consistency_errors,
    assert_planning_side_effect_free,
    authoritative_party_matches,
    audit_fixture_events,
    booking_consistency_errors,
    duration_satisfies_goal,
    event_attendee_fixtures,
    execution_evidence_errors,
    execution_evidence_from_trace,
    free_busy_fixtures,
    free_busy_authority_errors,
    goal1_duration_fixture,
    phase1_party_directory,
    phase1_party_records,
    phase1_tool_policy,
    provider_401_fixture,
    subscription_fixtures,
    workflow_actions,
)


class DeterministicCalendarTools:
    """In-memory read-only CalendarTool implementation for local demonstrations."""

    def __init__(self) -> None:
        self.parties = [{
            "id": "plant-1",
            "email": "plant@example.test",
            "job_title": "Plant Manager",
        }]
        self.events: list[dict[str, Any]] = []
        self.free_busy_result = {"result": {"subjects": [{
            "subject": "plant@example.test",
            "visibility": "free_busy",
            "busy": [],
            "calendars": [{"access_level": "free_busy"}],
        }]}}
        self.calendars = [{"id": "calendar-1", "owner_party_id": "plant-1"}]
        self.availability_rules = [{
            "id": "rule-1",
            "party_id": "plant-1",
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            "date_overrides": [],
        }]
        self.scheduling_preferences = [{"default_meeting_duration": 30, "min_notice_hours": 0}]
        self.trace: list[ToolCallRecord] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> None:
        self.trace.append(ToolCallRecord(name, arguments))

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        self._record("Party.list", filters)
        job_title = filters.get("job_title")
        if job_title is None:
            return self.parties
        return [
            party for party in self.parties
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


class FixedGoalModel:
    """Provider-free model stand-in that selects the known demo goal."""

    def __init__(self, decision: SelectGoal) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, request: ModelRequest) -> SelectGoal:
        self.calls += 1
        return self.decision


@dataclass(frozen=True)
class DeterministicScenario:
    task: AgentTask
    tools: DeterministicCalendarTools
    model: FixedGoalModel
    expected_status: ResultStatus


def deterministic_scenario() -> DeterministicScenario:
    task = replace(
        find_30_minutes_task(),
        context={
            "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
            "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
            "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
        },
    )
    model = FixedGoalModel(SelectGoal(task.goal, task.instruction, {}))
    return DeterministicScenario(
        task=task,
        tools=DeterministicCalendarTools(),
        model=model,
        expected_status=ResultStatus.PLANNED,
    )