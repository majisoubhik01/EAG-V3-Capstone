"""Offline Phase 1 fixtures and validators for the independent Harness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Iterable, Sequence

from calendar_agent.models import (
    AgentJob,
    AgentJobStep,
    AgentMessage,
    AgentToolPolicy,
    AuthorityFailure,
    AuthorityFailureKind,
    CalendarSubscription,
    EventAttendee,
    FreeBusyInterval,
    MeetingBooking,
    PartyPage,
    PartyRecord,
    ProviderFailure,
    RiskLevel,
    SchedulingWorkflowAction,
    ToolCallRecord,
)


@dataclass(frozen=True)
class HostPoolBoundary:
    resource: str = "HostPool"
    accessible: bool = False
    reason: str = "HostPool is outside the scheduling agent authority boundary."


@dataclass(frozen=True)
class PartyDirectory:
    records: tuple[PartyRecord, ...]
    page_size: int = 100

    def page(
        self,
        *,
        page: int = 1,
        query: str | None = None,
        job_title: str | None = None,
    ) -> PartyPage:
        if page < 1:
            raise ValueError("page must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        normalized_query = (query or "").strip().lower()
        normalized_title = (job_title or "").strip().lower()
        filtered = [
            party
            for party in self.records
            if (not normalized_query or normalized_query in " ".join((party.name, party.email, party.job_title)).lower())
            and (not normalized_title or party.job_title.strip().lower() == normalized_title)
        ]
        start = (page - 1) * self.page_size
        records = tuple(filtered[start : start + self.page_size])
        return PartyPage(
            records=records,
            page=page,
            page_size=self.page_size,
            total_count=len(filtered),
            has_more=start + len(records) < len(filtered),
        )

    def all_pages(self, *, query: str | None = None, job_title: str | None = None) -> tuple[PartyPage, ...]:
        pages: list[PartyPage] = []
        page_number = 1
        while True:
            current = self.page(page=page_number, query=query, job_title=job_title)
            pages.append(current)
            if not current.has_more:
                return tuple(pages)
            page_number += 1


class Phase1PartyCalendarTools:
    """Minimal read-only adapter used to exercise planner pagination offline."""

    def __init__(self) -> None:
        self.directory = phase1_party_directory()
        self.trace: list[ToolCallRecord] = []

    def find_parties_page(self, **filters: Any) -> PartyPage:
        self.trace.append(ToolCallRecord("Party.list", dict(filters)))
        return self.directory.page(page=int(filters.get("page", 1)))

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        return [
            {"id": party.id, "name": party.name, "email": party.email, "job_title": party.job_title}
            for party in self.directory.page(page=1).records
        ]


def phase1_party_records() -> list[dict[str, str]]:
    """Return 230 plausible records with no authoritative plant-role match."""

    first_names = ("Aarav", "Isha", "Kabir", "Meera", "Nikhil", "Riya", "Vihaan", "Zoya")
    last_names = ("Shah", "Patil", "Kulkarni", "Joshi", "Rao", "Mehta", "Desai")
    records: list[dict[str, str]] = []
    for index in range(230):
        first_name = first_names[index % len(first_names)]
        last_name = last_names[(index // len(first_names)) % len(last_names)]
        records.append(
            {
                "id": f"party-{index + 1:03d}",
                "name": f"{first_name} {last_name}",
                "email": f"employee{index + 1:03d}@example.test",
                "job_title": "" if index % 3 == 0 else "Operations Specialist",
            }
        )
    return records


def phase1_party_directory() -> PartyDirectory:
    return PartyDirectory(
        records=tuple(
            PartyRecord(
                id=record["id"],
                name=record["name"],
                email=record["email"],
                job_title=record["job_title"],
            )
            for record in phase1_party_records()
        )
    )


def authoritative_party_matches(records: Iterable[PartyRecord], query: str) -> tuple[PartyRecord, ...]:
    normalized = query.strip().lower()
    return tuple(
        party
        for party in records
        if party.job_title.strip().lower() in {"plant manager", "plant head"}
        and normalized in party.job_title.strip().lower()
    )


def event_attendee_fixtures() -> tuple[EventAttendee, ...]:
    return (
        EventAttendee("audit-1", "party-001", "employee001@example.test", "required", "accepted"),
        EventAttendee("audit-1", "party-002", "employee002@example.test", "optional", "tentative"),
        EventAttendee("audit-2", "party-003", "employee003@example.test", "required", "declined"),
        EventAttendee("audit-2", "party-004", "employee004@example.test", "organizer", "accepted"),
        EventAttendee("booking-1", None, "chair@example.test", "chair", "needs_action"),
    )


def attendee_consistency_errors(
    events: Iterable[dict[str, Any]],
    attendees: Iterable[EventAttendee],
) -> tuple[str, ...]:
    event_list = list(events)
    attendee_list = list(attendees)
    event_ids = {event.get("id") for event in event_list}
    errors: list[str] = []
    for attendee in attendee_list:
        if attendee.event_id not in event_ids:
            errors.append(f"attendee {attendee.email} references missing event {attendee.event_id}")
    for event in event_list:
        event_id = event.get("id")
        rows = [attendee for attendee in attendee_list if attendee.event_id == event_id]
        participant_count = event.get("participant_count")
        if isinstance(participant_count, int) and participant_count != len(rows):
            errors.append(f"event {event_id} participant_count disagrees with attendee rows")
        if participant_count and not rows:
            errors.append(f"event {event_id} has participant metadata but no attendee rows")
    return tuple(errors)


def required_attendee_feasible(attendees: Iterable[EventAttendee]) -> bool:
    return all(not attendee.required or attendee.response_status != "declined" for attendee in attendees)


def _parse_aware(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


@dataclass(frozen=True)
class NormalizedFreeBusy:
    intervals: tuple[tuple[datetime, datetime], ...]
    errors: tuple[str, ...] = ()
    authoritative: bool = True


def normalize_free_busy(intervals: Iterable[FreeBusyInterval | dict[str, Any]]) -> NormalizedFreeBusy:
    normalized: list[tuple[datetime, datetime]] = []
    errors: list[str] = []
    for index, item in enumerate(intervals):
        if isinstance(item, FreeBusyInterval):
            start_value, end_value = item.start_at, item.end_at
        elif isinstance(item, dict):
            start_value = item.get("start") or item.get("start_at")
            end_value = item.get("end") or item.get("end_at")
        else:
            errors.append(f"interval {index} is not a supported record")
            continue
        start = _parse_aware(start_value)
        end = _parse_aware(end_value)
        if start is None or end is None:
            errors.append(f"interval {index} has a malformed or naive timestamp")
            continue
        if end <= start:
            errors.append(f"interval {index} does not have a positive duration")
            continue
        normalized.append((start.astimezone(timezone.utc), end.astimezone(timezone.utc)))
    return NormalizedFreeBusy(tuple(normalized), tuple(errors), not errors)


def free_busy_fixtures() -> dict[str, Any]:
    return {
        "timezone_aware": [{"start": "2026-09-22T09:00:00+00:00", "end": "2026-09-22T09:30:00+00:00"}],
        "utc": [{"start": "2026-09-22T09:00:00Z", "end": "2026-09-22T09:30:00Z"}],
        "asia_kolkata": [{"start": "2026-09-22T14:30:00+05:30", "end": "2026-09-22T15:00:00+05:30"}],
        "explicit_offset": [{"start": "2026-09-22T05:00:00-04:00", "end": "2026-09-22T05:30:00-04:00"}],
        "malformed": [{"start": "not-a-date", "end": "2026-09-22T09:30:00Z"}],
        "duplicate": [
            {"start": "2026-09-22T09:00:00Z", "end": "2026-09-22T09:30:00Z"},
            {"start": "2026-09-22T09:00:00Z", "end": "2026-09-22T09:30:00Z"},
        ],
        "overlapping": [
            {"start": "2026-09-22T09:00:00Z", "end": "2026-09-22T10:00:00Z"},
            {"start": "2026-09-22T09:30:00Z", "end": "2026-09-22T10:30:00Z"},
        ],
        "stale": {"last_updated": "2026-09-19T09:00:00Z", "max_age_hours": 24},
        "inaccessible": {"access_level": "none", "authoritative": False},
        "empty": [],
        "capped_paginated": {"page_size": 100, "total_count": 230, "has_more": True},
    }


def free_busy_authority_errors(
    metadata: dict[str, Any],
    *,
    now: datetime,
    max_age: timedelta,
) -> tuple[str, ...]:
    """Validate local free/busy evidence metadata, not live provider authority."""

    errors: list[str] = []
    if metadata.get("access_level") == "none" or metadata.get("authoritative") is False:
        errors.append("free/busy calendar is inaccessible")
    if metadata.get("has_more") is True:
        errors.append("free/busy result is capped or paginated")
    updated = _parse_aware(metadata.get("last_updated"))
    if updated is None:
        errors.append("free/busy freshness timestamp is missing or malformed")
    elif now - updated > max_age:
        errors.append("free/busy result is stale")
    return tuple(errors)


@dataclass(frozen=True)
class SchedulingLink:
    id: str
    title: str
    duration_minutes: int


def goal1_duration_fixture() -> dict[str, Any]:
    return {
        "default_duration_minutes": 30,
        "scheduling_link": SchedulingLink("link-plant-visit", "Plant visit - Chakan", 60),
    }


def duration_satisfies_goal(required_minutes: int, link: SchedulingLink) -> bool:
    return required_minutes == link.duration_minutes


def audit_fixture_events() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "audit-1",
            "title": "Supplier quality audit",
            "start_at": "2026-09-22T10:00:00+00:00",
            "end_at": "2026-09-22T11:30:00+00:00",
            "timezone": "UTC",
            "calendar_id": "calendar-1",
            "pre_buffer_minutes": 15,
            "post_buffer_minutes": 30,
            "minimum_notice_hours": 72,
        },
        {"id": "before", "title": "Preparation", "start_at": "2026-09-22T08:00:00+00:00", "end_at": "2026-09-22T09:00:00+00:00", "timezone": "UTC", "calendar_id": "calendar-1"},
        {"id": "ends-at-audit", "title": "Handover", "start_at": "2026-09-22T10:30:00+00:00", "end_at": "2026-09-22T11:30:00+00:00", "timezone": "UTC", "calendar_id": "calendar-1"},
        {"id": "starts-at-audit-end", "title": "Boundary handoff", "start_at": "2026-09-22T11:30:00+00:00", "end_at": "2026-09-22T12:00:00+00:00", "timezone": "UTC", "calendar_id": "calendar-1"},
        {"id": "overlap", "title": "Shift overlap", "start_at": "2026-09-22T11:00:00+00:00", "end_at": "2026-09-22T12:00:00+00:00", "timezone": "UTC", "calendar_id": "calendar-1"},
        {"id": "after-1", "title": "Production review", "start_at": "2026-09-22T12:00:00+00:00", "end_at": "2026-09-22T13:00:00+00:00", "timezone": "UTC", "calendar_id": "calendar-1"},
        {"id": "after-2", "title": "Customer call", "start_at": "2026-09-22T14:00:00+00:00", "end_at": "2026-09-22T15:30:00+00:00", "timezone": "UTC", "calendar_id": "calendar-2"},
    )


def booking_fixture() -> tuple[MeetingBooking, ...]:
    return (
        MeetingBooking("booking-1", "event-1", "Supplier review", "2026-09-22T12:00:00+00:00", "2026-09-22T13:00:00+00:00", attendee_count=2),
        MeetingBooking("booking-missing-event", "missing-event", "", "bad", "2026-09-22T13:00:00+00:00", attendee_count=4, metadata={"historical_title": "Old supplier meeting"}),
    )


def booking_consistency_errors(
    bookings: Iterable[MeetingBooking],
    events: Iterable[dict[str, Any]],
    attendees: Iterable[EventAttendee],
) -> tuple[str, ...]:
    booking_list = list(bookings)
    event_list = list(events)
    attendee_list = list(attendees)
    event_by_id = {event.get("id"): event for event in event_list}
    booking_by_event = {booking.event_id: booking for booking in booking_list if booking.event_id is not None}
    errors: list[str] = []
    for booking in booking_list:
        if not booking.title:
            errors.append(f"booking {booking.id} has stale or empty title")
        booking_start = _parse_aware(booking.start_at)
        booking_end = _parse_aware(booking.end_at)
        if booking_start is None or booking_end is None or booking_end <= booking_start:
            errors.append(f"booking {booking.id} has malformed timing")
        event = event_by_id.get(booking.event_id)
        if event is None:
            errors.append(f"booking {booking.id} points to missing event")
            continue
        if booking.title != event.get("title"):
            errors.append(f"booking {booking.id} has stale or empty title")
        if booking.start_at != event.get("start_at") or booking.end_at != event.get("end_at"):
            errors.append(f"booking {booking.id} has stale event timing")
        count = sum(1 for attendee in attendee_list if attendee.event_id == booking.event_id)
        if booking.attendee_count is not None and booking.attendee_count != count:
            errors.append(f"booking {booking.id} attendee count disagrees with rows")
    for event in event_list:
        if event.get("id") not in booking_by_event:
            errors.append(f"event {event.get('id')} points to missing booking")
    return tuple(errors)


def workflow_actions() -> tuple[SchedulingWorkflowAction, ...]:
    return (
        SchedulingWorkflowAction("send_email", RiskLevel.POSTING),
        SchedulingWorkflowAction("create_activity", RiskLevel.WRITE),
        SchedulingWorkflowAction("booking", RiskLevel.WRITE),
        SchedulingWorkflowAction("reschedule", RiskLevel.WRITE),
        SchedulingWorkflowAction("cancel", RiskLevel.DESTRUCTIVE),
    )


def planning_side_effect_errors(invoked_actions: Iterable[str]) -> tuple[str, ...]:
    forbidden = {action.action for action in workflow_actions()}
    return tuple(f"planning invoked side effect: {action}" for action in invoked_actions if action in forbidden)


def assert_planning_side_effect_free(invoked_actions: Iterable[str]) -> None:
    errors = planning_side_effect_errors(invoked_actions)
    if errors:
        raise AssertionError("; ".join(errors))


def phase1_tool_policy() -> AgentToolPolicy:
    return AgentToolPolicy(
        read_only=True,
        risk_mode=RiskLevel.READ,
        allowed_domains=("calendar", "party"),
        denied_domains=("host",),
        allowed_entities=("Party", "Calendar", "CalendarEvent", "FreeBusy"),
        denied_entities=("HostPool",),
        allowed_actions=("Party.list", "Calendar.list", "CalendarEvent.list", "endpoint.calendar.free_busy"),
        denied_actions=("send_email", "create_activity", "booking", "cancel", "reschedule"),
        max_records_per_query=100,
    )


def authority_failure_fixtures() -> tuple[AuthorityFailure, ...]:
    return (
        AuthorityFailure(AuthorityFailureKind.TOOL_MISSING, "The requested tool is not registered.", "Unknown.list"),
        AuthorityFailure(AuthorityFailureKind.AGENT_UNAUTHORIZED, "The agent policy denies this resource.", "HostPool.list"),
        AuthorityFailure(AuthorityFailureKind.PROVIDER_FAILURE, "Provider failed before model completion.", provider_status=401),
        AuthorityFailure(AuthorityFailureKind.TOOL_PERMISSION_DENIED, "The tool executed and returned permission denied.", "CalendarEvent.update"),
    )


def provider_401_fixture() -> ProviderFailure:
    return ProviderFailure(status_code=401, error="UNAUTHORIZED")


def subscription_fixtures() -> tuple[CalendarSubscription, ...]:
    return (
        CalendarSubscription("sub-ok", "calendar-1", "ok", "2026-09-22T08:00:00Z"),
        CalendarSubscription("sub-pending", "calendar-2", "pending", "2026-09-22T08:00:00Z"),
        CalendarSubscription("sub-error", "calendar-3", "error", "2026-09-21T08:00:00Z", "malformed feed"),
        CalendarSubscription("sub-stale", "calendar-4", "ok", "2026-09-18T08:00:00Z"),
    )


def subscription_errors(subscription: CalendarSubscription, *, now: datetime, max_age: timedelta) -> tuple[str, ...]:
    errors: list[str] = []
    if subscription.status != "ok":
        errors.append(f"subscription status is {subscription.status}")
    if subscription.feed_error:
        errors.append(f"subscription feed error: {subscription.feed_error}")
    synced = _parse_aware(subscription.last_sync_at)
    if synced is None:
        errors.append("subscription last-sync timestamp is malformed")
    elif now - synced > max_age:
        errors.append("subscription data is stale")
    return tuple(errors)


def execution_evidence_from_trace(
    trace: Sequence[ToolCallRecord],
    *,
    job_id: str = "job-phase1",
    evaluation_decision: str | None = None,
    approval_state: str | None = None,
) -> AgentJob:
    steps = tuple(
        AgentJobStep(
            step_id=f"step-{index + 1}",
            action=call.tool,
            tool_input=dict(call.arguments),
            status="succeeded" if call.succeeded else "failed",
            error=call.error,
            authority_failure=(
                AuthorityFailure(AuthorityFailureKind.TOOL_PERMISSION_DENIED, call.error, call.tool)
                if call.error == "permission_denied"
                else None
            ),
        )
        for index, call in enumerate(trace)
    )
    messages = tuple(
        AgentMessage(f"message-{index + 1}", "tool", step.action, step.step_id)
        for index, step in enumerate(steps)
    )
    return AgentJob(
        job_id=job_id,
        status="completed" if all(step.status == "succeeded" for step in steps) else "failed",
        steps=steps,
        messages=messages,
        evaluation_decision=evaluation_decision,
        approval_state=approval_state,
    )


def execution_evidence_errors(job: AgentJob, trace: Sequence[ToolCallRecord]) -> tuple[str, ...]:
    errors: list[str] = []
    valid_job_statuses = ("completed", "failed")
    valid_step_statuses = ("succeeded", "failed")
    if not isinstance(job.status, str) or job.status not in valid_job_statuses:
        errors.append(f"job has unsupported aggregate status: {job.status!r}")
    step_statuses_valid = True
    for step in job.steps:
        if not isinstance(step.status, str) or step.status not in valid_step_statuses:
            errors.append(f"step {step.step_id} has unsupported status: {step.status!r}")
            step_statuses_valid = False
    if job.status in valid_job_statuses and step_statuses_valid:
        expected_job_status = "completed" if all(
            step.status == "succeeded" for step in job.steps
        ) else "failed"
        if job.status != expected_job_status:
            errors.append(
                f"job status {job.status!r} contradicts its step statuses; expected {expected_job_status!r}"
            )
    if len(job.steps) != len(trace):
        errors.append("every tool invocation must have one execution step")
    if len({step.step_id for step in job.steps}) != len(job.steps):
        errors.append("execution step IDs must be unique")
    for step, call in zip(job.steps, trace):
        if step.action != call.tool or step.tool_input != call.arguments:
            errors.append(f"step {step.step_id} does not match its invocation")
        expected_status = "succeeded" if call.succeeded else "failed"
        if step.status != expected_status:
            errors.append(f"step {step.step_id} status does not match its invocation")
        if not call.succeeded and not step.error:
            errors.append(f"step {step.step_id} dropped the tool error")
    return tuple(errors)


def replay_fingerprint(job: AgentJob) -> str:
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "steps": [
            {"id": step.step_id, "action": step.action, "input": step.tool_input, "status": step.status, "error": step.error}
            for step in job.steps
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def stale_subscription_reference(now: datetime | None = None) -> tuple[str, ...]:
    reference = now or datetime(2026, 9, 22, 8, tzinfo=timezone.utc)
    return subscription_errors(subscription_fixtures()[-1], now=reference, max_age=timedelta(hours=24))
