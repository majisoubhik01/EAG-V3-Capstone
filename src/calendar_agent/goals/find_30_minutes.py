"""Planning-only handler for ``calendar.find_30_minutes``."""

from __future__ import annotations

from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Any

from ..models import AgentResult, AgentTask, CandidateSlot, ResultStatus
from ..tools import CalendarTool


GOAL = "calendar.find_30_minutes"

_ROLE_TITLES = ("plant manager", "plant head")


def _clarification(task: AgentTask, summary: str, *ambiguities: str) -> AgentResult:
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary=summary,
        ambiguities=list(ambiguities),
    )


def _resolve_plant_head(tools: CalendarTool) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for title in _ROLE_TITLES:
        for party in tools.find_parties(limit=1000, job_title=title.title()):
            party_id = party.get("id")
            job_title = str(party.get("job_title") or "").strip().lower()
            email = party.get("email")
            if (
                isinstance(party_id, str)
                and party_id
                and isinstance(email, str)
                and email
                and job_title in _ROLE_TITLES
            ):
                candidates[party_id] = party
    return list(candidates.values())


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None
    return None


def _matching_rules(
    rules: list[dict[str, Any]], party_id: str, calendar_ids: set[str]
) -> list[dict[str, Any]]:
    matches = []
    for rule in rules:
        rule_party = rule.get("party_id") or rule.get("owner_party_id")
        rule_calendar = rule.get("calendar_id")
        if rule_party == party_id or rule_calendar in calendar_ids:
            matches.append(rule)
    return matches


def _rule_allows(rule: dict[str, Any], start: datetime, end: datetime) -> bool:
    timezone_name = rule.get("timezone")
    weekly_hours = rule.get("weekly_hours")
    if not isinstance(timezone_name, str) or not isinstance(weekly_hours, list):
        return False
    try:
        local_start = start.astimezone(ZoneInfo(timezone_name))
        local_end = end.astimezone(ZoneInfo(timezone_name))
    except Exception:
        return False
    if local_start.date() != local_end.date():
        return False

    for override in rule.get("date_overrides") or []:
        if not isinstance(override, dict) or override.get("date") != local_start.date().isoformat():
            continue
        if override.get("type") in {"unavailable", "blocked"}:
            return False
        if override.get("type") == "available":
            return True

    day = local_start.strftime("%A").lower()
    for period in weekly_hours:
        if not isinstance(period, dict) or period.get("day", "").lower() != day:
            continue
        if period.get("enabled") is not True:
            continue
        try:
            period_start = time.fromisoformat(str(period["start"]))
            period_end = time.fromisoformat(str(period["end"]))
        except (KeyError, ValueError):
            continue
        if period_start <= local_start.timetz().replace(tzinfo=None) and period_end >= local_end.timetz().replace(tzinfo=None):
            return True
    return False


def _busy_intervals(subject: dict[str, Any]) -> list[tuple[datetime, datetime]] | None:
    busy = subject.get("busy")
    if not isinstance(busy, list):
        return None
    intervals = []
    for item in busy:
        if not isinstance(item, dict):
            return None
        start = _parse_datetime(item.get("start") or item.get("start_at"))
        end = _parse_datetime(item.get("end") or item.get("end_at"))
        if start is None or end is None or end <= start:
            return None
        intervals.append((start, end))
    return intervals


def plan(task: AgentTask, tools: CalendarTool) -> AgentResult:
    """Resolve the participant and return only a state-supported slot."""

    parties = _resolve_plant_head(tools)

    if len(parties) != 1:
        return _clarification(
            task,
            "The plant head cannot be deterministically identified from Party data.",
            f"Expected exactly one valid plant-head Party; found {len(parties)}.",
            "The resolver accepts only an explicit Plant Manager or plant head job title and a valid email.",
        )

    participant = parties[0]
    email = participant.get("email")
    start = task.context.get("start")
    end = task.context.get("end")
    if not isinstance(email, str) or not isinstance(start, datetime) or not isinstance(end, datetime):
        return _clarification(task, "A bounded timezone-aware search window is required.", "The task did not provide valid start and end datetimes.")
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        return _clarification(task, "The search window is invalid.", "Start and end must be timezone-aware and end must be after start.")

    calendars = tools.list_calendars(owner_party_id=participant["id"], limit=1000)
    calendars = [c for c in calendars if c.get("id") and c.get("owner_party_id") == participant["id"]]
    if not calendars:
        return _clarification(task, "No deterministically owned calendar was found for the plant head.", "A calendar owned by the resolved Party is required.")
    if len(calendars) > 1:
        return _clarification(task, "Multiple deterministically owned calendars were found for the plant head.", "A single owned calendar is required; the handler will not choose one arbitrarily.")
    calendar_id = str(calendars[0]["id"])
    calendar_ids = {calendar_id}

    preferences = tools.list_scheduling_preferences(limit=1000)
    if len(preferences) != 1:
        return _clarification(task, "Scheduling preferences are not uniquely available.", f"Expected one preferences record; found {len(preferences)}.")
    preference = preferences[0]
    try:
        default_duration = float(preference.get("default_meeting_duration"))
    except (TypeError, ValueError):
        return _clarification(task, "Scheduling preferences are invalid.", "The default meeting duration is missing or not numeric.")
    if default_duration != 30:
        return _clarification(task, "The configured default meeting duration is not the requested 30 minutes.", f"Observed default duration: {default_duration} minutes.")
    min_notice = preference.get("min_notice_hours")
    try:
        min_notice_delta = timedelta(hours=float(min_notice))
    except (TypeError, ValueError):
        return _clarification(task, "Scheduling preferences are invalid.", "Minimum notice is missing or not numeric.")

    now = task.context.get("now")
    if not isinstance(now, datetime) or now.tzinfo is None:
        return _clarification(task, "A reference time is required to apply minimum notice.", "The task did not provide a timezone-aware now value.")
    earliest = max(start, now + min_notice_delta)
    rules = _matching_rules(tools.list_availability_rules(limit=1000), participant["id"], calendar_ids)
    valid_rules = [r for r in rules if isinstance(r.get("weekly_hours"), list) and isinstance(r.get("timezone"), str)]
    if not valid_rules:
        return _clarification(task, "No valid availability rule is linked to the resolved plant head.", "An explicitly party- or calendar-linked availability rule is required.")

    free_busy = tools.free_busy(start=start, end=end, emails=[email], calendar_ids=list(calendar_ids))
    result = free_busy.get("result", {}) if isinstance(free_busy, dict) else {}
    subjects = result.get("subjects", []) if isinstance(result, dict) else []
    subject = next((s for s in subjects if isinstance(s, dict) and s.get("subject") in {email, *calendar_ids}), {}) if isinstance(subjects, list) else {}
    subject_calendars = subject.get("calendars", []) if isinstance(subject, dict) else []
    access = subject_calendars[0].get("access_level") if subject_calendars and isinstance(subject_calendars[0], dict) else None
    visibility = subject.get("visibility") if isinstance(subject, dict) else None
    reasons = {str(subject.get("visibility_code", ""))} if isinstance(subject, dict) else set()
    reasons.update(str(c.get("reason_code", "")) for c in subject_calendars if isinstance(c, dict))
    if access in (None, "none") or visibility not in {"free_busy", "read", "write"} or reasons & {"freebusy.notShared", "freebusy.noCalendar"}:
        return _clarification(task, "The participant's calendar is not shared enough to confirm availability.", "An empty or hidden free/busy result is not confirmed free time.")
    intervals = _busy_intervals(subject)
    if intervals is None:
        return _clarification(task, "The participant's free/busy response is missing or invalid.", "Availability cannot be confirmed from malformed busy intervals.")

    cursor = earliest.replace(second=0, microsecond=0)
    if earliest.second or earliest.microsecond:
        cursor += timedelta(minutes=1)
    remainder = cursor.minute % 30
    if remainder:
        cursor += timedelta(minutes=30 - remainder)
    duration = timedelta(minutes=30)
    while cursor + duration <= end:
        slot_end = cursor + duration
        busy_overlap = any(cursor < busy_end and slot_end > busy_start for busy_start, busy_end in intervals)
        rule_match = next((rule for rule in valid_rules if _rule_allows(rule, cursor, slot_end)), None)
        if not busy_overlap and rule_match is not None:
            timezone_name = str(rule_match["timezone"])
            slot = CandidateSlot(
                start_at=cursor.isoformat(),
                end_at=slot_end.isoformat(),
                timezone=timezone_name,
                availability_confirmed=True,
                reason="visible free/busy, minimum notice, and linked availability rule all support this 30-minute slot",
                party_id=str(participant["id"]),
                calendar_id=calendar_id,
            )
            return AgentResult(
                goal=task.goal,
                status=ResultStatus.PLANNED,
                summary="A verified 30-minute slot was found; no event was created.",
                candidate_slots=[slot],
                claimed_outcome={"slot": slot},
            )
        cursor += duration

    return _clarification(task, "No 30-minute slot satisfies the visible free/busy and availability constraints.", "All candidate intervals were busy or outside linked availability rules.")