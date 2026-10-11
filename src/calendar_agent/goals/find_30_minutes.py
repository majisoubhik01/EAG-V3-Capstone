"""Planning-only handler for ``calendar.find_30_minutes``."""

from __future__ import annotations

from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import Any

from ..models import AgentResult, AgentTask, CandidateSlot, ResultStatus
from ..tools import CalendarTool


GOAL = "calendar.find_30_minutes"

_ROLE_TITLES = ("plant manager", "plant head")
_PARTY_PAGE_SIZE = 100
_MAX_PARTY_PAGES = 100
_LEGACY_PARTY_LIMIT = 1000


class _PartyResolutionError(ValueError):
    pass


def _clarification(task: AgentTask, summary: str, *ambiguities: str) -> AgentResult:
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary=summary,
        ambiguities=list(ambiguities),
    )


def _resolve_plant_head(tools: CalendarTool) -> list[dict[str, Any]]:
    """Resolve from complete pages; legacy lists cannot signal hidden truncation.

    The legacy list path preserves its existing behavior below the requested
    limit, but rejects a saturated response because it may be incomplete.
    """

    candidates: dict[str, dict[str, Any]] = {}
    resolution_error: str | None = None
    for title in _ROLE_TITLES:
        page_reader = getattr(tools, "find_parties_page", None)
        if callable(page_reader):
            page_number = 1
            paginated_records: list[dict[str, Any]] = []
            total_count: int | None = None
            for page_number in range(1, _MAX_PARTY_PAGES + 1):
                try:
                    page = page_reader(
                        page=page_number,
                        page_size=_PARTY_PAGE_SIZE,
                        job_title=title.title(),
                    )
                    records = getattr(page, "records", None)
                    reported_page = getattr(page, "page", None)
                    reported_page_size = getattr(page, "page_size", None)
                    reported_total = getattr(page, "total_count", None)
                    has_more = getattr(page, "has_more", None)
                except Exception as exc:
                    raise _PartyResolutionError(
                        f"Party page {page_number} could not be retrieved or parsed ({type(exc).__name__})."
                    ) from None

                if (
                    not isinstance(records, (list, tuple))
                    or type(reported_page) is not int
                    or reported_page != page_number
                    or type(reported_page_size) is not int
                    or reported_page_size != _PARTY_PAGE_SIZE
                    or type(reported_total) is not int
                    or reported_total < 0
                    or type(has_more) is not bool
                    or len(records) > _PARTY_PAGE_SIZE
                ):
                    raise _PartyResolutionError(
                        f"Party page {page_number} did not match the pagination contract."
                    )
                if total_count is None:
                    total_count = reported_total
                elif reported_total != total_count:
                    raise _PartyResolutionError("Party pagination changed its total record count.")

                expected_has_more = page_number * _PARTY_PAGE_SIZE < reported_total
                if has_more is not expected_has_more:
                    raise _PartyResolutionError(
                        f"Party page {page_number} reported inconsistent completion metadata."
                    )
                page_records: list[dict[str, Any]] = []
                for record in records:
                    if isinstance(record, dict):
                        party = record
                    else:
                        try:
                            party = {
                                "id": record.id,
                                "email": record.email,
                                "job_title": record.job_title,
                            }
                            if hasattr(record, "name"):
                                party["name"] = record.name
                        except Exception as exc:
                            raise _PartyResolutionError(
                                f"Party page {page_number} contained an unreadable record ({type(exc).__name__})."
                            ) from None
                    page_records.append(party)
                paginated_records.extend(page_records)

                if not has_more:
                    if len(paginated_records) != reported_total:
                        raise _PartyResolutionError(
                            "Party pagination ended before its reported total was retrieved."
                        )
                    break
                if page_number == _MAX_PARTY_PAGES:
                    raise _PartyResolutionError(
                        "Party pagination reached its page limit while more records remained."
                    )
            parties = paginated_records
        else:
            parties = tools.find_parties(
                limit=_LEGACY_PARTY_LIMIT,
                job_title=title.title(),
            )
            if not isinstance(parties, list):
                raise _PartyResolutionError("The legacy Party result was not a list.")
            if len(parties) >= _LEGACY_PARTY_LIMIT:
                raise _PartyResolutionError(
                    "The legacy Party result reached its limit and may be incomplete."
                )

        for party in parties:
            if not isinstance(party, dict):
                resolution_error = resolution_error or "Party results contained an unreadable record."
                continue
            job_title = party.get("job_title")
            if not isinstance(job_title, str) or not job_title.strip():
                resolution_error = resolution_error or "A Party record is missing its job title."
                continue
            normalized_title = job_title.strip().lower()
            if normalized_title not in _ROLE_TITLES:
                continue
            party_id = party.get("id")
            email = party.get("email")
            if (
                not isinstance(party_id, str)
                or not party_id.strip()
                or not isinstance(email, str)
                or not email.strip()
            ):
                resolution_error = resolution_error or "A plant-head Party record has incomplete identity fields."
                continue
            previous = candidates.get(party_id)
            if previous is not None:
                if any(
                    isinstance(previous.get(field), str)
                    and isinstance(party.get(field), str)
                    and previous[field].strip().casefold() != party[field].strip().casefold()
                    for field in ("email", "job_title", "name")
                ):
                    resolution_error = resolution_error or (
                        f"Party ID {party_id!r} has conflicting identity records."
                    )
            else:
                candidates[party_id] = party
    if resolution_error is not None:
        raise _PartyResolutionError(resolution_error)
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

    try:
        parties = _resolve_plant_head(tools)
    except _PartyResolutionError as exc:
        return _clarification(
            task,
            "The plant head cannot be deterministically identified from complete Party data.",
            str(exc),
        )

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
    link_duration = task.context.get("scheduling_link_duration_minutes")
    if link_duration is not None:
        try:
            link_duration_value = float(link_duration)
        except (TypeError, ValueError):
            return _clarification(task, "The scheduling-link duration is invalid.", "A numeric scheduling-link duration is required.")
        if link_duration_value != 30:
            return _clarification(
                task,
                "The selected scheduling link does not satisfy the requested 30-minute duration.",
                f"The scheduling link requires {link_duration_value:g} minutes; the goal requires 30 minutes.",
            )
    min_notice = preference.get("min_notice_hours")
    if isinstance(min_notice, bool):
        return _clarification(
            task,
            "Scheduling preferences are invalid.",
            "Minimum notice must be a non-negative finite number of hours.",
        )
    try:
        min_notice_hours = float(min_notice)
    except (TypeError, ValueError, OverflowError):
        return _clarification(
            task,
            "Scheduling preferences are invalid.",
            "Minimum notice must be a non-negative finite number of hours.",
        )
    if min_notice_hours < 0:
        return _clarification(
            task,
            "Scheduling preferences are invalid.",
            "Minimum notice must be a non-negative finite number of hours.",
        )
    try:
        min_notice_delta = timedelta(hours=min_notice_hours)
    except (ValueError, OverflowError):
        return _clarification(
            task,
            "Scheduling preferences are invalid.",
            "Minimum notice must be a non-negative finite number of hours.",
        )

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