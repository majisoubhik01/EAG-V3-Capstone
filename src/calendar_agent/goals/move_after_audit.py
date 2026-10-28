"""Planning-only handler for ``calendar.move_after_audit``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any
from zoneinfo import ZoneInfo

from ..models import (
    AgentResult,
    AgentTask,
    AuditReference,
    CalendarMutationError,
    CalendarEventPage,
    CalendarEventRef,
    EventMove,
    MutationDisabledError,
    ResultStatus,
)
from ..tools import CalendarEventListError, CalendarTool
from ..confirmation import confirms_shift


GOAL = "calendar.move_after_audit"
_EVENT_LIST_LIMIT = 1000
_MAX_EVENT_LIST_PAGES = 100
_DEFAULT_PROPOSED_SHIFT = timedelta(minutes=30)
_SHIFT_SEARCH_INCREMENT = timedelta(minutes=30)
_MAX_ALTERNATIVE_SHIFT = timedelta(hours=8)


def _event_ref(
    event: dict[str, object],
    fallback_timezone: ZoneInfo | None = None,
) -> CalendarEventRef:
    event_timezone = _event_timezone(event, fallback_timezone)
    start_value = event.get("start_at")
    end_value = event.get("end_at")
    start_at = str(start_value or "")
    end_at = str(end_value or "")
    if event_timezone is not None:
        if isinstance(start_value, str):
            start_at = _normalized_timestamp(start_value, event_timezone)
        if isinstance(end_value, str):
            end_at = _normalized_timestamp(end_value, event_timezone)
    return CalendarEventRef(
        id=str(event.get("number") or event.get("id", "")),
        title=str(event.get("title", "")),
        start_at=start_at,
        end_at=end_at,
        calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
        status=event.get("status") if isinstance(event.get("status"), str) else None,
        timezone=event_timezone.key if event_timezone is not None else None,
    )


def _is_background_event(event: dict[str, Any]) -> bool:
    """All-day and subscription-feed events (e.g. holidays) are not reschedulable meetings."""

    return event.get("all_day") is True or bool(event.get("subscription_id"))


def _event_id(event: dict[str, Any]) -> str | None:
    value = event.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_audit_candidate(event: dict[str, Any]) -> bool:
    text = " ".join(str(event.get(key) or "") for key in ("title", "description"))
    return "audit" in text.lower()


def _parse_timezone(timezone_name: object) -> ZoneInfo | None:
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        return None
    try:
        return ZoneInfo(timezone_name.strip())
    except (KeyError, OSError, ValueError):
        return None


def _event_timezone(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> ZoneInfo | None:
    timezone_name = event.get("timezone")
    if timezone_name is None:
        return fallback_timezone
    return _parse_timezone(timezone_name)


def _localize_wall_time(value: datetime, event_timezone: ZoneInfo) -> datetime | None:
    instants: dict[datetime, datetime] = {}
    for fold in (0, 1):
        candidate = value.replace(tzinfo=event_timezone, fold=fold)
        instant = candidate.astimezone(timezone.utc)
        round_trip = instant.astimezone(event_timezone)
        if round_trip.replace(tzinfo=None) == value:
            instants[instant] = candidate
    if len(instants) != 1:
        return None
    return next(iter(instants.values()))


def _parse_event_datetime(value: object, event_timezone: ZoneInfo | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed if parsed.utcoffset() is not None else None
    if event_timezone is None:
        return None
    return _localize_wall_time(parsed, event_timezone)


def _normalized_timestamp(value: str, event_timezone: ZoneInfo) -> str:
    parsed = _parse_event_datetime(value, event_timezone)
    if parsed is None:
        return value
    try:
        original = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if original.tzinfo is not None and original.utcoffset() is not None:
        return value
    return parsed.isoformat()


def _parse_event_interval(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> tuple[datetime, datetime] | None:
    start_value = event.get("start_at")
    end_value = event.get("end_at")
    event_timezone = _event_timezone(event, fallback_timezone)
    if event_timezone is None:
        return None
    start = _parse_event_datetime(start_value, event_timezone)
    end = _parse_event_datetime(end_value, event_timezone)
    if start is None or end is None:
        return None
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if end_utc <= start_utc:
        return None
    return start_utc, end_utc


def _parse_event_times(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> tuple[datetime, datetime] | None:
    return _parse_event_interval(event, fallback_timezone)


def _resolve_calendar_timezone(tools: CalendarTool, calendar_id: str) -> ZoneInfo:
    calendar = tools.get_calendar(calendar_id)
    if not isinstance(calendar, dict) or calendar.get("id") != calendar_id:
        raise CalendarEventListError(
            "The owning calendar could not be resolved uniquely for timezone fallback."
        )
    event_timezone = _parse_timezone(calendar.get("timezone"))
    if event_timezone is None:
        raise CalendarEventListError(
            "The owning calendar has no valid IANA timezone for timestamp fallback."
        )
    return event_timezone


def _explicit_timezone_fallback(task: AgentTask) -> ZoneInfo | None:
    timezone_name = task.context.get("naive_timestamp_timezone")
    use_calendar_timezone = task.context.get("use_calendar_timezone_fallback", False)
    if type(use_calendar_timezone) is not bool:
        raise ValueError("use_calendar_timezone_fallback must be a boolean.")
    if timezone_name is not None and use_calendar_timezone:
        raise ValueError(
            "Configure either a fixed timestamp fallback or a calendar timezone fallback, not both."
        )
    if timezone_name is None:
        return None
    event_timezone = _parse_timezone(timezone_name)
    if event_timezone is None:
        raise ValueError("naive_timestamp_timezone must be a valid IANA timezone.")
    return event_timezone


def _parse_event_start(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> datetime | None:
    event_timezone = _event_timezone(event, fallback_timezone)
    start = _parse_event_datetime(event.get("start_at"), event_timezone)
    return start.astimezone(timezone.utc) if start is not None else None


def _intervals_overlap(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> bool:
    return first_start < second_end and first_end > second_start


def _clarification(
    task: AgentTask,
    audit_reference: AuditReference,
    summary: str,
    *ambiguities: str,
) -> AgentResult:
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary=summary,
        ambiguities=list(ambiguities),
        audit_reference=audit_reference,
    )


def _list_complete_events(
    tools: CalendarTool,
    *,
    calendar_id: str | None = None,
    search: str | None = None,
    event_number: str | None = None,
) -> list[dict[str, Any]]:
    if calendar_id is None and search is None and event_number is None:
        raise CalendarEventListError("CalendarEvent listing requires an explicit filter.")
    page_reader = getattr(tools, "list_events_page", None)
    if not callable(page_reader):
        raise CalendarEventListError(
            "The calendar adapter does not expose pagination metadata."
        )

    records: list[dict[str, Any]] = []
    expected_total: int | None = None
    seen_page_signatures: set[str] = set()
    seen_event_ids: set[str] = set()
    offset = 0

    for _ in range(_MAX_EVENT_LIST_PAGES):
        arguments: dict[str, Any] = {
            "limit": _EVENT_LIST_LIMIT,
            "offset": offset,
        }
        if calendar_id is not None:
            arguments["calendar_id"] = calendar_id
        if search is not None:
            arguments["search"] = search
        if event_number is not None:
            arguments["number"] = event_number

        page = page_reader(**arguments)
        if not isinstance(page, CalendarEventPage):
            raise CalendarEventListError(
                "The calendar adapter returned no typed event-page metadata."
            )
        try:
            page.validate(
                requested_limit=_EVENT_LIST_LIMIT,
                requested_offset=offset,
            )
        except (TypeError, ValueError) as exc:
            raise CalendarEventListError(
                f"CalendarEvent page metadata is invalid: {exc}"
            ) from exc

        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise CalendarEventListError(
                "CalendarEvent total changed during pagination."
            )
        if page.total >= _EVENT_LIST_LIMIT:
            raise CalendarEventListError(
                f"CalendarEvent listing reached the {_EVENT_LIST_LIMIT}-record safety limit."
            )

        page_event_ids: set[str] = set()
        for event in page.records:
            if calendar_id is not None and event.get("calendar_id") != calendar_id:
                raise CalendarEventListError(
                    "CalendarEvent page returned a record outside the requested calendar."
                )
            if event_number is not None and event.get("number") != event_number:
                raise CalendarEventListError(
                    "CalendarEvent number filter returned a different event number."
                )
            event_id = _event_id(event)
            if (
                event_id is not None
                and event_id in seen_event_ids
                and event_id not in page_event_ids
            ):
                raise CalendarEventListError(
                    "CalendarEvent pagination returned a repeated event ID."
                )
            if event_id is not None:
                page_event_ids.add(event_id)
        seen_event_ids.update(page_event_ids)

        try:
            page_signature = json.dumps(
                page.records,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except (TypeError, ValueError) as exc:
            raise CalendarEventListError(
                "CalendarEvent page records are not valid JSON objects."
            ) from exc
        if page_signature in seen_page_signatures:
            raise CalendarEventListError(
                "CalendarEvent pagination returned a repeated page."
            )
        seen_page_signatures.add(page_signature)

        records.extend(page.records)
        if not page.has_more:
            if len(records) != page.total:
                raise CalendarEventListError(
                    "CalendarEvent pagination ended before the reported total."
                )
            return records

        next_offset = page.next_offset
        if next_offset <= offset:
            raise CalendarEventListError(
                "CalendarEvent pagination did not advance."
            )
        offset = next_offset

    raise CalendarEventListError(
        f"CalendarEvent pagination reached the {_MAX_EVENT_LIST_PAGES}-page safety limit."
    )


def _resolve_exact_audit(
    task: AgentTask,
    tools: CalendarTool,
    *,
    audit_number: str,
    audit_title: str,
) -> tuple[dict[str, Any], str, AuditReference] | AgentResult:
    reference = AuditReference(candidates=(), search_terms=(audit_number,))
    try:
        records = _list_complete_events(tools, event_number=audit_number)
    except CalendarEventListError as exc:
        return _clarification(
            task,
            reference,
            "The specified audit could not be read completely.",
            str(exc),
        )

    candidates = [event for event in records if event.get("number") == audit_number]
    reference = AuditReference(
        candidates=tuple(_event_ref(event) for event in candidates),
        search_terms=(audit_number,),
    )
    if len(candidates) != 1:
        return _clarification(
            task,
            reference,
            "The specified audit could not be resolved uniquely.",
            f"Expected one event numbered {audit_number!r}; found {len(candidates)}.",
        )

    audit_event = candidates[0]
    if audit_event.get("title") != audit_title:
        return _clarification(
            task,
            reference,
            "The event number resolved to a different audit title.",
            "The audit title must match the explicit task reference exactly.",
        )
    status = audit_event.get("status")
    if not isinstance(status, str) or status.strip().casefold() != "confirmed":
        return _clarification(
            task,
            reference,
            "The specified audit is not verified as confirmed.",
            "A confirmed audit status is required before preparing a move plan.",
        )
    calendar_id = audit_event.get("calendar_id")
    if not isinstance(calendar_id, str) or not calendar_id.strip():
        return _clarification(
            task,
            reference,
            "The specified audit has no verified calendar association.",
            "The event's calendar_id is required; no calendar will be guessed.",
        )

    requested_calendar_id = task.context.get("calendar_id")
    if (
        requested_calendar_id is not None
        and requested_calendar_id != calendar_id
    ):
        return _clarification(
            task,
            reference,
            "The supplied calendar does not match the audit's associated calendar.",
            "The calendar must be taken from the exact audit record.",
        )

    return audit_event, calendar_id.strip(), reference


def plan(task: AgentTask, tools: CalendarTool) -> AgentResult:
    """Build a deterministic, read-only plan for events after one audit."""

    fallback_reference = AuditReference(candidates=(), search_terms=("timezone",))
    try:
        fallback_timezone = _explicit_timezone_fallback(task)
    except ValueError as exc:
        return _clarification(
            task,
            fallback_reference,
            "The configured timestamp timezone fallback is invalid.",
            str(exc),
        )
    use_calendar_timezone = task.context.get("use_calendar_timezone_fallback", False)

    audit_number = task.context.get("audit_number")
    if audit_number is not None:
        raw_title = task.context.get("audit_title")
        reference = AuditReference(
            candidates=(),
            search_terms=(audit_number,) if isinstance(audit_number, str) else (),
        )
        if (
            not isinstance(audit_number, str)
            or not audit_number.strip()
            or not isinstance(raw_title, str)
            or not raw_title.strip()
        ):
            return _clarification(
                task,
                reference,
                "The exact audit reference is incomplete.",
                "A nonblank audit_number and its expected audit_title are required.",
            )
        exact_resolution = _resolve_exact_audit(
            task,
            tools,
            audit_number=audit_number.strip(),
            audit_title=raw_title,
        )
        if isinstance(exact_resolution, AgentResult):
            return exact_resolution
        audit_event, calendar_id, reference = exact_resolution
        if (
            fallback_timezone is None
            and use_calendar_timezone
            and audit_event.get("timezone") is None
        ):
            try:
                fallback_timezone = _resolve_calendar_timezone(tools, calendar_id)
            except CalendarEventListError as exc:
                return _clarification(
                    task,
                    reference,
                    "The owning calendar timezone could not be verified for timestamp fallback.",
                    str(exc),
                )
        audit_times = _parse_event_times(audit_event, fallback_timezone)
        if audit_times is None:
            reference = AuditReference(
                candidates=(_event_ref(audit_event, fallback_timezone),),
                search_terms=(audit_number.strip(),),
            )
            return _clarification(
                task,
                reference,
                "The specified audit interval or timezone cannot be verified.",
                "Timestamps must be valid, unambiguous, and resolvable with the event timezone or an explicitly configured fallback.",
            )
        reference = AuditReference(
            candidates=(_event_ref(audit_event, fallback_timezone),),
            search_terms=(audit_number.strip(),),
        )
    else:
        raw_audit_title = task.context.get("audit_title")
        if raw_audit_title is not None and (
            not isinstance(raw_audit_title, str) or not raw_audit_title.strip()
        ):
            return _clarification(
                task,
                AuditReference(candidates=(), search_terms=()),
                "The expected audit title is invalid.",
                "Provide the exact non-empty audit title to disambiguate audit candidates.",
            )
        expected_audit_title = (
            raw_audit_title.strip()
            if isinstance(raw_audit_title, str)
            else None
        )
        search_term = expected_audit_title or "audit"
        reference = AuditReference(candidates=(), search_terms=(search_term,))
        raw_calendar_id = task.context.get("calendar_id")
        if not isinstance(raw_calendar_id, str) or not raw_calendar_id.strip():
            return _clarification(
                task,
                reference,
                "A calendar scope is required.",
                "Provide the calendar_id whose events should be searched and planned.",
            )
        calendar_id = raw_calendar_id.strip()

        try:
            audit_records = _list_complete_events(
                tools,
                calendar_id=calendar_id,
                search=search_term,
            )
        except CalendarEventListError as exc:
            return _clarification(
                task,
                reference,
                "Audit resolution is blocked because the search may be incomplete.",
                str(exc),
            )

        audit_like = [
            event
            for event in audit_records
            if _is_audit_candidate(event)
            and (
                expected_audit_title is None
                or (
                    isinstance(event.get("title"), str)
                    and event["title"].strip().casefold()
                    == expected_audit_title.casefold()
                )
            )
        ]
        selected_event_id = task.context.get("audit_event_id")
        if isinstance(selected_event_id, str) and selected_event_id.strip():
            # A caller that already chose one of several same-titled audits pins it by ID.
            audit_like = [
                event
                for event in audit_like
                if _event_id(event) == selected_event_id.strip()
            ]
        if (
            fallback_timezone is None
            and use_calendar_timezone
            and any(event.get("timezone") is None for event in audit_like)
        ):
            try:
                fallback_timezone = _resolve_calendar_timezone(tools, calendar_id)
            except CalendarEventListError as exc:
                return _clarification(
                    task,
                    reference,
                    "The owning calendar timezone could not be verified for timestamp fallback.",
                    str(exc),
                )
        reference = AuditReference(
            candidates=tuple(
                _event_ref(event, fallback_timezone) for event in audit_like
            ),
            search_terms=(search_term,),
        )
        if not audit_like:
            return _clarification(
                task,
                reference,
                "No audit candidate was found.",
                (
                    f"An event with the exact title {expected_audit_title!r} is required."
                    if expected_audit_title is not None
                    else "An event whose title or description contains 'audit' is required."
                ),
            )

        parsed_audits = [
            (event, _parse_event_times(event, fallback_timezone))
            for event in audit_like
        ]
        malformed_audits = [
            event
            for event, parsed in parsed_audits
            if parsed is None or _event_id(event) is None
        ]
        valid_audits = [
            (event, parsed)
            for event, parsed in parsed_audits
            if parsed is not None and _event_id(event) is not None
        ]
        if malformed_audits:
            return _clarification(
                task,
                reference,
                "Audit resolution is ambiguous because an audit candidate has invalid or timezone-ambiguous datetimes.",
                f"{len(malformed_audits)} audit candidate(s) could not be validated.",
            )
        if len(valid_audits) != 1:
            return _clarification(
                task,
                reference,
                "Which audit should be used?",
                f"Found {len(valid_audits)} valid audit candidates; exactly one must be selected.",
            )

        audit_event, audit_times = valid_audits[0]
        assert audit_times is not None

    _, audit_end = audit_times
    shift = task.context.get("shift")
    shift_was_proposed = shift is None
    if shift_was_proposed:
        shift = _DEFAULT_PROPOSED_SHIFT
    elif not isinstance(shift, timedelta) or shift <= timedelta(0):
        return _clarification(
            task,
            reference,
            "What positive shift duration amount should be used?",
            "No shift duration was provided; the planner will not invent one.",
        )

    try:
        events = _list_complete_events(tools, calendar_id=calendar_id)
    except CalendarEventListError as exc:
        return _clarification(
            task,
            reference,
            "The affected-event set may be incomplete.",
            str(exc),
        )
    if (
        fallback_timezone is None
        and use_calendar_timezone
        and any(event.get("timezone") is None for event in events)
    ):
        try:
            fallback_timezone = _resolve_calendar_timezone(tools, calendar_id)
        except CalendarEventListError as exc:
            return _clarification(
                task,
                reference,
                "The owning calendar timezone could not be verified for timestamp fallback.",
                str(exc),
            )
    audit_record_id = _event_id(audit_event)
    assert audit_record_id is not None
    matching_audits = [event for event in events if _event_id(event) == audit_record_id]
    if len(matching_audits) != 1:
        occurrence = "missing from" if not matching_audits else "duplicated in"
        selected_audit_id = (
            audit_number.strip()
            if isinstance(audit_number, str)
            else audit_record_id
        )
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because the resolved audit is not present exactly once.",
            f"The selected audit ID {selected_audit_id[:100]!r} was {occurrence} the full event listing "
            f"({len(matching_audits)} matches; expected exactly one).",
        )
    full_listing_audit_times = _parse_event_times(
        matching_audits[0], fallback_timezone
    )
    if (
        full_listing_audit_times is None
        or full_listing_audit_times != audit_times
        or (
            isinstance(audit_number, str)
            and matching_audits[0].get("number") != audit_number.strip()
        )
    ):
        return _clarification(
            task,
            reference,
            "Audit resolution is ambiguous because its interval changed between reads.",
            "The selected audit interval must match in the audit search and full event listing.",
        )
    event_id_counts: dict[str, int] = {}
    for event in events:
        event_id = _event_id(event)
        if event_id is not None:
            event_id_counts[event_id] = event_id_counts.get(event_id, 0) + 1
    duplicate_event_ids = sorted(
        event_id for event_id, count in event_id_counts.items() if count > 1
    )
    if duplicate_event_ids:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because event IDs are duplicated.",
            f"Duplicate event IDs: {', '.join(duplicate_event_ids)}.",
        )

    movable_events: list[tuple[datetime, datetime, dict[str, Any], ZoneInfo, str]] = []
    timezone_ambiguous: list[dict[str, Any]] = []
    malformed_events: list[dict[str, Any]] = []
    malformed_after_audit: list[dict[str, Any]] = []
    identity_ambiguous: list[dict[str, Any]] = []
    for event in events:
        event_id = _event_id(event)
        if event_id == audit_record_id or _is_background_event(event):
            continue
        parsed = _parse_event_times(event, fallback_timezone)
        if parsed is None:
            interval = _parse_event_interval(event, fallback_timezone)
            if interval is not None:
                if interval[0] > audit_end:
                    timezone_ambiguous.append(event)
            else:
                start = _parse_event_start(event, fallback_timezone)
                if start is not None and start > audit_end:
                    malformed_after_audit.append(event)
                else:
                    malformed_events.append(event)
            continue

        event_start, event_end = parsed
        if event_start <= audit_end:
            continue
        event_timezone = _event_timezone(event, fallback_timezone)
        if event_timezone is None:
            timezone_ambiguous.append(event)
            continue
        if event_id is None:
            identity_ambiguous.append(event)
            continue
        movable_events.append((event_start, event_end, event, event_timezone, event_id))

    if timezone_ambiguous:
        return _clarification(
            task,
            reference,
            "The affected-event boundary is ambiguous because event timezone data is missing.",
            f"{len(timezone_ambiguous)} event(s) have datetime values without an explicit timezone.",
        )
    if malformed_after_audit:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because an event after the audit has malformed timestamp or timezone data.",
            f"{len(malformed_after_audit)} event(s) that would be moved could not be validated.",
        )
    if malformed_events:
        sample_ids = [_event_id(event) or "<missing id>" for event in malformed_events[:5]]
        remaining = len(malformed_events) - len(sample_ids)
        sample = ", ".join(sample_ids)
        if remaining:
            sample += f", and {remaining} more"
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because event timestamps are unusable.",
            f"{len(malformed_events)} event(s) could not be classified or checked for destination conflicts; sample IDs: {sample}.",
        )
    if identity_ambiguous:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because a movable event has no valid identity.",
            f"{len(identity_ambiguous)} event(s) that would be moved have a missing or blank ID.",
        )

    if not movable_events:
        return _clarification(
            task,
            reference,
            "No valid events start strictly after the audit end.",
            "Events overlapping the audit or ending at its boundary are not affected.",
        )

    raw_max_shift_search = task.context.get("max_shift_search")
    if raw_max_shift_search is None:
        candidate_shifts = [shift]
    elif (
        not shift_was_proposed
        or not isinstance(raw_max_shift_search, timedelta)
        or raw_max_shift_search < shift
        or raw_max_shift_search > _MAX_ALTERNATIVE_SHIFT
        or (raw_max_shift_search - shift) % _SHIFT_SEARCH_INCREMENT != timedelta(0)
    ):
        return _clarification(
            task,
            reference,
            "The alternative shift search bound is invalid.",
            "Only a proposed shift may use a positive, 30-minute-aligned search bound up to eight hours.",
        )
    else:
        candidate_shifts = [
            shift + _SHIFT_SEARCH_INCREMENT * step
            for step in range(
                int((raw_max_shift_search - shift) / _SHIFT_SEARCH_INCREMENT) + 1
            )
        ]

    def build_moves(candidate_shift: timedelta) -> list[EventMove]:
        proposed: list[tuple[datetime, EventMove]] = []
        for event_start, event_end, event, event_timezone, event_id in movable_events:
            proposed_start = (event_start + candidate_shift).astimezone(event_timezone)
            proposed_end = (event_end + candidate_shift).astimezone(event_timezone)
            proposed.append(
                (
                    event_start,
                    EventMove(
                        event_id=event_id,
                        title=str(event.get("title", "")),
                        original_start_at=_normalized_timestamp(
                            str(event["start_at"]), event_timezone
                        ),
                        original_end_at=_normalized_timestamp(
                            str(event["end_at"]), event_timezone
                        ),
                        proposed_start_at=proposed_start.isoformat(),
                        proposed_end_at=proposed_end.isoformat(),
                        duration_seconds=(event_end - event_start).total_seconds(),
                        calendar_id=(
                            event.get("calendar_id")
                            if isinstance(event.get("calendar_id"), str)
                            else None
                        ),
                        timezone=event_timezone.key,
                    ),
                )
            )
        proposed.sort(key=lambda item: item[0])
        return [move for _, move in proposed]

    def find_conflicts(candidate_moves: list[EventMove]) -> set[str]:
        candidate_move_ids = {move.event_id for move in candidate_moves}
        found: set[str] = set()
        for index, move in enumerate(candidate_moves):
            proposed_start = datetime.fromisoformat(
                move.proposed_start_at.replace("Z", "+00:00")
            )
            proposed_end = datetime.fromisoformat(
                move.proposed_end_at.replace("Z", "+00:00")
            )
            for other_move in candidate_moves[index + 1 :]:
                other_start = datetime.fromisoformat(
                    other_move.proposed_start_at.replace("Z", "+00:00")
                )
                other_end = datetime.fromisoformat(
                    other_move.proposed_end_at.replace("Z", "+00:00")
                )
                if _intervals_overlap(
                    proposed_start, proposed_end, other_start, other_end
                ):
                    # A uniform shift preserves overlaps between moved events,
                    # so only pairs that did not already overlap are new conflicts.
                    original_start = datetime.fromisoformat(
                        move.original_start_at.replace("Z", "+00:00")
                    )
                    original_end = datetime.fromisoformat(
                        move.original_end_at.replace("Z", "+00:00")
                    )
                    other_original_start = datetime.fromisoformat(
                        other_move.original_start_at.replace("Z", "+00:00")
                    )
                    other_original_end = datetime.fromisoformat(
                        other_move.original_end_at.replace("Z", "+00:00")
                    )
                    if _intervals_overlap(
                        original_start,
                        original_end,
                        other_original_start,
                        other_original_end,
                    ):
                        continue
                    found.add(f"{move.event_id} with {other_move.event_id}")
            for event in events:
                event_id = _event_id(event)
                if (
                    event_id in candidate_move_ids
                    or event_id == audit_record_id
                    or _is_background_event(event)
                ):
                    continue
                parsed = _parse_event_interval(event, fallback_timezone)
                if parsed is not None and _intervals_overlap(
                    proposed_start, proposed_end, *parsed
                ):
                    found.add(
                        f"{move.event_id} with {event_id or 'an unidentified event'}"
                    )
        return found

    moves: list[EventMove] = []
    initial_conflicts: set[str] = set()
    selected_shift: timedelta | None = None
    for candidate_shift in candidate_shifts:
        candidate_moves = build_moves(candidate_shift)
        conflicts = find_conflicts(candidate_moves)
        if not conflicts:
            selected_shift = candidate_shift
            moves = candidate_moves
            break
        if not initial_conflicts:
            initial_conflicts = conflicts

    if selected_shift is None:
        if len(candidate_shifts) > 1:
            return _clarification(
                task,
                reference,
                "The 30-minute proposal conflicts, and no conflict-free alternative was found within the eight-hour limit.",
                f"Checked {len(candidate_shifts)} shifts in 30-minute increments; no event was updated.",
            )
        return _clarification(
            task,
            reference,
            "The rescheduling proposal conflicts with an existing event.",
            "Conflicting proposed intervals: " + ", ".join(sorted(initial_conflicts)) + ".",
        )

    shift = selected_shift

    fallback_note = (
        f" The configured {fallback_timezone.key} timezone fallback is an explicit planning assumption, not a verified platform contract."
        if fallback_timezone is not None
        else ""
    )
    requested_shift = _DEFAULT_PROPOSED_SHIFT if shift_was_proposed else shift
    selected_minutes = int(shift.total_seconds() // 60)
    if shift_was_proposed and shift != requested_shift:
        confirmation_prompt = (
            "The initial 30-minute shift conflicts. The first conflict-free "
            f"alternative is {selected_minutes} minutes. Confirm this alternative "
            "or provide a different positive shift before any update."
        )
        plan_summary = (
            "The 30-minute proposal conflicts; a conflict-free "
            f"{selected_minutes}-minute alternative plan is ready."
        )
    elif shift_was_proposed:
        confirmation_prompt = (
            "Confirm the proposed 30-minute shift or provide a different "
            "positive duration before any update."
        )
        plan_summary = "A read-only 30-minute rescheduling plan is ready."
    else:
        confirmation_prompt = (
            "Confirm this plan or provide a different positive shift duration "
            "before any update."
        )
        plan_summary = "A read-only rescheduling plan is ready."
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.PLANNED,
        summary=(
            plan_summary
            + fallback_note
            + " Confirmation is required; no events have been moved."
        ),
        audit_reference=reference,
        rescheduling_plan=moves,
        claimed_outcome={
            "moved": False,
            "planned_event_count": len(moves),
            "shift": shift,
            "requested_shift": requested_shift,
            "shift_was_proposed": shift_was_proposed,
            "alternative_shift_proposed": shift != requested_shift,
            "confirmation_required": True,
            "confirmation_prompt": confirmation_prompt,
            "timestamp_timezone_fallback": (
                fallback_timezone.key if fallback_timezone is not None else None
            ),
            "timestamp_timezone_fallback_is_platform_contract": False,
        },
    )


def _move_signature(result: AgentResult) -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                move.event_id,
                move.original_start_at,
                move.original_end_at,
                move.proposed_start_at,
                move.proposed_end_at,
            )
            for move in result.rescheduling_plan
        )
    )


def _execution_failure(
    result: AgentResult,
    *,
    summary: str,
    completed_updates: int,
    attempted_updates: int | None = None,
) -> AgentResult:
    return AgentResult(
        goal=result.goal,
        status=ResultStatus.FAILED,
        summary=summary,
        audit_reference=result.audit_reference,
        rescheduling_plan=list(result.rescheduling_plan),
        claimed_outcome={
            "updates_requested": len(result.rescheduling_plan),
            "updates_attempted": (
                completed_updates
                if attempted_updates is None
                else attempted_updates
            ),
            "updates_completed": completed_updates,
            "verification_passed": False,
        },
    )


def apply_confirmed_move(
    task: AgentTask,
    proposal: AgentResult,
    tools: CalendarTool,
    *,
    confirmation_text: object,
    allow_disposable_writes: bool,
) -> AgentResult:
    """Apply a fresh, conflict-checked plan only after exact human confirmation."""

    if type(allow_disposable_writes) is not bool:
        raise TypeError("allow_disposable_writes must be a boolean")
    if proposal.goal != GOAL or proposal.status is not ResultStatus.PLANNED:
        return _clarification(
            task,
            proposal.audit_reference,
            "There is no eligible move proposal to execute.",
            "Create a fresh conflict-free proposal before confirming a move.",
        )

    proposal_outcome = proposal.claimed_outcome
    proposed_shift = proposal_outcome.get("shift")
    if (
        not isinstance(proposed_shift, timedelta)
        or proposal_outcome.get("confirmation_required") is not True
    ):
        return _clarification(
            task,
            proposal.audit_reference,
            "The proposal does not contain a confirmable shift.",
            "A positive, conflict-free shift proposal is required.",
        )
    if not confirms_shift(confirmation_text, proposed_shift):
        return _clarification(
            task,
            proposal.audit_reference,
            "The confirmation did not explicitly approve the proposed shift.",
            "Reply with an affirmative statement that repeats the exact proposed duration.",
        )
    if not allow_disposable_writes or not getattr(
        tools, "disposable_mutations_enabled", False
    ):
        return _clarification(
            task,
            proposal.audit_reference,
            "Calendar updates are disabled in this execution context.",
            "No events were changed; updates are permitted only through an explicitly enabled disposable-copy tool.",
        )

    fresh_context = dict(task.context)
    fresh_context.pop("confirmation_text", None)
    fresh_context.pop("max_shift_search", None)
    fresh_context["shift"] = proposed_shift
    fresh_task = AgentTask(task.goal, task.instruction, fresh_context)
    fresh_proposal = plan(fresh_task, tools)
    if (
        fresh_proposal.status is not ResultStatus.PLANNED
        or fresh_proposal.claimed_outcome.get("shift") != proposed_shift
    ):
        return _clarification(
            task,
            fresh_proposal.audit_reference,
            "The calendar changed or the confirmed shift is no longer conflict-free.",
            "Review the updated calendar and confirm a new proposal; no events were changed.",
        )
    if _move_signature(fresh_proposal) != _move_signature(proposal):
        fresh_proposal.summary = (
            "The calendar state changed after the prior proposal. "
            "Review and confirm this refreshed plan; no events were changed."
        )
        fresh_proposal.claimed_outcome["confirmation_required"] = True
        fresh_proposal.claimed_outcome["confirmation_prompt"] = (
            "The plan changed. Confirm the displayed shift again before any update."
        )
        return fresh_proposal

    reference = fresh_proposal.audit_reference
    if reference is None or len(reference.candidates) != 1:
        return _clarification(
            task,
            reference,
            "The confirmed audit reference is no longer unique.",
            "No events were changed.",
        )
    calendar_id = reference.candidates[0].calendar_id
    if not isinstance(calendar_id, str) or not calendar_id.strip():
        return _clarification(
            task,
            reference,
            "The confirmed audit has no verified calendar scope.",
            "No events were changed.",
        )

    fallback_name = fresh_proposal.claimed_outcome.get(
        "timestamp_timezone_fallback"
    )
    fallback_timezone = _parse_timezone(fallback_name)
    try:
        current_events = _list_complete_events(
            tools,
            calendar_id=calendar_id,
        )
    except CalendarEventListError as exc:
        return _clarification(
            task,
            reference,
            "The calendar changed or could not be re-read before updates.",
            str(exc),
        )

    events_by_id: dict[str, dict[str, Any]] = {}
    for event in current_events:
        event_id = _event_id(event)
        if event_id is None or event_id in events_by_id:
            return _clarification(
                task,
                reference,
                "The calendar state is ambiguous before updates.",
                "Event identities must be present and unique; no events were changed.",
            )
        events_by_id[event_id] = event

    audit_ref = reference.candidates[0]
    audit_matches = [
        event
        for event in current_events
        if event.get("number") == audit_ref.id
        or _event_id(event) == audit_ref.id
    ]
    if len(audit_matches) != 1:
        return _clarification(
            task,
            reference,
            "The confirmed audit is missing or ambiguous in the current calendar.",
            "No events were changed.",
        )
    current_audit_times = _parse_event_times(
        audit_matches[0], fallback_timezone
    )
    reference_times = (
        _parse_event_times(
            {
                "start_at": audit_ref.start_at,
                "end_at": audit_ref.end_at,
                "timezone": audit_ref.timezone,
            },
            fallback_timezone,
        )
    )
    if current_audit_times is None or current_audit_times != reference_times:
        return _clarification(
            task,
            reference,
            "The audit interval changed after the proposal.",
            "Review and confirm a new proposal; no events were changed.",
        )
    audit_end = current_audit_times[1]

    expected_move_ids: set[str] = set()
    for event in current_events:
        event_id = _event_id(event)
        if event_id == _event_id(audit_matches[0]):
            continue
        interval = _parse_event_times(event, fallback_timezone)
        if interval is None:
            return _clarification(
                task,
                reference,
                "An event could not be revalidated before updates.",
                "The move was cancelled; no events were changed.",
            )
        if interval[0] > audit_end:
            expected_move_ids.add(event_id or "")

    planned_moves = {move.event_id: move for move in fresh_proposal.rescheduling_plan}
    if (
        not expected_move_ids
        or "" in expected_move_ids
        or set(planned_moves) != expected_move_ids
    ):
        return _clarification(
            task,
            reference,
            "The affected-event set changed after the proposal.",
            "Review and confirm a new proposal; no events were changed.",
        )
    for event_id, move in planned_moves.items():
        event = events_by_id[event_id]
        current_times = _parse_event_times(event, fallback_timezone)
        move_original_times = _parse_event_times(
            {
                "start_at": move.original_start_at,
                "end_at": move.original_end_at,
                "timezone": move.timezone,
            },
            fallback_timezone,
        )
        if current_times is None or current_times != move_original_times:
            return _clarification(
                task,
                reference,
                "A planned event changed after the proposal.",
                "Review and confirm a new proposal; no events were changed.",
            )

    attempted_updates = 0
    completed_updates = 0
    for move in fresh_proposal.rescheduling_plan:
        attempted_updates += 1
        try:
            response = tools.update_event(
                id=move.event_id,
                start_at=move.proposed_start_at,
                end_at=move.proposed_end_at,
            )
        except (CalendarMutationError, MutationDisabledError) as exc:
            return _execution_failure(
                fresh_proposal,
                summary=(
                    f"Calendar update stopped after {attempted_updates} update "
                    f"attempt(s); {completed_updates} response(s) were verified "
                    f"({type(exc).__name__}). The disposable copy may contain "
                    "partial test changes; reread state before any retry."
                ),
                completed_updates=completed_updates,
                attempted_updates=attempted_updates,
            )
        if not isinstance(response, dict) or response.get("id") != move.event_id:
            return _execution_failure(
                fresh_proposal,
                summary=(
                    f"Calendar update response was not verifiable after "
                    f"{attempted_updates} update attempt(s); {completed_updates} "
                    "response(s) were verified. The disposable copy may contain "
                    "partial test changes; reread state before any retry."
                ),
                completed_updates=completed_updates,
                attempted_updates=attempted_updates,
            )
        completed_updates += 1

    try:
        updated_events = _list_complete_events(tools, calendar_id=calendar_id)
    except CalendarEventListError as exc:
        return _execution_failure(
            fresh_proposal,
            summary=(
                f"All {completed_updates} updates were requested, but the final "
                f"calendar read failed ({type(exc).__name__}); completion is unverified."
            ),
            completed_updates=completed_updates,
            attempted_updates=attempted_updates,
        )

    updated_by_id = {
        event_id: event
        for event in updated_events
        if (event_id := _event_id(event)) is not None
    }
    if (
        len(updated_by_id) != len(updated_events)
        or set(updated_by_id) != set(events_by_id)
    ):
        return _execution_failure(
            fresh_proposal,
            summary=(
                "Final calendar read contains missing, duplicate, or unexpected "
                "event IDs."
            ),
            completed_updates=completed_updates,
            attempted_updates=attempted_updates,
        )
    planned_event_ids = set(planned_moves)
    for event_id, event in events_by_id.items():
        if event_id in planned_event_ids:
            continue
        if event != updated_by_id[event_id]:
            return _execution_failure(
                fresh_proposal,
                summary=(
                    "An event outside the confirmed move set changed during "
                    f"execution; {attempted_updates} planned update attempt(s) "
                    "were issued."
                ),
                completed_updates=completed_updates,
                attempted_updates=attempted_updates,
            )
    for move in fresh_proposal.rescheduling_plan:
        event = updated_by_id.get(move.event_id)
        actual_times = (
            _parse_event_times(event, fallback_timezone)
            if event is not None
            else None
        )
        expected_times = _parse_event_times(
            {
                "start_at": move.proposed_start_at,
                "end_at": move.proposed_end_at,
                "timezone": move.timezone,
            },
            fallback_timezone,
        )
        if actual_times is None or actual_times != expected_times:
            return _execution_failure(
                fresh_proposal,
                summary=(
                    f"Post-update verification failed after {completed_updates} "
                    "event updates; completion is unverified."
                ),
                completed_updates=completed_updates,
                attempted_updates=attempted_updates,
            )

    return AgentResult(
        goal=task.goal,
        status=ResultStatus.COMPLETED,
        summary=(
            f"Confirmed shift applied and verified for {completed_updates} "
            "event(s) in the disposable calendar copy."
        ),
        audit_reference=fresh_proposal.audit_reference,
        rescheduling_plan=list(fresh_proposal.rescheduling_plan),
        claimed_outcome={
            "moved": True,
            "shift": proposed_shift,
            "confirmation_required": False,
            "updates_attempted": attempted_updates,
            "updates_verified": completed_updates,
        },
    )