"""Offline-first mapping for the Keystone Calendar Agent's resource reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from calendar_agent.models import CalendarEventRef, EventAttendee, PartyRecord

from .client import (
    KeystoneClient,
    KeystoneClientError,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
)
from .authorization import (
    GOAL_ONE_LIST_READ_OPERATIONS,
    RESOURCE_READ_OPERATIONS,
    validate_json_schema_instance,
)
from .discovery import CompletenessStatus
from .pagination import (
    MAX_RESOURCE_PAGES,
    MAX_RESOURCE_RECORDS,
    ResourceCollectionResult,
    collect_resource_pages,
)


_PENDING_PAGE_REASONS = frozenset(
    {
        "The final observed page says collection should continue.",
        "Observed record count does not match the expected total.",
    }
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class KeystoneListRequest:
    """A recorded request description; it is not an executable tool handle."""

    tool_name: str
    arguments: dict[str, Any]


class _KeystonePageSchemaValidationError(KeystoneProtocolError):
    def __init__(
        self,
        message: str,
        requests: tuple[KeystoneListRequest, ...],
    ) -> None:
        self.requests = requests
        super().__init__(message)


@dataclass(frozen=True)
class KeystoneCalendarRecord:
    id: str
    owner_party_id: str | None
    timezone: str | None


@dataclass(frozen=True)
class KeystoneMappedRecord(Generic[_T]):
    """One source record with explicit mapping uncertainty."""

    source_id: str | None
    value: _T | None
    unresolved_fields: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return (
            self.source_id is not None
            and self.value is not None
            and not self.unresolved_fields
            and not self.errors
        )


@dataclass(frozen=True)
class KeystoneResourceListResult(Generic[_T]):
    """A paginated read result that keeps raw completeness evidence visible."""

    requests: tuple[KeystoneListRequest, ...]
    pagination: ResourceCollectionResult
    records: tuple[KeystoneMappedRecord[_T], ...] | None
    read_error: str | None = None

    @property
    def collection_complete(self) -> bool:
        return self.pagination.is_complete

    @property
    def is_complete(self) -> bool:
        return (
            self.collection_complete
            and self.records is not None
            and all(record.is_resolved for record in self.records)
            and self.read_error is None
        )


def _text_field(
    record: Mapping[str, Any],
    field_name: str,
    *,
    required: bool,
    unresolved: list[str],
    errors: list[str],
) -> str | None:
    if field_name not in record or record[field_name] is None:
        if required:
            errors.append(f"Required field {field_name} is absent.")
        else:
            unresolved.append(field_name)
        return None
    value = record[field_name]
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name} must be a non-empty string.")
        return None
    return value


def map_party_record(
    record: Mapping[str, Any],
) -> KeystoneMappedRecord[PartyRecord]:
    unresolved: list[str] = []
    errors: list[str] = []
    source_id = _text_field(
        record, "id", required=True, unresolved=unresolved, errors=errors
    )
    name = _text_field(
        record, "name", required=True, unresolved=unresolved, errors=errors
    )
    email = _text_field(
        record, "email", required=True, unresolved=unresolved, errors=errors
    )
    job_title = _text_field(
        record, "job_title", required=True, unresolved=unresolved, errors=errors
    )
    value = None
    if (
        source_id is not None
        and name is not None
        and email is not None
        and job_title is not None
        and not errors
    ):
        value = PartyRecord(
            id=source_id,
            name=name,
            email=email,
            job_title=job_title,
        )
    return KeystoneMappedRecord(
        source_id=source_id,
        value=value,
        unresolved_fields=tuple(unresolved),
        errors=tuple(errors),
    )


def map_calendar_record(
    record: Mapping[str, Any],
) -> KeystoneMappedRecord[KeystoneCalendarRecord]:
    unresolved: list[str] = []
    errors: list[str] = []
    source_id = _text_field(
        record, "id", required=True, unresolved=unresolved, errors=errors
    )
    owner_party_id = _text_field(
        record,
        "owner_party_id",
        required=False,
        unresolved=unresolved,
        errors=errors,
    )
    timezone = _text_field(
        record,
        "timezone",
        required=False,
        unresolved=unresolved,
        errors=errors,
    )
    value = None
    if source_id is not None and not errors:
        value = KeystoneCalendarRecord(
            id=source_id,
            owner_party_id=owner_party_id,
            timezone=timezone,
        )
    return KeystoneMappedRecord(
        source_id=source_id,
        value=value,
        unresolved_fields=tuple(unresolved),
        errors=tuple(errors),
    )


def map_calendar_event_record(
    record: Mapping[str, Any],
) -> KeystoneMappedRecord[CalendarEventRef]:
    unresolved: list[str] = []
    errors: list[str] = []
    source_id = _text_field(
        record, "id", required=True, unresolved=unresolved, errors=errors
    )
    title = _text_field(
        record, "title", required=True, unresolved=unresolved, errors=errors
    )
    start_at = _text_field(
        record, "start_at", required=True, unresolved=unresolved, errors=errors
    )
    end_at = _text_field(
        record, "end_at", required=True, unresolved=unresolved, errors=errors
    )
    calendar_id = _text_field(
        record,
        "calendar_id",
        required=False,
        unresolved=unresolved,
        errors=errors,
    )
    status = _text_field(
        record, "status", required=False, unresolved=unresolved, errors=errors
    )
    timezone = None
    if record.get("timezone") is not None:
        timezone = _text_field(
            record, "timezone", required=True, unresolved=unresolved, errors=errors
        )
    value = None
    if (
        source_id is not None
        and title is not None
        and start_at is not None
        and end_at is not None
        and not errors
    ):
        value = CalendarEventRef(
            id=source_id,
            title=title,
            start_at=start_at,
            end_at=end_at,
            calendar_id=calendar_id,
            status=status,
            timezone=timezone,
        )
    return KeystoneMappedRecord(
        source_id=source_id,
        value=value,
        unresolved_fields=tuple(unresolved),
        errors=tuple(errors),
    )


def map_event_attendee_record(
    record: Mapping[str, Any],
) -> KeystoneMappedRecord[EventAttendee]:
    unresolved: list[str] = []
    errors: list[str] = []
    source_id = _text_field(
        record, "id", required=True, unresolved=unresolved, errors=errors
    )
    event_id = _text_field(
        record, "event_id", required=True, unresolved=unresolved, errors=errors
    )
    email = _text_field(
        record, "email", required=True, unresolved=unresolved, errors=errors
    )
    party_id = _text_field(
        record,
        "party_id",
        required=False,
        unresolved=unresolved,
        errors=errors,
    )
    status = _text_field(
        record, "status", required=True, unresolved=unresolved, errors=errors
    )

    organizer_present = "is_organizer" in record
    organizer = record.get("is_organizer")
    if organizer_present and type(organizer) is not bool:
        errors.append("is_organizer must be a boolean when present.")

    raw_role = record.get("role")
    role: str | None = None
    if raw_role is None:
        if organizer is True:
            role = "organizer"
        else:
            unresolved.append("role")
    elif not isinstance(raw_role, str):
        errors.append("role must be a string when present.")
    elif organizer is True:
        errors.append("role and is_organizer cannot both be represented by one shared role.")
    else:
        role = {
            "req": "required",
            "opt": "optional",
            "chair": "chair",
        }.get(raw_role)
        if role is None:
            errors.append("role is not one of the supported Keystone attendee values.")

    supported_statuses = {"accepted", "declined", "tentative", "needs_action"}
    if status is not None and status not in supported_statuses:
        errors.append("status is not a supported attendee response.")

    value = None
    if (
        source_id is not None
        and event_id is not None
        and email is not None
        and role is not None
        and status is not None
        and not errors
    ):
        value = EventAttendee(
            event_id=event_id,
            party_id=party_id,
            email=email,
            role=role,
            response_status=status,
        )
    return KeystoneMappedRecord(
        source_id=source_id,
        value=value,
        unresolved_fields=tuple(unresolved),
        errors=tuple(errors),
    )


def map_goal_one_record(
    record: Mapping[str, Any],
) -> KeystoneMappedRecord[dict[str, Any]]:
    source_id = record.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        return KeystoneMappedRecord(
            source_id=None,
            value=None,
            errors=("record has no valid string ID",),
        )
    return KeystoneMappedRecord(source_id=source_id, value=dict(record))


class KeystoneCalendarAdapter:
    """Allowlisted resource adapter with opt-in disposable event mutations."""

    def __init__(
        self,
        client: KeystoneClient,
        *,
        synthetic_response_pages: Mapping[
            tuple[str, int], Mapping[str, Any] | KeystoneClientError
        ] | None = None,
        max_pages: int = MAX_RESOURCE_PAGES,
        max_records: int = MAX_RESOURCE_RECORDS,
    ) -> None:
        if type(client) is not KeystoneClient:
            raise TypeError("client must be the concrete KeystoneClient")
        if synthetic_response_pages is not None and not isinstance(
            synthetic_response_pages, Mapping
        ):
            raise TypeError("synthetic_response_pages must be a mapping")
        collect_resource_pages([], max_pages=max_pages, max_records=max_records)
        self._client = client
        self._synthetic_response_pages = synthetic_response_pages
        self._max_pages = max_pages
        self._max_records = max_records

    @property
    def disposable_mutations_enabled(self) -> bool:
        return self._client.hosted_mutation_transport_enabled

    def _create_event_on_disposable_copy(
        self, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        authorization = self._client.authorize_mutation_request(
            "CalendarEvent.create", arguments
        )
        return self._client.call_authorized_mutation(authorization)

    def _update_event_on_disposable_copy(
        self, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        authorization = self._client.authorize_mutation_request(
            "CalendarEvent.update", arguments
        )
        return self._client.call_authorized_mutation(authorization)

    def list_parties(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> KeystoneResourceListResult[PartyRecord]:
        return self._list_resource("Party.list", filters, limit, map_party_record)

    def list_calendars(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> KeystoneResourceListResult[KeystoneCalendarRecord]:
        return self._list_resource("Calendar.list", filters, limit, map_calendar_record)

    def list_events(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> KeystoneResourceListResult[CalendarEventRef]:
        return self._list_resource(
            "CalendarEvent.list", filters, limit, map_calendar_event_record
        )

    def list_event_attendees(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> KeystoneResourceListResult[EventAttendee]:
        return self._list_resource(
            "EventAttendee.list", filters, limit, map_event_attendee_record
        )

    def list_scheduling_preferences(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> KeystoneResourceListResult[dict[str, Any]]:
        return self._list_resource(
            "SchedulingPreferences.list", filters, limit, map_goal_one_record
        )

    def list_availability_rules(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> KeystoneResourceListResult[dict[str, Any]]:
        return self._list_resource(
            "AvailabilityRule.list", filters, limit, map_goal_one_record
        )

    def read_goal_one_endpoint(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> tuple[KeystoneListRequest, dict[str, Any]]:
        if operation != "endpoint.calendar.free_busy":
            raise KeystoneOperationNotAllowedError(
                "Keystone endpoint is not in the Goal 1 read allowlist."
            )
        request = KeystoneListRequest(operation, dict(arguments))
        authorization = self._client.authorize_read_request(operation, request.arguments)
        descriptor = self._client.validate_read_authorization(authorization)
        if self._synthetic_response_pages is None:
            response = self._client.call_authorized_read(authorization)
        else:
            response = self._synthetic_response_pages.get((operation, 0))
            if response is None:
                raise KeystoneProtocolError(
                    f"{operation} synthetic response data is missing."
                )
            if isinstance(response, KeystoneClientError):
                raise response
        if not isinstance(response, Mapping):
            raise KeystoneProtocolError(
                f"{operation} structured response is not an object."
            )
        validate_json_schema_instance(
            descriptor.output_schema,
            response,
            location=(
                "hosted Goal 1 endpoint response"
                if self._synthetic_response_pages is None
                else "synthetic Goal 1 endpoint response"
            ),
        )
        return request, dict(response)

    def _list_resource(
        self,
        tool_name: str,
        filters: Mapping[str, Any] | None,
        limit: int,
        mapper: Callable[[Mapping[str, Any]], KeystoneMappedRecord[_T]],
    ) -> KeystoneResourceListResult[_T]:
        if tool_name not in RESOURCE_READ_OPERATIONS | GOAL_ONE_LIST_READ_OPERATIONS:
            raise KeystoneOperationNotAllowedError(
                "Keystone Calendar adapter operation is not allowlisted."
            )
        arguments = self._list_arguments(filters, limit)
        pages: list[Mapping[str, Any]] = []
        requests: list[KeystoneListRequest] = []
        pagination = collect_resource_pages(
            pages,
            max_pages=self._max_pages,
            max_records=self._max_records,
            collection_reference=tool_name,
        )
        read_error: str | None = None

        for page_index in range(self._max_pages):
            request = KeystoneListRequest(tool_name, dict(arguments))
            authorization = self._client.authorize_read_request(
                tool_name, request.arguments
            )
            self._client.validate_read_authorization(authorization)
            requests.append(request)
            try:
                if self._synthetic_response_pages is None:
                    response = self._client.call_authorized_read(authorization)
                else:
                    response_key = (tool_name, request.arguments["offset"])
                    if response_key not in self._synthetic_response_pages:
                        read_error = "SyntheticResponseMissing"
                        break
                    response = self._synthetic_response_pages[response_key]
                    if isinstance(response, KeystoneClientError):
                        raise response
            except KeystoneClientError as exc:
                read_error = type(exc).__name__
                break
            try:
                validate_json_schema_instance(
                    self._client.validate_read_authorization(authorization).output_schema,
                    response,
                    location=(
                        "hosted structured resource response"
                        if self._synthetic_response_pages is None
                        else "synthetic structured resource response"
                    ),
                )
            except KeystoneProtocolError as exc:
                raise _KeystonePageSchemaValidationError(
                    str(exc),
                    tuple(requests),
                ) from exc
            if not isinstance(response, Mapping):
                pages.append({})
            else:
                pages.append(response)
            pagination = collect_resource_pages(
                pages,
                max_pages=self._max_pages,
                max_records=self._max_records,
                collection_reference=tool_name,
            )
            if pagination.is_complete:
                break
            if (
                page_index + 1 >= self._max_pages
                or not self._needs_next_page(pagination)
            ):
                break
            last_response = pages[-1]
            data = last_response["data"]
            arguments = {
                **arguments,
                "offset": last_response["offset"] + len(data),
            }

        mapped_records = None
        if pagination.is_complete and pagination.records is not None:
            mapped_records = tuple(mapper(record) for record in pagination.records)
        return KeystoneResourceListResult(
            requests=tuple(requests),
            pagination=pagination,
            records=mapped_records,
            read_error=read_error,
        )

    @staticmethod
    def _needs_next_page(result: ResourceCollectionResult) -> bool:
        if result.completeness.status is not CompletenessStatus.INCOMPLETE:
            return False
        if not result.collection_evidence.pages:
            return False
        if result.collection_evidence.pages[-1].has_more is not True:
            return False
        return set(result.completeness.reasons).issubset(_PENDING_PAGE_REASONS)

    @staticmethod
    def _list_arguments(
        filters: Mapping[str, Any] | None,
        limit: int,
    ) -> dict[str, Any]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        if filters is None:
            filters = {}
        if not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping")
        if "limit" in filters or "offset" in filters:
            raise ValueError("limit and offset must be supplied through adapter parameters")
        if any(not isinstance(name, str) for name in filters):
            raise ValueError("filter names must be strings")
        return {**filters, "limit": limit, "offset": 0}
