"""Offline adapter bridge for the existing Calendar Agent goal handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Callable, NoReturn, TypeVar
from zoneinfo import ZoneInfo

from calendar_agent.agent import CalendarAgent
from calendar_agent.models import (
    AgentResult,
    AgentTask,
    CalendarEventPage,
    CalendarEventRef,
    CalendarMutationError,
    EventAttendee,
    MutationDisabledError,
    PartyRecord,
    ResultStatus,
    ToolCallRecord,
)

from .calendar_adapter import (
    KeystoneCalendarAdapter,
    KeystoneCalendarRecord,
    KeystoneListRequest,
    KeystoneMappedRecord,
    KeystoneResourceListResult,
    _KeystonePageSchemaValidationError,
)
from .client import (
    KeystoneClientError,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
)
from .discovery import CompletenessStatus


class KeystoneGoalIntegrationError(RuntimeError):
    """A verified-data failure while adapting Keystone resources to a goal."""

    category = "integration_error"


class KeystoneCapabilityUnavailableError(KeystoneGoalIntegrationError):
    category = "capability_unavailable"


class KeystoneAuthorizationFailureError(KeystoneGoalIntegrationError):
    category = "authorization_failure"


class KeystoneResourceReadError(KeystoneGoalIntegrationError):
    category = "resource_read_failure"


class KeystoneIncompleteDataError(KeystoneGoalIntegrationError):
    category = "incomplete_data"


class KeystoneRecordMappingError(KeystoneGoalIntegrationError):
    category = "record_mapping_failure"


class KeystoneSchemaValidationError(KeystoneGoalIntegrationError):
    category = "schema_validation_failure"


_T = TypeVar("_T")


class KeystoneCalendarTool:
    """CalendarTool facade with disabled-by-default disposable mutations."""

    def __init__(
        self,
        adapter: KeystoneCalendarAdapter,
        *,
        allow_disposable_mutations: bool = False,
    ) -> None:
        if type(adapter) is not KeystoneCalendarAdapter:
            raise TypeError("adapter must be the concrete KeystoneCalendarAdapter")
        if type(allow_disposable_mutations) is not bool:
            raise TypeError("allow_disposable_mutations must be a boolean")
        self._adapter = adapter
        self._allow_disposable_mutations = allow_disposable_mutations
        self.trace: list[ToolCallRecord] = []

    @property
    def disposable_mutations_enabled(self) -> bool:
        return (
            self._allow_disposable_mutations
            and self._adapter.disposable_mutations_enabled
        )

    def _record_requests(
        self,
        requests: tuple[KeystoneListRequest, ...],
        failure: KeystoneGoalIntegrationError | None = None,
        *,
        tool_name: str | None = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> None:
        if not requests:
            if failure is not None and tool_name is not None:
                self.trace.append(
                    ToolCallRecord(
                        tool_name,
                        dict(arguments or {}),
                        succeeded=False,
                        error=failure.category,
                    )
                )
            return
        for index, request in enumerate(requests):
            failed = failure is not None and index == len(requests) - 1
            self.trace.append(
                ToolCallRecord(
                    request.tool_name,
                    request.arguments,
                    succeeded=not failed,
                    error=failure.category if failed and failure is not None else None,
                )
            )

    @staticmethod
    def _mapping_error(
        tool_name: str,
        record: KeystoneMappedRecord[Any],
    ) -> KeystoneGoalIntegrationError | None:
        if record.errors:
            return KeystoneRecordMappingError(
                f"{tool_name} record {record.source_id!r} is malformed: "
                + "; ".join(record.errors)
            )
        if record.value is None or record.source_id is None:
            return KeystoneRecordMappingError(
                f"{tool_name} record has no validated identity or mapped value."
            )
        if record.unresolved_fields:
            return KeystoneIncompleteDataError(
                f"{tool_name} record {record.source_id!r} has unresolved fields: "
                + ", ".join(record.unresolved_fields)
            )
        return None

    @staticmethod
    def _to_goal_record(
        tool_name: str,
        record: KeystoneMappedRecord[Any],
        expected_type: type[_T],
    ) -> dict[str, Any]:
        failure = KeystoneCalendarTool._mapping_error(tool_name, record)
        if failure is not None:
            raise failure
        value = record.value
        if not isinstance(value, expected_type):
            raise KeystoneRecordMappingError(
                f"{tool_name} record {record.source_id!r} has an unexpected mapped type."
            )

        if isinstance(value, EventAttendee):
            mapped = {
                "id": record.source_id,
                "event_id": value.event_id,
                "party_id": value.party_id,
                "email": value.email,
                "role": value.role,
                "status": value.response_status,
            }
        elif isinstance(value, dict):
            mapped = dict(value)
        else:
            mapped = asdict(value)
        if mapped.get("id") != record.source_id:
            raise KeystoneRecordMappingError(
                f"{tool_name} record identity changed during mapping."
            )

        timezone_name = mapped.get("timezone")
        if timezone_name is not None:
            if not isinstance(timezone_name, str):
                raise KeystoneRecordMappingError(
                    f"{tool_name} record {record.source_id!r} has an invalid timezone type."
                )
            try:
                ZoneInfo(timezone_name)
            except (KeyError, OSError, ValueError):
                raise KeystoneRecordMappingError(
                    f"{tool_name} record {record.source_id!r} has an invalid timezone identifier."
                ) from None
        return mapped

    def _list(
        self,
        tool_name: str,
        filters: Mapping[str, Any] | None,
        limit: int,
        reader: Callable[..., KeystoneResourceListResult[_T]],
        expected_type: type[Any],
    ) -> list[dict[str, Any]]:
        normalized_filters = {} if filters is None else dict(filters)
        arguments = {**normalized_filters, "limit": limit, "offset": 0}
        try:
            result = reader(filters=normalized_filters, limit=limit)
        except KeystoneProtocolError as exc:
            failure = KeystoneSchemaValidationError(str(exc))
            requests = (
                exc.requests
                if isinstance(exc, _KeystonePageSchemaValidationError)
                else ()
            )
            self._record_requests(
                requests,
                failure,
                tool_name=tool_name,
                arguments=arguments,
            )
            raise failure from None
        except KeystoneOperationNotAllowedError as exc:
            failure = KeystoneAuthorizationFailureError(str(exc))
            self._record_requests((), failure, tool_name=tool_name, arguments=arguments)
            raise failure from None
        except KeystoneClientError as exc:
            failure = KeystoneResourceReadError(str(exc))
            self._record_requests((), failure, tool_name=tool_name, arguments=arguments)
            raise failure from None

        failure: KeystoneGoalIntegrationError | None = None
        try:
            if result.read_error is not None:
                if result.read_error == "McpResponseEnvelopeUnverified":
                    raise KeystoneCapabilityUnavailableError(
                        "Keystone business-read transport is disabled and its MCP call envelope is unverified."
                    )
                if result.read_error == "SyntheticResponseMissing":
                    raise KeystoneIncompleteDataError(
                        f"{tool_name} synthetic page data is incomplete."
                    )
                if result.read_error == "KeystoneOperationNotAllowedError":
                    raise KeystoneCapabilityUnavailableError(
                        f"{tool_name} is not available through the disabled Keystone read transport."
                    )
                raise KeystoneResourceReadError(
                    f"{tool_name} read failed ({result.read_error})."
                )
            if result.pagination.completeness.status is not CompletenessStatus.COMPLETE:
                reasons = "; ".join(result.pagination.completeness.reasons)
                raise KeystoneIncompleteDataError(
                    f"{tool_name} collection is not complete: {reasons}"
                )
            if result.records is None:
                raise KeystoneIncompleteDataError(
                    f"{tool_name} collection completed without mapped records."
                )
            return [
                self._to_goal_record(tool_name, record, expected_type)
                for record in result.records
            ]
        except KeystoneGoalIntegrationError as exc:
            failure = exc
            raise
        finally:
            self._record_requests(result.requests, failure)

    def find_parties(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
        **extra_filters: Any,
    ) -> list[dict[str, Any]]:
        return self._list(
            "Party.list",
            {**dict(filters or {}), **extra_filters},
            limit,
            self._adapter.list_parties,
            PartyRecord,
        )

    def list_calendars(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
        **extra_filters: Any,
    ) -> list[dict[str, Any]]:
        return self._list(
            "Calendar.list",
            {**dict(filters or {}), **extra_filters},
            limit,
            self._adapter.list_calendars,
            KeystoneCalendarRecord,
        )

    def list_events(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
        **extra_filters: Any,
    ) -> list[dict[str, Any]]:
        combined_filters = {**dict(filters or {}), **extra_filters}
        return self._list(
            "CalendarEvent.list",
            combined_filters,
            limit,
            self._adapter.list_events,
            CalendarEventRef,
        )

    def list_events_page(self, **filters: Any) -> CalendarEventPage:
        limit = filters.pop("limit", 20)
        offset = filters.pop("offset", 0)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")

        records = self.list_events(filters=filters, limit=limit)
        page = CalendarEventPage(
            records=tuple(records[offset : offset + limit]),
            total=len(records),
            limit=limit,
            offset=offset,
        )
        page.validate(requested_limit=limit, requested_offset=offset)
        return page

    def list_event_attendees(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 20,
        **extra_filters: Any,
    ) -> list[dict[str, Any]]:
        return self._list(
            "EventAttendee.list",
            {**dict(filters or {}), **extra_filters},
            limit,
            self._adapter.list_event_attendees,
            EventAttendee,
        )

    def _unavailable(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        detail: str | None = None,
    ) -> NoReturn:
        error = KeystoneCapabilityUnavailableError(
            detail or f"{operation} is unavailable through the approved Keystone integration."
        )
        self.trace.append(
            ToolCallRecord(
                operation,
                dict(arguments),
                succeeded=False,
                error=error.category,
            )
        )
        raise error

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        calendars = self.list_calendars(
            filters={"id": calendar_id},
            limit=1000,
        )
        matches = [calendar for calendar in calendars if calendar.get("id") == calendar_id]
        if len(matches) != 1:
            raise KeystoneIncompleteDataError(
                "Calendar.list did not resolve exactly one owning calendar for timezone fallback."
            )
        if not isinstance(matches[0].get("timezone"), str):
            raise KeystoneIncompleteDataError(
                "The owning calendar has no verified IANA timezone for timestamp fallback."
            )
        return matches[0]

    def get_event(self, event_id: str) -> dict[str, Any]:
        self._unavailable("CalendarEvent.get", {"id": event_id})

    def list_scheduling_preferences(
        self,
        *,
        limit: int = 1000,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return self._list(
            "SchedulingPreferences.list",
            filters or None,
            limit,
            self._adapter.list_scheduling_preferences,
            dict,
        )

    def list_availability_rules(
        self,
        *,
        limit: int = 1000,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return self._list(
            "AvailabilityRule.list",
            filters or None,
            limit,
            self._adapter.list_availability_rules,
            dict,
        )

    def free_busy(self, **arguments: Any) -> dict[str, Any]:
        normalized = dict(arguments)
        for field in ("start", "end"):
            value = normalized.get(field)
            if hasattr(value, "isoformat"):
                normalized[field] = value.isoformat()
        try:
            request, response = self._adapter.read_goal_one_endpoint(
                "endpoint.calendar.free_busy", normalized
            )
        except KeystoneOperationNotAllowedError as exc:
            failure = KeystoneAuthorizationFailureError(str(exc))
            self._record_requests(
                (),
                failure,
                tool_name="endpoint.calendar.free_busy",
                arguments=normalized,
            )
            raise failure from None
        except KeystoneProtocolError as exc:
            failure = KeystoneSchemaValidationError(str(exc))
            self._record_requests(
                (),
                failure,
                tool_name="endpoint.calendar.free_busy",
                arguments=normalized,
            )
            raise failure from None
        except KeystoneClientError as exc:
            failure = KeystoneResourceReadError(
                f"endpoint.calendar.free_busy read failed ({type(exc).__name__})."
            )
            self._record_requests(
                (),
                failure,
                tool_name="endpoint.calendar.free_busy",
                arguments=normalized,
            )
            raise failure from None
        self._record_requests((request,))
        return response

    def _mutation_disabled(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> NoReturn:
        self.trace.append(
            ToolCallRecord(
                operation,
                dict(arguments),
                mutating=True,
                succeeded=False,
                error="disabled",
            )
        )
        raise MutationDisabledError(f"Mutation tool disabled: {operation}")

    def _write(self, operation: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.disposable_mutations_enabled:
            self._mutation_disabled(operation, arguments)
        try:
            if operation == "CalendarEvent.create":
                response = self._adapter._create_event_on_disposable_copy(arguments)
            elif operation == "CalendarEvent.update":
                response = self._adapter._update_event_on_disposable_copy(arguments)
            else:
                self._mutation_disabled(operation, arguments)
        except KeystoneClientError as exc:
            self.trace.append(
                ToolCallRecord(
                    operation,
                    dict(arguments),
                    mutating=True,
                    succeeded=False,
                    error=type(exc).__name__,
                )
            )
            raise CalendarMutationError(
                f"{operation} failed; verify event state before retrying."
            ) from exc
        if not isinstance(response, dict) or not isinstance(response.get("id"), str):
            self.trace.append(
                ToolCallRecord(
                    operation,
                    dict(arguments),
                    mutating=True,
                    succeeded=False,
                    error="unverified_response",
                )
            )
            raise CalendarMutationError(
                f"{operation} returned no verifiable event ID."
            )
        self.trace.append(
            ToolCallRecord(operation, dict(arguments), mutating=True)
        )
        return response

    def create_event(self, **arguments: Any) -> dict[str, Any]:
        if not self.disposable_mutations_enabled:
            self._mutation_disabled("CalendarEvent.create", arguments)
        return self._write("CalendarEvent.create", arguments)

    def update_event(self, **arguments: Any) -> dict[str, Any]:
        if not self.disposable_mutations_enabled:
            self._mutation_disabled("CalendarEvent.update", arguments)
        if set(arguments) - {"id", "start_at", "end_at"}:
            raise ValueError(
                "CalendarEvent.update accepts only ID and scheduled start/end fields."
            )
        if not {"id", "start_at", "end_at"}.issubset(arguments):
            raise ValueError(
                "CalendarEvent.update requires ID, start_at, and end_at."
            )
        return self._write("CalendarEvent.update", arguments)

    def delete_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.delete", {"id": event_id})

    def confirm_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.confirm", {"id": event_id})

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.mark_tentative", {"id": event_id})

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.cancel", {"id": event_id})


def run_keystone_goal(
    task: AgentTask,
    adapter: KeystoneCalendarAdapter,
    *,
    allow_disposable_mutations: bool = False,
) -> AgentResult:
    """Run an existing goal with only offline Keystone adapter data."""

    if task.goal == "calendar.move_after_audit":
        context = dict(task.context)
        context["use_calendar_timezone_fallback"] = True
        task = AgentTask(task.goal, task.instruction, context)
    tools = KeystoneCalendarTool(
        adapter,
        allow_disposable_mutations=allow_disposable_mutations,
    )
    try:
        return CalendarAgent(
            tools,
            allow_disposable_writes=allow_disposable_mutations,
        ).run(task)
    except KeystoneGoalIntegrationError as exc:
        return AgentResult(
            goal=task.goal,
            status=ResultStatus.NEEDS_CLARIFICATION,
            summary="The Keystone adapter cannot provide sufficient verified data for this goal.",
            ambiguities=[f"{exc.category}: {exc}"],
            tool_calls=list(tools.trace),
        )
