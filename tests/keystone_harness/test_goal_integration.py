from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from calendar_agent.models import AgentTask, MutationDisabledError, ResultStatus
import harnesses.suryodaya_harness.hosted_runner as hosted_runner
from harnesses.keystone_harness import (
    DiscoveredTool,
    KeystoneCalendarAdapter,
    KeystoneCapabilityUnavailableError,
    KeystoneClient,
    KeystoneClientError,
    KeystoneConfig,
    KeystoneTransportError,
    run_keystone_goal,
)
from harnesses.suryodaya_harness.hosted_runner import (
    KEYSTONE_AUDIT_TITLE,
    check_keystone_goal_one,
    check_keystone_goal_two,
)
from harnesses.suryodaya_harness.journal import HostedRunJournal
from harnesses.keystone_harness.goal_integration import KeystoneCalendarTool


_RESOURCE_FILTERS: dict[str, dict[str, dict[str, Any]]] = {
    "Party.list": {"job_title": {"type": "string"}},
    "Calendar.list": {
        "id": {"type": "string"},
        "owner_party_id": {"type": "string"},
    },
    "CalendarEvent.list": {
        "search": {"type": "string"},
        "calendar_id": {"type": "string"},
    },
    "EventAttendee.list": {"event_id": {"type": "string"}},
}

_TOOL_NAMES = tuple(_RESOURCE_FILTERS)
_BASE_TIME = datetime(2026, 9, 22, 8, tzinfo=timezone.utc)
_FIND_CONTEXT = {
    "now": _BASE_TIME,
    "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
    "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
}
_AUDIT = {
    "id": "audit-1",
    "title": "Internal quality audit",
    "start_at": "2026-09-22T09:00:00+00:00",
    "end_at": "2026-09-22T10:00:00+00:00",
    "calendar_id": "calendar-1",
    "status": "confirmed",
    "timezone": "UTC",
}


class _DiscoveryResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _DiscoveryOnlySession:
    def __init__(self, tools: tuple[DiscoveredTool, ...]) -> None:
        self.tools = tools
        self.methods: list[str] = []

    def post(self, _url: str, **kwargs: Any) -> _DiscoveryResponse:
        request = kwargs["json"]
        self.methods.append(request["method"])
        if request["method"] != "tools/list":
            raise AssertionError("synthetic integration must not call business tools")
        return _DiscoveryResponse({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                        "outputSchema": tool.output_schema,
                        "annotations": tool.annotations,
                        "_meta": tool.meta,
                    }
                    for tool in self.tools
                ],
                "nextCursor": None,
            },
        })


def _tool(name: str, *, output_schema: dict[str, Any] | None = None) -> DiscoveredTool:
    properties: dict[str, Any] = {
        "limit": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
            "default": 20,
        },
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        **_RESOURCE_FILTERS[name],
    }
    return DiscoveredTool(
        name=name,
        read_only_hint=True,
        description=f"List {name.split('.')[0]} records",
        input_schema={
            "type": "object",
            "properties": properties,
            "required": [],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True, "destructiveHint": False},
        output_schema=output_schema or {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                        },
                        "required": ["id"],
                        "additionalProperties": True,
                    },
                },
                "total": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 0},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["data", "total", "limit", "offset"],
            "additionalProperties": True,
        },
        meta={
            "agentswitch": {
                "permission": "read",
                "discovery": "caller",
                "risk": "READ",
            }
        },
    )


def _page(
    records: list[dict[str, Any]],
    *,
    total: int | str,
    limit: int = 1000,
    offset: int = 0,
) -> dict[str, Any]:
    return {
        "data": records,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def _adapter(
    pages: dict[tuple[str, int], dict[str, Any] | KeystoneClientError],
    *,
    tools: tuple[DiscoveredTool, ...] | None = None,
) -> tuple[KeystoneCalendarAdapter, _DiscoveryOnlySession]:
    session = _DiscoveryOnlySession(tuple(_tool(name) for name in _TOOL_NAMES) if tools is None else tools)
    client = KeystoneClient(
        KeystoneConfig(token="offline-integration-test-token"),
        session=session,  # type: ignore[arg-type]
    )
    client.list_tools()
    return (
        KeystoneCalendarAdapter(client, synthetic_response_pages=pages),
        session,
    )


def _use_distinct_event_query_pages(
    adapter: KeystoneCalendarAdapter,
    monkeypatch: pytest.MonkeyPatch,
    *,
    search_page: dict[str, Any],
    full_page: dict[str, Any],
) -> None:
    original_list_events = adapter.list_events

    def list_events(
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ):
        original_pages = adapter._synthetic_response_pages
        adapter._synthetic_response_pages = (
            {("CalendarEvent.list", 0): search_page}
            if filters is not None and filters.get("search") == "audit"
            else {("CalendarEvent.list", 0): full_page}
        )
        try:
            return original_list_events(filters=filters, limit=limit)
        finally:
            adapter._synthetic_response_pages = original_pages

    monkeypatch.setattr(adapter, "list_events", list_events)


def _party(party_id: str, *, title: str = "Plant Manager") -> dict[str, str]:
    return {
        "id": party_id,
        "name": f"Person {party_id}",
        "email": f"{party_id}@example.test",
        "job_title": title,
    }


def _calendar(
    calendar_id: str = "calendar-1",
    *,
    owner_party_id: str = "party-1",
    timezone_name: str = "UTC",
) -> dict[str, str]:
    return {
        "id": calendar_id,
        "owner_party_id": owner_party_id,
        "timezone": timezone_name,
    }


def _event(
    event_id: str,
    title: str,
    start: str,
    end: str,
    *,
    timezone_name: str = "UTC",
) -> dict[str, str]:
    return {
        "id": event_id,
        "title": title,
        "start_at": start,
        "end_at": end,
        "calendar_id": "calendar-1",
        "status": "confirmed",
        "timezone": timezone_name,
    }


def _run_find(adapter: KeystoneCalendarAdapter):
    return run_keystone_goal(
        AgentTask(
            "calendar.find_30_minutes",
            "Find 30 minutes with the plant head",
            dict(_FIND_CONTEXT),
        ),
        adapter,
    )


def _run_move(adapter: KeystoneCalendarAdapter):
    return run_keystone_goal(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": timedelta(hours=1), "calendar_id": "calendar-1"},
        ),
        adapter,
    )


def _first_page_with_1000_events() -> list[dict[str, str]]:
    return [
        dict(_AUDIT),
        *[
            _event(
                f"work-{index}",
                "Work",
                "2026-09-22T11:00:00+00:00",
                "2026-09-22T12:00:00+00:00",
            )
            for index in range(999)
        ],
    ]


def test_find_goal_resolves_unique_plant_manager_then_reports_missing_capabilities() -> None:
    adapter, session = _adapter({
        ("Party.list", 0): _page([_party("party-1")], total=1),
        ("Calendar.list", 0): _page([_calendar()], total=1),
    })

    result = _run_find(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert len(result.tool_calls) == 4
    assert sum(call.tool == "Party.list" for call in result.tool_calls) == 2
    assert any(call.tool == "Calendar.list" for call in result.tool_calls)
    assert any("SchedulingPreferences.list" in item for item in result.ambiguities)
    assert any("AvailabilityRule.list" in item for item in result.ambiguities)
    assert any("endpoint.calendar.free_busy" in item for item in result.ambiguities)
    assert session.methods == ["tools/list"]
    assert not any(call.mutating for call in result.tool_calls)


def test_find_goal_no_plant_manager_requires_clarification() -> None:
    adapter, _ = _adapter({
        ("Party.list", 0): _page([], total=0),
    })

    result = _run_find(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert any("found 0" in item for item in result.ambiguities)
    assert all(call.tool == "Party.list" for call in result.tool_calls)


def test_find_goal_multiple_plant_managers_require_clarification() -> None:
    adapter, _ = _adapter({
        ("Party.list", 0): _page(
            [_party("party-1"), _party("party-2")],
            total=2,
        ),
    })

    result = _run_find(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert any("found 2" in item for item in result.ambiguities)
    assert all(call.tool == "Party.list" for call in result.tool_calls)


def test_find_goal_missing_job_title_is_a_mapping_failure() -> None:
    incomplete_party = {
        "id": "party-1",
        "name": "Person party-1",
        "email": "party-1@example.test",
    }
    adapter, _ = _adapter({
        ("Party.list", 0): _page([incomplete_party], total=1),
    })

    result = _run_find(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert any("record_mapping_failure" in item for item in result.ambiguities)
    assert any("job_title" in item for item in result.ambiguities)


def test_find_goal_incomplete_calendar_pagination_is_not_an_empty_list() -> None:
    adapter, _ = _adapter({
        ("Party.list", 0): _page([_party("party-1")], total=1),
        ("Calendar.list", 0): _page([_calendar()], total=2),
    })

    result = _run_find(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert any("incomplete_data" in item for item in result.ambiguities)
    assert "Calendar.list" in " ".join(result.ambiguities)


@pytest.mark.parametrize(
    ("record", "category"),
    [
        (
            {"id": "calendar-1", "owner_party_id": "party-1"},
            "incomplete_data",
        ),
        (
            {
                "id": "calendar-1",
                "owner_party_id": "party-1",
                "timezone": "Mars/Phobos",
            },
            "record_mapping_failure",
        ),
    ],
)
def test_find_goal_missing_or_invalid_calendar_timezone_fails_closed(
    record: dict[str, str],
    category: str,
) -> None:
    adapter, _ = _adapter({
        ("Party.list", 0): _page([_party("party-1")], total=1),
        ("Calendar.list", 0): _page([record], total=1),
    })

    result = _run_find(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert any(category in item for item in result.ambiguities)
    assert not any(call.tool == "SchedulingPreferences.list" for call in result.tool_calls)


def test_move_goal_plans_complete_synthetic_events_without_mutation() -> None:
    events = [
        _event(
            "audit-1",
            "Internal quality audit",
            "2026-09-22T09:00:00+00:00",
            "2026-09-22T10:00:00+00:00",
        ),
        _event(
            "after-1",
            "Work",
            "2026-09-22T11:00:00+00:00",
            "2026-09-22T12:00:00+00:00",
        ),
    ]
    adapter, session = _adapter({
        ("CalendarEvent.list", 0): _page(events, total=2),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after-1"]
    assert result.claimed_outcome["moved"] is False
    assert not any(call.mutating for call in result.tool_calls)
    assert session.methods == ["tools/list"]


def test_hosted_keystone_goal_one_stays_unevaluated_without_required_capabilities() -> None:
    adapter, _ = _adapter({
        ("Party.list", 0): _page([_party("party-1")], total=1),
        ("Calendar.list", 0): _page([_calendar()], total=1),
    })

    check = check_keystone_goal_one(adapter, now=_BASE_TIME)

    assert not check.passed
    assert check.evidence.startswith("UNEVALUATED:")
    assert "availability rules" in check.evidence
    assert "mutation_attempts=0" in check.evidence


def test_hosted_keystone_goal_two_requires_confirmation_and_never_mutates() -> None:
    events = [
        _event(
            "audit-1",
            KEYSTONE_AUDIT_TITLE,
            "2026-09-22T09:00:00+00:00",
            "2026-09-22T10:00:00+00:00",
        ),
        _event(
            "other-audit",
            "Supplier audit",
            "2026-09-22T07:00:00+00:00",
            "2026-09-22T08:00:00+00:00",
        ),
        _event(
            "after-1",
            "Production review",
            "2026-09-22T11:00:00+00:00",
            "2026-09-22T12:00:00+00:00",
        ),
    ]
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(events, total=3),
    })

    check = check_keystone_goal_two(adapter)

    assert check.passed
    assert "calendar_resolved=true" in check.evidence
    assert "confirmation_required=true" in check.evidence
    assert "mutation_attempts=0" in check.evidence


def test_hosted_keystone_goal_two_pins_earliest_of_same_titled_audits() -> None:
    events = [
        _event(
            "audit-1",
            KEYSTONE_AUDIT_TITLE,
            "2026-09-22T09:00:00+00:00",
            "2026-09-22T10:00:00+00:00",
        ),
        _event(
            "audit-2",
            KEYSTONE_AUDIT_TITLE,
            "2026-09-22T13:00:00+00:00",
            "2026-09-22T14:00:00+00:00",
        ),
    ]
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(events, total=2),
    })

    check = check_keystone_goal_two(adapter)

    assert "did not resolve exactly one candidate" not in check.evidence


def test_hosted_keystone_goal_two_requires_confirmed_audit() -> None:
    audit = _event(
        "audit-1",
        KEYSTONE_AUDIT_TITLE,
        "2026-09-22T09:00:00+00:00",
        "2026-09-22T10:00:00+00:00",
    )
    audit["status"] = "tentative"
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([audit], total=1),
    })

    check = check_keystone_goal_two(adapter)

    assert not check.passed
    assert "not verified as confirmed" in check.evidence


def test_hosted_keystone_goal_two_journals_backend_reads_before_predicate(
    monkeypatch,
    tmp_path,
) -> None:
    events = [
        _event(
            "audit-1",
            KEYSTONE_AUDIT_TITLE,
            "2026-09-22T09:00:00+00:00",
            "2026-09-22T10:00:00+00:00",
        ),
        _event(
            "after-1",
            "Production review",
            "2026-09-22T11:00:00+00:00",
            "2026-09-22T12:00:00+00:00",
        ),
    ]
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(events, total=2),
    })
    journal = HostedRunJournal(
        tmp_path,
        instance="keystone",
        run_id="before-score",
    )
    task_id = "keystone.goal2_read_only_plan"
    journal.begin(
        task_id=task_id,
        title="Keystone read-only Goal 2 plan",
        goal="calendar.move_after_audit",
    )
    original_verify = hosted_runner.verify_result

    def verify_after_journal(result, *, expected_status):
        record = journal._load(task_id)
        assert record["phase"] == "execution_recorded"
        assert record["observation"]["exact_title_match_count"] == 1
        assert record["agent_result"]["status"] == "planned"
        return original_verify(result, expected_status=expected_status)

    monkeypatch.setattr(hosted_runner, "verify_result", verify_after_journal)

    check = check_keystone_goal_two(adapter, journal=journal)

    assert check.passed
    journal.finish(check)


def test_move_goal_uses_the_matching_calendars_timezone_for_naive_timestamps() -> None:
    audit = {
        **_AUDIT,
        "start_at": "2026-09-22T09:00:00",
        "end_at": "2026-09-22T10:00:00",
        "timezone": None,
    }
    after = {
        **_event(
            "after-1",
            "Work",
            "2026-09-22T11:00:00",
            "2026-09-22T12:00:00",
        ),
        "timezone": None,
    }
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([audit, after], total=2),
        ("Calendar.list", 0): _page(
            [_calendar(timezone_name="America/Los_Angeles")],
            total=1,
        ),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.PLANNED
    move = result.rescheduling_plan[0]
    assert move.original_start_at == "2026-09-22T11:00:00-07:00"
    assert move.proposed_start_at == "2026-09-22T12:00:00-07:00"
    assert move.timezone == "America/Los_Angeles"
    assert move.duration_seconds == 3600
    assert result.audit_reference is not None
    assert result.audit_reference.candidates[0].start_at == "2026-09-22T09:00:00-07:00"
    assert [call.tool for call in result.tool_calls].count("Calendar.list") == 1
    assert not any(call.mutating for call in result.tool_calls)


@pytest.mark.parametrize(
    ("calendar", "expected_category"),
    [
        (_calendar() | {"timezone": None}, "incomplete_data"),
        (_calendar() | {"timezone": "Mars/Phobos"}, "record_mapping_failure"),
    ],
)
def test_move_goal_missing_or_invalid_calendar_fallback_timezone_fails_closed(
    calendar: dict[str, str | None],
    expected_category: str,
) -> None:
    audit = {**_AUDIT, "timezone": None}
    after = {**_event("after-1", "Work", "2026-09-22T11:00:00", "2026-09-22T12:00:00"), "timezone": None}
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([audit, after], total=2),
        ("Calendar.list", 0): _page([calendar], total=1),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any(expected_category in item for item in result.ambiguities)


def test_move_goal_missing_or_ambiguous_audit_requires_clarification() -> None:
    no_audit_adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(
            [_event("work-1", "Work", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")],
            total=1,
        ),
    })
    multiple_audits = [
        _event("audit-1", "Internal audit", "2026-09-22T09:00:00+00:00", "2026-09-22T10:00:00+00:00"),
        _event("audit-2", "Supplier audit", "2026-09-22T09:00:00+00:00", "2026-09-22T10:30:00+00:00"),
    ]
    multiple_adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(multiple_audits, total=2),
    })

    no_audit_result = _run_move(no_audit_adapter)
    multiple_result = _run_move(multiple_adapter)

    assert no_audit_result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not no_audit_result.rescheduling_plan
    assert multiple_result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not multiple_result.rescheduling_plan


def test_move_goal_missing_required_event_fields_is_not_dropped() -> None:
    missing_start = {
        "id": "after-1",
        "title": "Work",
        "end_at": "2026-09-22T12:00:00+00:00",
        "calendar_id": "calendar-1",
        "status": "confirmed",
        "timezone": "UTC",
    }
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([_AUDIT, missing_start], total=2),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("record_mapping_failure" in item for item in result.ambiguities)
    assert any("start_at" in item for item in result.ambiguities)


@pytest.mark.parametrize(
    ("start", "end", "zone"),
    [
        ("not-a-date", "2026-09-22T12:00:00+00:00", "UTC"),
        (
            "2026-09-22T11:00:00+00:00",
            "2026-09-22T12:00:00+00:00",
            "Mars/Phobos",
        ),
    ],
)
def test_move_goal_malformed_naive_or_invalid_timezone_fails_closed(
    start: str,
    end: str,
    zone: str,
) -> None:
    events = [
        _AUDIT,
        _event("after-1", "Work", start, end, timezone_name=zone),
    ]
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(events, total=2),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_move_goal_naive_timestamps_use_valid_event_timezone() -> None:
    audit = {
        **_AUDIT,
        "start_at": "2026-09-22T09:00:00",
        "end_at": "2026-09-22T10:00:00",
    }
    after = _event("after-1", "Work", "2026-09-22T11:00:00", "2026-09-22T12:00:00")
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([audit, after], total=2),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.PLANNED
    assert result.rescheduling_plan[0].original_start_at == "2026-09-22T11:00:00+00:00"
    assert result.rescheduling_plan[0].duration_seconds == 3600
    assert not any(call.mutating for call in result.tool_calls)


def test_move_goal_duplicate_ids_across_audit_boundary_fails_closed() -> None:
    events = [
        _AUDIT,
        _event("x", "Earlier", "2026-09-22T09:00:00+00:00", "2026-09-22T09:30:00+00:00"),
        _event("x", "Later", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ]
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(events, total=3),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("incomplete_data" in item for item in result.ambiguities)


def test_move_goal_incomplete_pagination_does_not_produce_a_plan() -> None:
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([_AUDIT], total=2),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("incomplete_data" in item for item in result.ambiguities)


def test_move_goal_later_page_schema_failure_preserves_request_trace() -> None:
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(
            _first_page_with_1000_events(),
            total=1001,
            limit=1000,
        ),
        ("CalendarEvent.list", 1000): _page(
            [_event("work-final", "Work", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")],
            total="1001",
            limit=1000,
            offset=1000,
        ),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("schema_validation_failure" in item for item in result.ambiguities)
    assert [call.arguments["offset"] for call in result.tool_calls] == [0, 1000]
    assert [call.succeeded for call in result.tool_calls] == [True, False]
    assert [call.error for call in result.tool_calls] == [
        None,
        "schema_validation_failure",
    ]


def test_move_goal_later_page_mapping_failure_preserves_request_trace() -> None:
    malformed_event = {
        "id": "work-final",
        "start_at": "2026-09-22T11:00:00+00:00",
        "end_at": "2026-09-22T12:00:00+00:00",
        "calendar_id": "calendar-1",
        "status": "confirmed",
        "timezone": "UTC",
    }
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(
            _first_page_with_1000_events(),
            total=1001,
            limit=1000,
        ),
        ("CalendarEvent.list", 1000): _page(
            [malformed_event],
            total=1001,
            limit=1000,
            offset=1000,
        ),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("record_mapping_failure" in item for item in result.ambiguities)
    assert [call.arguments["offset"] for call in result.tool_calls] == [0, 1000]
    assert [call.succeeded for call in result.tool_calls] == [True, False]
    assert [call.error for call in result.tool_calls] == [
        None,
        "record_mapping_failure",
    ]


def test_move_goal_schema_validation_error_is_explicit() -> None:
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page([_AUDIT], total="1"),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("schema_validation_failure" in item for item in result.ambiguities)


def test_adapter_authorization_rejection_is_classified_and_traced() -> None:
    rejected_tool = replace(
        _tool("CalendarEvent.list"),
        meta={
            "agentswitch": {
                "permission": "write",
                "discovery": "caller",
                "risk": "READ",
            }
        },
    )
    tools = tuple(
        rejected_tool if name == "CalendarEvent.list" else _tool(name)
        for name in _TOOL_NAMES
    )
    adapter, session = _adapter({}, tools=tools)

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("authorization_failure" in item for item in result.ambiguities)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "CalendarEvent.list"
    assert result.tool_calls[0].succeeded is False
    assert result.tool_calls[0].error == "authorization_failure"
    assert session.methods == ["tools/list"]


def test_adapter_resource_read_failure_is_classified_and_traced() -> None:
    adapter, session = _adapter({
        ("CalendarEvent.list", 0): KeystoneTransportError(
            "synthetic offline read failure"
        ),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("resource_read_failure" in item for item in result.ambiguities)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "CalendarEvent.list"
    assert result.tool_calls[0].succeeded is False
    assert result.tool_calls[0].error == "resource_read_failure"
    assert session.methods == ["tools/list"]


def test_move_goal_inconsistent_audit_search_and_full_listing_requires_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_listing_audit = {
        **_AUDIT,
        "start_at": "2026-09-22T10:00:00+00:00",
        "end_at": "2026-09-22T11:00:00+00:00",
    }
    adapter, _ = _adapter({})
    _use_distinct_event_query_pages(
        adapter,
        monkeypatch,
        search_page=_page([_AUDIT], total=1),
        full_page=_page(
            [
                full_listing_audit,
                _event(
                    "work-1",
                    "Work",
                    "2026-09-22T10:30:00+00:00",
                    "2026-09-22T11:30:00+00:00",
                ),
            ],
            total=2,
        ),
    )

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "interval changed between reads" in result.summary


def test_move_goal_missing_event_and_calendar_timezone_fails_closed() -> None:
    event_without_timezone = _event(
        "work-1",
        "Work",
        "2026-09-22T11:00:00+00:00",
        "2026-09-22T12:00:00+00:00",
    )
    del event_without_timezone["timezone"]
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(
            [_AUDIT, event_without_timezone],
            total=2,
        ),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("incomplete_data" in item for item in result.ambiguities)
    assert any("Calendar.list" in item for item in result.ambiguities)


def test_move_goal_nonpositive_event_interval_fails_closed() -> None:
    adapter, _ = _adapter({
        ("CalendarEvent.list", 0): _page(
            [
                _AUDIT,
                _event(
                    "work-1",
                    "Work",
                    "2026-09-22T11:00:00+00:00",
                    "2026-09-22T10:30:00+00:00",
                ),
            ],
            total=2,
        ),
    })

    result = _run_move(adapter)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "malformed" in result.summary


def test_attendee_mapping_preserves_record_and_event_identity() -> None:
    attendee = {
        "id": "attendee-1",
        "event_id": "event-1",
        "email": "person@example.test",
        "party_id": "party-1",
        "role": "req",
        "status": "accepted",
        "is_organizer": False,
    }
    adapter, session = _adapter({
        ("EventAttendee.list", 0): _page([attendee], total=1),
    })
    tools = KeystoneCalendarTool(adapter)

    result = tools.list_event_attendees(event_id="event-1", limit=1000)

    assert result == [{
        "id": "attendee-1",
        "event_id": "event-1",
        "party_id": "party-1",
        "email": "person@example.test",
        "role": "required",
        "status": "accepted",
    }]
    assert session.methods == ["tools/list"]


def test_free_busy_and_mutations_are_unavailable_without_business_calls() -> None:
    adapter, session = _adapter({})
    tools = KeystoneCalendarTool(adapter)

    with pytest.raises(KeystoneCapabilityUnavailableError, match="endpoint.calendar.free_busy"):
        tools.free_busy(
            start=datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
            end=datetime(2026, 9, 22, 10, tzinfo=timezone.utc),
        )
    with pytest.raises(MutationDisabledError):
        tools.update_event(id="event-1")

    assert session.methods == ["tools/list"]
    assert not any(call.tool == "tools/call" for call in tools.trace)
    assert all(not call.succeeded for call in tools.trace)
    assert any(call.mutating for call in tools.trace)
