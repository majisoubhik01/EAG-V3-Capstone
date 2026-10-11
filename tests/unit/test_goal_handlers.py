from copy import deepcopy
from datetime import datetime, timezone

from calendar_agent.agent import CalendarAgent
from calendar_agent.models import AgentTask, PartyPage, PartyRecord, ResultStatus, ToolCallRecord
from harness.assertions import verify_result, verify_supported_slot

from .fakes import FakeCalendarTools


class PagedPartyTools(FakeCalendarTools):
    def __init__(self, pages, *, fail_on: tuple[str, int] | None = None) -> None:
        super().__init__(
            calendars=[{"id": "calendar-1", "owner_party_id": "plant-1"}],
            availability_rules=[{
                "party_id": "plant-1",
                "timezone": "UTC",
                "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            }],
            scheduling_preferences=[{"default_meeting_duration": 30, "min_notice_hours": 0}],
            free_busy_result=_visible_free_busy([]),
        )
        self.pages = pages
        self.fail_on = fail_on

    def find_parties_page(self, *, page: int, page_size: int, job_title: str) -> PartyPage:
        arguments = {"page": page, "page_size": page_size, "job_title": job_title}
        if self.fail_on == (job_title, page):
            self.trace.append(ToolCallRecord("Party.list", arguments, succeeded=False, error="RuntimeError"))
            raise RuntimeError("fake page retrieval failure")
        self.trace.append(ToolCallRecord("Party.list", arguments))
        return self.pages[(job_title, page)]


def _party_page(
    records: tuple[PartyRecord, ...],
    *,
    page: int = 1,
    total_count: int,
) -> PartyPage:
    return PartyPage(
        records=records,
        page=page,
        page_size=100,
        total_count=total_count,
        has_more=page * 100 < total_count,
    )


def test_plant_manager_ambiguity_is_explicit() -> None:
    tools = FakeCalendarTools(
        parties=[
            {"id": "one", "name": "A", "email": "a@example.test", "job_title": "Plant Manager"},
            {"id": "two", "name": "B", "email": "b@example.test", "job_title": "Plant Manager"},
        ]
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "exactly one" in " ".join(result.ambiguities)
    assert [call.tool for call in result.tool_calls] == ["Party.list", "Party.list"]
    assert not any(call.mutating for call in result.tool_calls)


def test_multiple_audits_are_not_arbitrarily_selected() -> None:
    tools = FakeCalendarTools(
        events=[
            {"id": "audit-1", "title": "Internal quality audit", "start_at": "2026-09-19T09:30", "end_at": "2026-09-19T12:30"},
            {"id": "audit-2", "title": "Supplier quality audit", "start_at": "2026-09-28T10:00", "end_at": "2026-09-28T11:00"},
        ]
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert result.audit_reference is not None
    assert len(result.audit_reference.candidates) == 2
    assert not any(call.mutating for call in result.tool_calls)


def _valid_context() -> dict[str, datetime]:
    return {
        "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }


def _tools_for_party(
    *,
    free_busy_result: dict[str, object],
    rules: list[dict[str, object]] | None = None,
    parties: list[dict[str, object]] | None = None,
    calendars: list[dict[str, object]] | None = None,
    min_notice_hours: float = 0,
) -> FakeCalendarTools:
    actual_parties = parties if parties is not None else [{
        "id": "plant-1",
        "name": "Plant Manager",
        "email": "plant@example.test",
        "job_title": "Plant Manager",
    }]
    party = actual_parties[0] if actual_parties else None
    return FakeCalendarTools(
        parties=actual_parties,
        calendars=calendars if calendars is not None else ([{"id": "calendar-1", "owner_party_id": party["id"]}] if party else []),
        availability_rules=(rules if rules is not None else ([{
            "id": "rule-1",
            "party_id": party["id"],
            "timezone": "UTC",
            "weekly_hours": [{"day": "tuesday", "start": "09:00", "end": "17:00", "enabled": True}],
            "date_overrides": [],
        }] if party else [])),
        scheduling_preferences=[{"default_meeting_duration": 30, "min_notice_hours": min_notice_hours}],
        free_busy_result=free_busy_result,
    )


def _visible_free_busy(busy: list[dict[str, str]]) -> dict[str, object]:
    return {"result": {"subjects": [{
        "subject": "plant@example.test",
        "visibility": "free_busy",
        "busy": busy,
        "calendars": [{"access_level": "free_busy"}],
    }]}}


def test_unique_plant_head_with_verified_free_slot_is_planned() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    slot_verification = verify_supported_slot(
        result,
        expected_start="2026-09-22T09:00:00+00:00",
        expected_end="2026-09-22T09:30:00+00:00",
        expected_calendar_id="calendar-1",
    )
    assert slot_verification.passed
    assert verify_result(result, expected_status=ResultStatus.PLANNED).passed
    assert not any(call.mutating for call in result.tool_calls)


def test_zero_minimum_notice_preserves_planning_behavior() -> None:
    result = CalendarAgent(_tools_for_party(
        free_busy_result=_visible_free_busy([]),
        min_notice_hours=0,
    )).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.PLANNED
    assert result.candidate_slots[0].start_at == "2026-09-22T09:00:00+00:00"


def test_negative_minimum_notice_fails_closed_without_candidate_slots() -> None:
    for min_notice_hours in (-2, -1e-12):
        tools = _tools_for_party(
            free_busy_result=_visible_free_busy([]),
            min_notice_hours=min_notice_hours,
        )
        context = {
            "now": datetime(2026, 9, 22, 12, tzinfo=timezone.utc),
            "start": datetime(2026, 9, 22, 9, tzinfo=timezone.utc),
            "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
        }

        result = CalendarAgent(tools).run(
            AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", context)
        )

        assert result.status is ResultStatus.NEEDS_CLARIFICATION, min_notice_hours
        assert not result.candidate_slots, min_notice_hours
        assert "non-negative" in " ".join(result.ambiguities), min_notice_hours


def test_positive_minimum_notice_preserves_existing_behavior() -> None:
    result = CalendarAgent(_tools_for_party(
        free_busy_result=_visible_free_busy([]),
        min_notice_hours=1,
    )).run(
        AgentTask(
            "calendar.find_30_minutes",
            "Find 30 minutes with the plant head",
            {
                "now": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
                "start": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
                "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
            },
        )
    )

    assert result.status is ResultStatus.PLANNED
    assert result.candidate_slots[0].start_at == "2026-09-22T09:00:00+00:00"


def test_malformed_minimum_notice_values_fail_closed() -> None:
    for invalid_value in (None, "not numeric", float("inf"), True):
        result = CalendarAgent(_tools_for_party(
            free_busy_result=_visible_free_busy([]),
            min_notice_hours=invalid_value,
        )).run(
            AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
        )

        assert result.status is ResultStatus.NEEDS_CLARIFICATION, invalid_value
        assert not result.candidate_slots, invalid_value


def test_paged_party_resolution_fails_closed_when_a_later_page_fails() -> None:
    first_page_records = tuple(
        PartyRecord(f"party-{index}", f"Person {index}", f"person{index}@example.test", "Plant Manager")
        for index in range(100)
    )
    tools = PagedPartyTools(
        {("Plant Manager", 1): _party_page(first_page_records, total_count=101)},
        fail_on=("Plant Manager", 2),
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "could not be retrieved" in " ".join(result.ambiguities)
    assert [call.arguments["page"] for call in tools.trace] == [1, 2]
    assert not any(call.tool == "Calendar.list" for call in tools.trace)


def test_paged_party_resolution_fails_when_page_limit_still_has_more() -> None:
    pages = {}
    for page_number in range(1, 101):
        records = tuple(
            PartyRecord(
                f"party-{page_number}-{index}",
                f"Person {page_number}-{index}",
                f"person-{page_number}-{index}@example.test",
                "Plant Manager",
            )
            for index in range(100)
        )
        pages[("Plant Manager", page_number)] = _party_page(
            records,
            page=page_number,
            total_count=10001,
        )
    tools = PagedPartyTools(pages)

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "page limit" in " ".join(result.ambiguities)
    assert len(tools.trace) == 100


def test_legacy_party_result_at_limit_is_not_treated_as_complete() -> None:
    parties = [
        {
            "id": f"party-{index}",
            "email": f"person-{index}@example.test",
            "job_title": "Plant Manager",
        }
        for index in range(1000)
    ]
    tools = _tools_for_party(parties=parties, free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "may be incomplete" in " ".join(result.ambiguities)
    assert [call.tool for call in result.tool_calls] == ["Party.list"]


def test_conflicting_duplicate_party_ids_require_clarification() -> None:
    manager = PartyRecord("plant-1", "Plant Manager", "manager@example.test", "Plant Manager")
    head = PartyRecord("plant-1", "Plant Head", "head@example.test", "Plant Head")
    tools = PagedPartyTools({
        ("Plant Manager", 1): _party_page((manager,), total_count=1),
        ("Plant Head", 1): _party_page((head,), total_count=1),
    })

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "conflicting identity records" in " ".join(result.ambiguities)


def test_complete_paged_party_resolution_preserves_unique_candidate() -> None:
    manager = PartyRecord("plant-1", "Plant Manager", "plant@example.test", "Plant Manager")
    tools = PagedPartyTools({
        ("Plant Manager", 1): _party_page((manager,), total_count=1),
        ("Plant Head", 1): _party_page((), total_count=0),
    })

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.PLANNED
    assert result.candidate_slots[0].party_id == "plant-1"
    assert [call.arguments["job_title"] for call in tools.trace[:2]] == ["Plant Manager", "Plant Head"]


def test_successful_find_planning_does_not_mutate_source_state() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([]))
    before = deepcopy({
        "parties": tools.parties,
        "calendars": tools.calendars,
        "availability_rules": tools.availability_rules,
        "scheduling_preferences": tools.scheduling_preferences,
        "free_busy_result": tools.free_busy_result,
    })

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    after = {
        "parties": tools.parties,
        "calendars": tools.calendars,
        "availability_rules": tools.availability_rules,
        "scheduling_preferences": tools.scheduling_preferences,
        "free_busy_result": tools.free_busy_result,
    }
    assert result.status is ResultStatus.PLANNED
    assert result.status is not ResultStatus.COMPLETED
    assert after == before
    assert not any(call.mutating for call in result.tool_calls)


def test_minimum_notice_with_seconds_rounds_up_without_crossing_boundary() -> None:
    now = datetime(2026, 9, 22, 8, 0, 1, tzinfo=timezone.utc)
    minimum_notice_boundary = datetime(2026, 9, 22, 9, 0, 1, tzinfo=timezone.utc)
    context = {
        "now": now,
        "start": datetime(2026, 9, 22, 8, tzinfo=timezone.utc),
        "end": datetime(2026, 9, 22, 17, tzinfo=timezone.utc),
    }
    tools = _tools_for_party(
        free_busy_result=_visible_free_busy([]),
        min_notice_hours=1,
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", context)
    )

    verification = verify_supported_slot(
        result,
        expected_start="2026-09-22T09:30:00+00:00",
        expected_end="2026-09-22T10:00:00+00:00",
        minimum_notice_boundary=minimum_notice_boundary,
        expected_calendar_id="calendar-1",
    )
    assert verification.passed


def test_multiple_owned_calendars_require_clarification() -> None:
    tools = _tools_for_party(
        free_busy_result=_visible_free_busy([]),
        calendars=[
            {"id": "calendar-1", "owner_party_id": "plant-1"},
            {"id": "calendar-2", "owner_party_id": "plant-1"},
        ],
    )

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "Multiple" in result.summary
    assert not any(call.mutating for call in result.tool_calls)


def test_not_shared_free_busy_is_not_confirmed_free() -> None:
    tools = _tools_for_party(free_busy_result={"result": {"subjects": [{
        "subject": "plant@example.test",
        "visibility": "unknown",
        "visibility_code": "freebusy.notShared",
        "busy": [],
        "calendars": [{"access_level": "none", "reason_code": "freebusy.notShared"}],
    }]}})

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots
    assert "not confirmed" in " ".join(result.ambiguities)
    assert not any(call.mutating for call in result.tool_calls)


def test_no_plant_head_is_clarification() -> None:
    tools = _tools_for_party(parties=[], free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


def test_multiple_plant_heads_are_clarification() -> None:
    parties = [
        {"id": "plant-1", "name": "One", "email": "one@example.test", "job_title": "Plant Manager"},
        {"id": "plant-2", "name": "Two", "email": "two@example.test", "job_title": "Plant Manager"},
    ]
    tools = _tools_for_party(parties=parties, free_busy_result=_visible_free_busy([]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head")
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "found 2" in " ".join(result.ambiguities)


def test_missing_valid_availability_is_clarification() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([]), rules=[])

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert "availability rule" in result.summary


def test_only_busy_intervals_produce_no_slot() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([{
        "start": "2026-09-22T09:00:00+00:00",
        "end": "2026-09-22T17:00:00+00:00",
    }]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.candidate_slots


def test_valid_free_slot_after_busy_interval_is_independently_supported() -> None:
    tools = _tools_for_party(free_busy_result=_visible_free_busy([{
        "start": "2026-09-22T09:00:00+00:00",
        "end": "2026-09-22T09:30:00+00:00",
    }]))

    result = CalendarAgent(tools).run(
        AgentTask("calendar.find_30_minutes", "Find 30 minutes with the plant head", _valid_context())
    )

    slot_verification = verify_supported_slot(
        result,
        expected_start="2026-09-22T09:30:00+00:00",
        expected_end="2026-09-22T10:00:00+00:00",
    )
    assert slot_verification.passed