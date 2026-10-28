from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import harnesses.suryodaya_harness.hosted_runner as hosted_runner
from calendar_agent import AgentSwitchError
from calendar_agent.models import AgentResult, ResultStatus, ToolCallRecord
from harnesses.suryodaya_harness.hosted_runner import (
    HostedConfigurationError,
    HostedCheck,
    check_backend_rescheduling_state,
    check_suryodaya_goal_one,
    check_suryodaya_goal_two,
    load_hosted_environment,
    run_hosted,
    write_hosted_results,
)
from harnesses.suryodaya_harness.journal import HostedRunJournal
from harnesses.suryodaya_harness.results import parse_results
from tests.unit.fakes import FakeCalendarTools


NOW = datetime(2026, 10, 20, 8, tzinfo=timezone.utc)


def complete_goal_one_tools() -> FakeCalendarTools:
    return FakeCalendarTools(
        parties=[{
            "id": "party-1",
            "email": "plant@example.test",
            "job_title": "Plant Manager",
        }],
        calendars=[{"id": "calendar-1", "owner_party_id": "party-1"}],
        availability_rules=[{
            "id": "rule-1",
            "party_id": "party-1",
            "timezone": "UTC",
            "weekly_hours": [{
                "day": "tuesday",
                "start": "09:00",
                "end": "17:00",
                "enabled": True,
            }],
            "date_overrides": [],
        }],
        scheduling_preferences=[{
            "default_meeting_duration": 30,
            "min_notice_hours": 0,
        }],
        free_busy_result={
            "result": {
                "ok": True,
                "window": {
                    "start": NOW.isoformat(),
                    "end": (NOW + timedelta(days=14)).isoformat(),
                },
                "subjects": [{
                    "subject": "plant@example.test",
                    "subject_kind": "email",
                    "visibility": "visible",
                    "visibility_code": "",
                    "busy": [],
                    "calendars": [{
                        "calendar_id": "calendar-1",
                        "calendar_name": "My Calendar",
                        "access_level": "free_busy",
                        "actual_access_level": "free_busy",
                        "access_source": "grant",
                    }],
                    "message": "",
                }],
            },
        },
    )


def complete_goal_two_tools() -> FakeCalendarTools:
    audit = {
        "id": "audit-uuid",
        "number": "EVT-2026-00007",
        "title": "Internal quality audit — press shop and welding",
        "start_at": "2026-09-19T09:30:00+05:30",
        "end_at": "2026-09-19T12:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "calendar_id": "calendar-1",
        "status": "confirmed",
    }
    after = {
        "id": "work-uuid",
        "number": "EVT-2026-00008",
        "title": "Production review",
        "start_at": "2026-09-19T13:30:00+05:30",
        "end_at": "2026-09-19T14:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "calendar_id": "calendar-1",
        "status": "confirmed",
    }
    return FakeCalendarTools(
        events=[audit, after],
        calendars=[{"id": "calendar-1", "timezone": "Asia/Kolkata"}],
    )


def test_suryodaya_candidate_passes_harness_predicate_when_structurally_valid() -> None:
    tools = complete_goal_one_tools()

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert result.passed
    assert result.official_record()["passed"] is True
    assert "HARNESS_PREDICATE:" in result.evidence
    assert "30-minute candidate" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_suryodaya_candidate_fails_if_fresh_free_busy_reports_overlap() -> None:
    tools = complete_goal_one_tools()
    original_free_busy = tools.free_busy
    calls = 0

    def free_busy(**arguments):
        nonlocal calls
        calls += 1
        response = original_free_busy(**arguments)
        if calls == 2:
            response["result"]["subjects"][0]["busy"] = [{
                "calendar_id": "calendar-1",
                "start": "2026-10-20T09:00:00+00:00",
                "end": "2026-10-20T09:30:00+00:00",
            }]
        return response

    tools.free_busy = free_busy

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert "Candidate overlaps a fresh busy interval." in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_suryodaya_unresolved_party_is_non_passing_and_read_only() -> None:
    tools = FakeCalendarTools()

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert result.official_record()["passed"] is False
    assert "planner status=needs_clarification" in result.evidence
    assert "reason_code=participant_identity_ambiguous" in result.evidence
    assert "mutation_attempts=0" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_goal_two_plan_requires_confirmation_and_performs_no_mutation() -> None:
    tools = complete_goal_two_tools()

    check = check_suryodaya_goal_two(tools)

    assert check.passed
    assert check.evidence.startswith("HARNESS_PREDICATE:")
    assert check.task_id == "suryodaya.goal2_read_only_plan"
    assert "audit=EVT-2026-00007" in check.evidence
    assert "calendar=calendar-1" in check.evidence
    assert "confirmation_required=true" in check.evidence
    assert "events_moved=false" in check.evidence
    assert "confirm_or_change_shift=true" in check.evidence
    assert "mutation_attempts=0" in check.evidence
    assert "work-uuid" in check.evidence
    assert all(not call.mutating for call in tools.trace)


def test_goal_two_finds_conflict_free_alternative_and_waits_for_confirmation() -> None:
    tools = complete_goal_two_tools()
    tools.events.append({
        "id": "stationary-uuid",
        "number": "EVT-2026-00006",
        "title": "Existing hold",
        "start_at": "2026-09-19T11:00:00+05:30",
        "end_at": "2026-09-19T14:15:00+05:30",
        "timezone": "Asia/Kolkata",
        "calendar_id": "calendar-1",
        "status": "confirmed",
    })

    check = check_suryodaya_goal_two(tools)

    assert check.passed
    assert "requested_shift=30m;proposed_shift=60m" in check.evidence
    assert "confirmation_required=true" in check.evidence
    assert "events_moved=false" in check.evidence
    assert all(not call.mutating for call in tools.trace)


def test_goal_two_asks_when_no_alternative_is_conflict_free_within_bound() -> None:
    tools = complete_goal_two_tools()
    tools.events.append({
        "id": "stationary-uuid",
        "number": "EVT-2026-00006",
        "title": "Existing long hold",
        "start_at": "2026-09-19T11:00:00+05:30",
        "end_at": "2026-09-19T23:00:00+05:30",
        "timezone": "Asia/Kolkata",
        "calendar_id": "calendar-1",
        "status": "confirmed",
    })

    check = check_suryodaya_goal_two(tools)

    assert not check.passed
    assert "no conflict-free alternative was found within the eight-hour limit" in check.evidence
    assert all(not call.mutating for call in tools.trace)


def test_goal_two_uses_owning_calendar_timezone_for_naive_times() -> None:
    tools = complete_goal_two_tools()
    for event in tools.events:
        event["start_at"] = str(event["start_at"]).split("+", 1)[0]
        event["end_at"] = str(event["end_at"]).split("+", 1)[0]
        event["timezone"] = None

    check = check_suryodaya_goal_two(tools)

    assert check.passed
    assert "confirmation_required=true" in check.evidence
    assert "2026-09-19T14:00:00+05:30" in check.evidence
    assert "mutation_attempts=0" in check.evidence
    assert any(call.tool == "Calendar.get" for call in tools.trace)
    assert all(not call.mutating for call in tools.trace)


def test_goal_two_fails_closed_without_event_or_calendar_timezone() -> None:
    tools = complete_goal_two_tools()
    tools.calendars[0].pop("timezone")
    for event in tools.events:
        event["start_at"] = str(event["start_at"]).split("+", 1)[0]
        event["end_at"] = str(event["end_at"]).split("+", 1)[0]
        event["timezone"] = None

    check = check_suryodaya_goal_two(tools)

    assert not check.passed
    assert "did not produce a plan" in check.evidence
    assert any(call.tool == "Calendar.get" for call in tools.trace)
    assert all(not call.mutating for call in tools.trace)


def test_agent_switch_read_failure_remains_unevaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = complete_goal_one_tools()

    def unavailable(self, task):
        raise AgentSwitchError("synthetic read failure")

    monkeypatch.setattr(hosted_runner.CalendarAgent, "run", unavailable)

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert result.evidence.startswith("UNEVALUATED:")
    assert "AgentSwitchError" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_hosted_environment_is_bound_to_selected_instance() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://agentswitch.theschoolofai.in",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.instance == "suryodaya"
    assert config.base_url == "https://agentswitch.theschoolofai.in"
    assert "secret-token" not in repr(config)


def test_hosted_environment_uses_platform_injected_secure_instance_url() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test:8443/tenant/suryodaya",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.base_url == "https://runner.example.test:8443/tenant/suryodaya"


def test_hosted_environment_accepts_keystone_and_internal_http_url() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "Keystone",
        "AGENTSWITCH_BASE_URL": "http://runner.internal:8000/tenant/keystone",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.instance == "keystone"
    assert config.base_url == "http://runner.internal:8000/tenant/keystone"


def test_hosted_environment_accepts_platform_injected_http_instance_url() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "http://runner.example.test:8000/tenant/suryodaya",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.base_url == "http://runner.example.test:8000/tenant/suryodaya"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("AGENTSWITCH_BASE_URL", "", "must be set"),
        ("AGENTSWITCH_BASE_URL", "ftp://runner.example.test", "HTTP\\(S\\)"),
        ("AGENTSWITCH_BASE_URL", "https://user:password@runner.example.test", "credential-free"),
        ("AGENTSWITCH_BASE_URL", "https://runner.example.test/api/mcp", "instance base URL"),
        ("AGENTSWITCH_BASE_URL", "https://runner.example.test?tenant=suryodaya", "query or fragment"),
        ("AGENTSWITCH_INSTANCE", "unsupported", "Suryodaya or Keystone"),
    ],
)
def test_hosted_environment_rejects_invalid_configuration(field, value, message) -> None:
    values = {
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test",
        "AGENTSWITCH_TOKEN": "secret-token",
    }
    values[field] = value

    with pytest.raises(HostedConfigurationError, match=message) as error:
        load_hosted_environment(values)

    assert "secret-token" not in str(error.value)


def test_hosted_environment_requires_each_runner_value() -> None:
    for missing in ("AGENTSWITCH_INSTANCE", "AGENTSWITCH_BASE_URL", "AGENTSWITCH_TOKEN"):
        values = {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "https://runner.example.test",
            "AGENTSWITCH_TOKEN": "secret-token",
        }
        del values[missing]
        with pytest.raises(HostedConfigurationError, match=missing):
            load_hosted_environment(values)


def test_hosted_environment_accepts_credential_free_https_instance_base() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test/tenant/suryodaya/",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.base_url == "https://runner.example.test/tenant/suryodaya"


def test_hosted_configuration_error_is_unevaluated_and_secret_safe(tmp_path: Path) -> None:
    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "ftp://runner.example.test",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=tmp_path / "results.json",
    )

    serialized = output_path.read_text(encoding="utf-8")
    document = parse_results(serialized)
    assert checks[0].passed is False
    assert document["tasks"][0]["passed"] is False
    assert document["tasks"][0]["evidence"].startswith("UNEVALUATED:")
    assert "secret-token" not in serialized


def test_unexpected_hosted_initialization_error_is_not_masked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_initialization(config):
        raise RuntimeError("synthetic programming failure")

    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", fail_initialization)
    output_path = tmp_path / "results.json"

    with pytest.raises(RuntimeError, match="synthetic programming failure"):
        run_hosted(
            {
                "AGENTSWITCH_INSTANCE": "suryodaya",
                "AGENTSWITCH_BASE_URL": "https://runner.example.test:8443/tenant/suryodaya",
                "AGENTSWITCH_TOKEN": "synthetic-token",
            },
            output_path=output_path,
        )

    assert not output_path.exists()


def test_unexpected_planner_exception_is_not_masked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tools = complete_goal_one_tools()

    def fail_planning(self, task):
        raise RuntimeError("secret-token must not be serialized")

    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", lambda config: object())
    monkeypatch.setattr(hosted_runner, "AgentSwitchCalendarTools", lambda client: tools)
    monkeypatch.setattr(hosted_runner.CalendarAgent, "run", fail_planning)
    output_path = tmp_path / "results.json"

    with pytest.raises(RuntimeError, match="secret-token must not be serialized"):
        run_hosted(
            {
                "AGENTSWITCH_INSTANCE": "suryodaya",
                "AGENTSWITCH_BASE_URL": "https://runner.example.test",
                "AGENTSWITCH_TOKEN": "synthetic-token",
            },
            output_path=output_path,
            now=NOW,
        )

    assert not output_path.exists()


def test_suryodaya_candidate_serializes_as_harness_predicate_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tools = complete_goal_one_tools()
    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", lambda config: object())
    monkeypatch.setattr(hosted_runner, "AgentSwitchCalendarTools", lambda client: tools)

    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "https://runner.example.test",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=tmp_path / "results.json",
        now=NOW,
    )

    document = parse_results(output_path.read_text(encoding="utf-8"))
    task = document["tasks"][0]
    assert checks[0].passed is True
    assert task["passed"] is True
    assert task["score"] == 1.0
    assert task["evidence"].startswith("HARNESS_PREDICATE:")
    assert "official_evaluator=not_invoked" in document["summary"]


def test_hosted_toml_selects_both_supported_instances() -> None:
    config = Path("agentswitch-harness.toml").read_text(encoding="utf-8")

    assert 'instances = ["suryodaya", "keystone"]' in config


def test_tool_trace_summary_does_not_include_party_data() -> None:
    from harnesses.suryodaya_harness.hosted_runner import _read_trace_summary

    trace = [ToolCallRecord("Party.list", {"job_title": "Plant Manager"})]

    assert _read_trace_summary(trace) == "read_calls=1;failed_reads=0;mutation_attempts=0"


def test_goal_one_reason_code_distinguishes_calendar_scope_from_party_identity() -> None:
    missing_calendar = AgentResult(
        goal="calendar.find_30_minutes",
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary="No deterministically owned calendar was found for the plant head.",
    )
    ambiguous_party = AgentResult(
        goal="calendar.find_30_minutes",
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary="The plant head cannot be deterministically identified from Party data.",
    )

    assert hosted_runner._goal_one_reason_code(missing_calendar) == "calendar_scope_ambiguous"
    assert hosted_runner._goal_one_reason_code(ambiguous_party) == "participant_identity_ambiguous"


def test_run_hosted_dispatches_suryodaya_check_and_writes_contract_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tools = complete_goal_one_tools()
    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", lambda config: object())
    monkeypatch.setattr(hosted_runner, "AgentSwitchCalendarTools", lambda client: tools)
    results_path = tmp_path / "results.json"

    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "https://agentswitch.theschoolofai.in",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=results_path,
        now=NOW,
    )

    document = parse_results(output_path.read_text(encoding="utf-8"))
    assert output_path == results_path
    assert [check.task_id for check in checks] == [
        "suryodaya.goal1_read_only_candidate",
        "suryodaya.goal2_read_only_plan",
    ]
    assert [task["id"] for task in document["tasks"]] == [
        "suryodaya.goal1_read_only_candidate",
        "suryodaya.goal2_read_only_plan",
    ]
    assert checks[0].passed
    assert document["tasks"][0]["passed"] is True
    assert document["tasks"][0]["evidence"].startswith("HARNESS_PREDICATE:")
    assert not checks[1].passed
    assert "Goal 2 did not produce a plan" in document["tasks"][1]["evidence"]
    assert "official_evaluator=not_invoked" in document["summary"]
    assert "secret-token" not in results_path.read_text(encoding="utf-8")


def test_run_hosted_dispatches_keystone_without_live_calls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        hosted_runner.KeystoneClient,
        "list_tools",
        lambda self: SimpleNamespace(
            tools=tuple(
                SimpleNamespace(name=name)
                for name in hosted_runner.RESOURCE_READ_OPERATIONS
            )
        ),
    )
    monkeypatch.setattr(
        hosted_runner,
        "check_keystone_goal_one",
        lambda adapter, **kwargs: hosted_runner._task(
            task_id="keystone.goal1_read_only_candidate",
            title="Keystone Goal 1",
            passed=False,
            evidence="UNEVALUATED: fixture.",
        ),
    )
    monkeypatch.setattr(
        hosted_runner,
        "check_keystone_goal_two",
        lambda adapter, **kwargs: hosted_runner._task(
            task_id="keystone.goal2_read_only_plan",
            title="Keystone Goal 2",
            passed=False,
            evidence="UNEVALUATED: fixture.",
        ),
    )

    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "keystone",
            "AGENTSWITCH_BASE_URL": "http://runner.internal:8000/tenant/keystone",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=tmp_path / "results.json",
        now=NOW,
    )

    document = parse_results(output_path.read_text(encoding="utf-8"))
    assert [check.task_id for check in checks] == [
        "keystone.mcp_tools_list",
        "keystone.goal1_read_only_candidate",
        "keystone.goal2_read_only_plan",
    ]
    assert document["tasks"][0]["passed"] is True
    assert document["tasks"][1]["passed"] is False
    assert document["tasks"][2]["passed"] is False
    assert "official_evaluator=not_invoked" in document["summary"]
    assert "secret-token" not in output_path.read_text(encoding="utf-8")
    run_id = document["summary"].split("run_id=", maxsplit=1)[1].split(";", maxsplit=1)[0]
    journal_files = list(
        (tmp_path / ".harness-artifacts" / "runs" / run_id / "keystone").glob("*.json")
    )
    assert len(journal_files) == 3
    journal_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in journal_files
    ]
    assert all(record["phase"] == "scored" for record in journal_records)
    assert all(record["run_id"] == run_id for record in journal_records)


@pytest.mark.parametrize("passed", [True, False])
def test_main_success_exit_code_is_independent_of_task_results(
    monkeypatch,
    tmp_path: Path,
    passed: bool,
) -> None:
    output_path = tmp_path / "results.json"
    checks = (
        hosted_runner._task(
            task_id="fixture",
            title="Fixture task",
            passed=passed,
            evidence="fixture evidence",
        ),
    )
    monkeypatch.setattr(
        hosted_runner,
        "run_hosted",
        lambda environment, *, output_path: (output_path, checks),
    )

    exit_code = hosted_runner.main(["--results", str(output_path)])

    assert exit_code == 0


def test_main_returns_nonzero_when_results_cannot_be_written(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_run(environment, *, output_path):
        raise hosted_runner.ServerRunnerError("synthetic result failure")

    monkeypatch.setattr(hosted_runner, "run_hosted", fail_run)

    assert hosted_runner.main(["--results", str(tmp_path / "results.json")]) == 1


def test_custom_backend_checks_are_persisted_in_official_results_format(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "backend-results.json"
    check = check_backend_rescheduling_state(
        before=hosted_runner.BackendCalendarSnapshot(
            calendar_id="calendar-1",
            events=(),
            tool_calls=(),
        ),
        after=hosted_runner.BackendCalendarSnapshot(
            calendar_id="calendar-1",
            events=(),
            tool_calls=(),
        ),
        expected_moves=(),
        task_id="suryodaya.goal2.backend_state",
        title="Goal 2 backend state check",
    )

    written_path = write_hosted_results(
        output_path,
        (check,),
        summary="1 backend check recorded",
    )

    document = parse_results(written_path.read_text(encoding="utf-8"))
    assert document["tasks"] == [check.official_record()]
    assert document["summary"] == "1 backend check recorded"


def test_goal_two_journal_is_written_before_harness_predicate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    journal = HostedRunJournal(
        tmp_path / "journals",
        instance="suryodaya",
        run_id="before-score",
    )
    task_id = "suryodaya.goal2_read_only_plan"
    journal.begin(
        task_id=task_id,
        title="Suryodaya read-only Goal 2 plan",
        goal="calendar.move_after_audit",
    )
    original_verify = hosted_runner.verify_result
    verifier_calls = 0

    def verify_after_journal(result, *, expected_status):
        nonlocal verifier_calls
        verifier_calls += 1
        assert journal.has_execution_record(task_id)
        record = json.loads(journal.path_for(task_id).read_text(encoding="utf-8"))
        assert record["phase"] == "execution_recorded"
        return original_verify(result, expected_status=expected_status)

    monkeypatch.setattr(hosted_runner, "verify_result", verify_after_journal)

    check = check_suryodaya_goal_two(
        complete_goal_two_tools(),
        journal=journal,
    )

    assert check.passed
    assert verifier_calls == 1
    journal.finish(check)
