from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from harnesses.keystone_harness import (
    GOAL_ONE_ENDPOINT_READ_OPERATIONS,
    GOAL_ONE_LIST_READ_OPERATIONS,
    KeystoneCalendarAdapter,
    KeystoneClient,
    KeystoneConfig,
)
from harnesses.suryodaya_harness.hosted_runner import (
    KEYSTONE_INSTANCE,
    check_keystone_goal_one,
)
from harnesses.keystone_harness.goal_integration import KeystoneCalendarTool


_REQUIRED_OPERATIONS = frozenset(
    {
        "Party.list",
        "Calendar.list",
    }
) | GOAL_ONE_LIST_READ_OPERATIONS | GOAL_ONE_ENDPOINT_READ_OPERATIONS


def _party_identity_diagnostics(
    tools: KeystoneCalendarTool,
) -> tuple[dict[str, object], str | None]:
    candidates: dict[str, dict[str, object]] = {}
    conflicting_records: set[str] = set()
    invalid_role_records = 0
    queries: list[dict[str, object]] = []

    for role in ("Plant Manager", "Plant Head"):
        parties = tools.find_parties(limit=1000, job_title=role)
        role_matches = 0
        title_counts = {"Plant Manager": 0, "Plant Head": 0}
        for party in parties:
            title = party.get("job_title")
            if not isinstance(title, str) or not title.strip():
                invalid_role_records += 1
                continue
            normalized_title = title.strip().casefold()
            if normalized_title not in {"plant manager", "plant head"}:
                continue
            role_matches += 1
            if normalized_title == "plant manager":
                title_counts["Plant Manager"] += 1
            else:
                title_counts["Plant Head"] += 1
            party_id = party.get("id")
            email = party.get("email")
            if (
                not isinstance(party_id, str)
                or not party_id.strip()
                or not isinstance(email, str)
                or not email.strip()
            ):
                invalid_role_records += 1
                continue
            previous = candidates.get(party_id)
            if previous is not None and any(
                isinstance(previous.get(field), str)
                and isinstance(party.get(field), str)
                and previous[field].strip().casefold()
                != party[field].strip().casefold()
                for field in ("email", "job_title", "name")
            ):
                conflicting_records.add(party_id)
            else:
                candidates[party_id] = party
        queries.append(
            {
                "role_filter": role,
                "records_returned": len(parties),
                "valid_role_matches": role_matches,
                "matched_job_title_counts": title_counts,
            }
        )

    diagnostics = {
        "queries": queries,
        "distinct_candidate_count": len(candidates),
        "invalid_role_record_count": invalid_role_records,
        "conflicting_duplicate_identity_count": len(conflicting_records),
    }
    unique_party_id = (
        next(iter(candidates))
        if len(candidates) == 1
        and invalid_role_records == 0
        and not conflicting_records
        else None
    )
    return diagnostics, unique_party_id


def _calendar_scope_diagnostics(
    tools: KeystoneCalendarTool,
    party_id: str | None,
) -> dict[str, object]:
    if party_id is None:
        return {"evaluated": False, "reason": "party_identity_not_unique"}

    calendars = tools.list_calendars(limit=1000, owner_party_id=party_id)
    owned_calendars = [
        calendar
        for calendar in calendars
        if isinstance(calendar.get("id"), str)
        and calendar["id"].strip()
        and calendar.get("owner_party_id") == party_id
    ]
    return {
        "evaluated": True,
        "records_returned": len(calendars),
        "owned_calendar_count": len(owned_calendars),
        "owned_calendars_with_timezone": sum(
            isinstance(calendar.get("timezone"), str)
            and bool(calendar["timezone"].strip())
            for calendar in owned_calendars
        ),
    }


def main() -> int:
    argparse.ArgumentParser(
        description=(
            "Run a read-only Keystone Goal 1 candidate check; requires the "
            "KEYSTONE_* configuration set by Connect-Keystone."
        )
    ).parse_args()
    local_config = KeystoneConfig.from_env()
    if not local_config.token:
        raise SystemExit(
            "Keystone credentials are not configured. Run Connect-Keystone "
            "in this PowerShell session first."
        )
    token = local_config.require_token()
    runner_config = KeystoneConfig.for_hosted_runner(
        base_url=local_config.base_url,
        token=token,
    )
    client = KeystoneClient(
        runner_config,
        enable_hosted_read_transport=True,
    )
    discovery = client.list_tools()
    discovered_operations = {tool.name for tool in discovery.tools}
    missing = sorted(_REQUIRED_OPERATIONS - discovered_operations)
    if missing:
        report = {
            "instance": KEYSTONE_INSTANCE,
            "catalog_completeness": discovery.completeness.status.value,
            "required_operations_missing": missing,
            "candidate_check_ran": False,
            "passed": False,
            "business_reads_are_read_only": True,
            "mutations_enabled": False,
        }
    else:
        adapter = KeystoneCalendarAdapter(client)
        tools = KeystoneCalendarTool(adapter)
        identity_diagnostics, unique_party_id = _party_identity_diagnostics(tools)
        calendar_diagnostics = _calendar_scope_diagnostics(
            tools, unique_party_id
        )
        check = check_keystone_goal_one(
            adapter,
            now=datetime.now(timezone.utc),
        )
        report = {
            "instance": KEYSTONE_INSTANCE,
            "catalog_completeness": discovery.completeness.status.value,
            "required_operations_missing": [],
            "candidate_check_ran": True,
            "passed": check.passed,
            "evidence": check.evidence,
            "identity_diagnostics": identity_diagnostics,
            "calendar_diagnostics": calendar_diagnostics,
            "business_reads_are_read_only": True,
            "mutations_enabled": False,
        }
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
