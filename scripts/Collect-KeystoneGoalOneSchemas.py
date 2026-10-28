from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import requests


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from harnesses.keystone_harness.config import KeystoneConfig
from harnesses.keystone_harness.discovery import serialize_evidence


_MCP_PROTOCOL_VERSION = "2025-11-25"
_REQUIRED_OPERATIONS = frozenset(
    {
        "SchedulingPreferences.list",
        "AvailabilityRule.list",
        "endpoint.calendar.free_busy",
    }
)


def _is_goal_one_schema_name(name: str) -> bool:
    normalized = name.casefold()
    return (
        "schedulingpreferences" in normalized
        or "availabilityrule" in normalized
        or re.search(r"free[._-]?busy", normalized) is not None
    )


def _post_json(
    session: requests.Session,
    config: KeystoneConfig,
    token: str,
    payload: dict[str, object],
) -> dict[str, Any]:
    try:
        response = session.post(
            config.endpoint,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json=payload,
            timeout=config.timeout,
            allow_redirects=False,
        )
    except requests.Timeout:
        raise RuntimeError("Keystone MCP schema request timed out.") from None
    except requests.RequestException:
        raise RuntimeError("Keystone MCP schema request failed.") from None

    try:
        if type(response.status_code) is not int or not 200 <= response.status_code < 300:
            raise RuntimeError(
                f"Keystone MCP schema request returned HTTP {response.status_code}."
            )
        try:
            message = response.json()
        except ValueError:
            raise RuntimeError("Keystone MCP schema request returned invalid JSON.") from None
    finally:
        response.close()

    if not isinstance(message, dict):
        raise RuntimeError("Keystone MCP schema response was not a JSON object.")
    return message


def _rpc_result(message: dict[str, Any], request_id: int) -> dict[str, Any]:
    if (
        message.get("jsonrpc") != "2.0"
        or type(message.get("id")) is not int
        or message["id"] != request_id
    ):
        raise RuntimeError("Keystone MCP schema response did not match its request.")
    if message.get("error") is not None:
        raise RuntimeError("Keystone MCP schema request returned an RPC error.")
    result = message.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Keystone MCP schema response had no result object.")
    return result


def _initialize(
    session: requests.Session,
    config: KeystoneConfig,
    token: str,
) -> None:
    result = _rpc_result(
        _post_json(
            session,
            config,
            token,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "calendar-goal-one-schema-collector",
                        "version": "0.1.0",
                    },
                },
            },
        ),
        1,
    )
    if (
        result.get("protocolVersion") != _MCP_PROTOCOL_VERSION
        or not isinstance(result.get("capabilities"), dict)
        or not isinstance(result.get("serverInfo"), dict)
    ):
        raise RuntimeError("Keystone MCP initialize response was not valid.")

    try:
        response = session.post(
            config.endpoint,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
            timeout=config.timeout,
            allow_redirects=False,
            stream=True,
        )
    except requests.Timeout:
        raise RuntimeError("Keystone MCP initialization notification timed out.") from None
    except requests.RequestException:
        raise RuntimeError("Keystone MCP initialization notification failed.") from None
    try:
        if type(response.status_code) is not int or not 200 <= response.status_code < 300:
            raise RuntimeError(
                "Keystone MCP initialization notification returned an invalid HTTP status."
            )
    finally:
        response.close()


def _list_tools(
    session: requests.Session,
    config: KeystoneConfig,
    token: str,
) -> tuple[list[dict[str, Any]], int, bool]:
    tools: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None
    request_id = 2
    cursor_metadata_seen = False

    for _ in range(100):
        params: dict[str, object] = {}
        if cursor is not None:
            params["cursor"] = cursor
        result = _rpc_result(
            _post_json(
                session,
                config,
                token,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/list",
                    "params": params,
                },
            ),
            request_id,
        )
        page_tools = result.get("tools")
        if not isinstance(page_tools, list):
            raise RuntimeError("Keystone tools/list response had no tools array.")
        for item in page_tools:
            if not isinstance(item, dict):
                raise RuntimeError("Keystone tools/list contained an invalid descriptor.")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("Keystone tools/list contained a descriptor without a name.")
            if name in seen_names:
                raise RuntimeError("Keystone tools/list contained a duplicate tool name.")
            seen_names.add(name)
            if _is_goal_one_schema_name(name):
                tools.append(item)

        cursor_metadata_seen = cursor_metadata_seen or "nextCursor" in result
        next_cursor = result.get("nextCursor")
        if next_cursor is None:
            return tools, len(seen_names), cursor_metadata_seen
        if not isinstance(next_cursor, str) or not next_cursor.strip():
            raise RuntimeError("Keystone tools/list returned an invalid continuation cursor.")
        if next_cursor in seen_cursors:
            raise RuntimeError("Keystone tools/list repeated a continuation cursor.")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        request_id += 1

    raise RuntimeError("Keystone tools/list exceeded the pagination safety limit.")


def _build_report(
    tool_descriptors: list[dict[str, Any]],
    discovered_tool_count: int,
    catalog_completeness: str,
) -> dict[str, object]:
    discovered_names = {
        item["name"] for item in tool_descriptors if isinstance(item.get("name"), str)
    }
    return {
        "schema_version": 1,
        "instance": "keystone",
        "discovered_tool_count": discovered_tool_count,
        "catalog_completeness": catalog_completeness,
        "required_exact_operations_missing": sorted(
            _REQUIRED_OPERATIONS - discovered_names
        ),
        "goal_one_related_operations": tool_descriptors,
        "business_tools_called": False,
        "calendar_data_read": False,
        "mutation_tools_called": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Keystone Goal 1 MCP schemas using initialize and tools/list only."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "keystone-goal1-schemas.json",
        help="Output path (default: a JSON file in the system temp directory).",
    )
    args = parser.parse_args(argv)

    config = KeystoneConfig.from_env()
    token = config.require_token()
    with requests.Session() as session:
        _initialize(session, config, token)
        descriptors, discovered_count, catalog_complete = _list_tools(
            session, config, token
        )

    report = _build_report(
        descriptors,
        discovered_count,
        "complete" if catalog_complete else "unknown",
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        serialize_evidence(report) + "\n",
        encoding="utf-8",
    )
    print(f"Credential-free schema report written to: {output_path}")
    print(f"Goal 1 related operations found: {len(descriptors)}")
    print("Business tools called: False; calendar data read: False; mutations: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
