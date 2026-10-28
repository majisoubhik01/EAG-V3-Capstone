from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from harnesses.keystone_harness import (
    CollectionEvidence,
    CompletenessResult,
    CompletenessStatus,
    DiscoveredTool,
    KeystoneClient,
    KeystoneConfig,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
    KeystoneReadAuthorization,
    PRODUCTION_READ_ONLY_OPERATIONS,
    RESOURCE_READ_OPERATIONS,
    ToolDiscoveryResult,
    validate_resource_read_policy,
)


_FILTER_PROPERTIES: dict[str, dict[str, Any]] = {
    "Party.list": {
        "job_title": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "company_id": {"type": "string", "minLength": 1},
        "type": {"type": "string", "enum": ["individual", "organization"]},
        "portal_enabled": {
            "anyOf": [{"type": "boolean"}, {"type": "integer"}],
        },
    },
    "Calendar.list": {
        "id": {"type": "string", "minLength": 1},
        "owner_party_id": {"type": "string"},
        "timezone": {"type": "string"},
        "type": {
            "type": "string",
            "enum": ["personal", "team", "birthday", "subscribed", "holiday"],
        },
    },
    "CalendarEvent.list": {
        "search": {"type": "string"},
        "calendar_id": {"type": "string"},
        "start_at": {
            "anyOf": [
                {"type": "string"},
                {"type": "string", "pattern": "^eq:"},
            ],
        },
        "visibility": {
            "type": "string",
            "enum": ["public", "private", "confidential"],
        },
    },
    "EventAttendee.list": {
        "event_id": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "party_id": {"type": "string"},
        "role": {"type": "string", "enum": ["chair", "req", "opt"]},
        "status": {
            "type": "string",
            "enum": ["needs_action", "accepted", "declined", "tentative"],
        },
        "is_organizer": {
            "anyOf": [{"type": "boolean"}, {"type": "integer"}],
        },
    },
    "SchedulingPreferences.list": {
        "company_id": {"type": "string", "minLength": 1},
    },
    "AvailabilityRule.list": {
        "owner_party_id": {"type": "string", "minLength": 1},
        "calendar_id": {"type": "string", "minLength": 1},
        "company_id": {"type": "string", "minLength": 1},
    },
}


def _tool(
    name: str,
    *,
    read_only_hint: bool | None = True,
    annotations: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> DiscoveredTool:
    return DiscoveredTool(
        name=name,
        read_only_hint=read_only_hint,
        description="Synthetic list-tool descriptor",
        input_schema=(
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "default": 20,
                    },
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    **_FILTER_PROPERTIES.get(name, {}),
                },
                "required": [],
                "additionalProperties": False,
            }
            if input_schema is None
            else input_schema
        ),
        annotations=(
            {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            }
            if annotations is None
            else annotations
        ),
        output_schema=(
            {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string", "minLength": 1}},
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
            }
            if output_schema is None
            else output_schema
        ),
        meta=(
            {
                "agentswitch": {
                    "permission": "read",
                    "discovery": "caller",
                    "risk": "READ",
                }
            }
            if meta is None
            else meta
        ),
    )


def _discovery(*tools: DiscoveredTool) -> ToolDiscoveryResult:
    return ToolDiscoveryResult(
        tools=tools,
        collection_evidence=CollectionEvidence(()),
        completeness=CompletenessResult(
            CompletenessStatus.UNKNOWN,
            observed_record_count=None,
            expected_record_count=None,
            reasons=("synthetic catalog fixture",),
        ),
    )


def _arguments(operation: str) -> dict[str, Any]:
    base: dict[str, Any] = {"limit": 20, "offset": 0}
    return {
        "Party.list": {**base, "job_title": "Plant Manager", "email": "manager@example.test"},
        "Calendar.list": {**base, "owner_party_id": "party-1", "type": "personal"},
        "CalendarEvent.list": {
            **base,
            "calendar_id": "calendar-1",
            "start_at": "eq:2026-10-17T09:00:00Z",
        },
        "EventAttendee.list": {
            **base,
            "event_id": "event-1",
            "role": "req",
            "status": "accepted",
            "is_organizer": True,
        },
        "SchedulingPreferences.list": {**base, "company_id": "company-1"},
        "AvailabilityRule.list": {
            **base,
            "owner_party_id": "party-1",
            "calendar_id": "calendar-1",
        },
    }[operation]


class SyntheticToolsListResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class SyntheticToolsListSession:
    def __init__(self, descriptors: tuple[DiscoveredTool, ...]) -> None:
        self.descriptors = descriptors
        self.methods: list[str] = []

    def post(self, url: str, **kwargs: Any) -> SyntheticToolsListResponse:
        request = kwargs["json"]
        self.methods.append(request["method"])
        if request["method"] != "tools/list":
            raise AssertionError("synthetic session cannot invoke business tools")
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema,
                "annotations": tool.annotations,
                "_meta": tool.meta,
            }
            for tool in self.descriptors
        ]
        return SyntheticToolsListResponse({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"tools": tools, "nextCursor": None},
        })


def _client_with_discovery(
    *tools: DiscoveredTool,
) -> tuple[KeystoneClient, ToolDiscoveryResult, SyntheticToolsListSession]:
    session = SyntheticToolsListSession(tuple(tools))
    client = KeystoneClient(
        KeystoneConfig(token="offline-policy-test-token"),
        session=session,  # type: ignore[arg-type]
    )
    discovery = client.list_tools()
    return client, discovery, session


@pytest.mark.parametrize("operation", sorted(RESOURCE_READ_OPERATIONS))
def test_exact_resource_read_candidates_pass_with_consistent_metadata(
    operation: str,
) -> None:
    descriptor = _tool(operation)

    validated = validate_resource_read_policy(
        _discovery(descriptor),
        operation,
        _arguments(operation),
    )

    assert validated is descriptor


@pytest.mark.parametrize(
    ("operation", "descriptor"),
    [
        ("Party.get", _tool("Party.get")),
        ("Party.create", _tool("Party.create")),
        ("Party.list.extra", _tool("Party.list.extra")),
        ("endpoint.calendar.events", _tool("endpoint.calendar.events")),
    ],
)
def test_non_allowlisted_operations_are_denied_even_with_read_hints(
    operation: str,
    descriptor: DiscoveredTool,
) -> None:
    with pytest.raises(KeystoneOperationNotAllowedError, match="exact read candidate"):
        validate_resource_read_policy(
            _discovery(descriptor),
            operation,
            {"limit": 1, "offset": 0},
        )


def test_operation_must_be_discovered_exactly_once() -> None:
    descriptor = _tool("Party.list")
    with pytest.raises(KeystoneOperationNotAllowedError, match="exactly once"):
        validate_resource_read_policy(_discovery(), "Party.list", {"limit": 1, "offset": 0})
    with pytest.raises(KeystoneOperationNotAllowedError, match="exactly once"):
        validate_resource_read_policy(
            _discovery(descriptor, descriptor),
            "Party.list",
            {"limit": 1, "offset": 0},
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"read_only_hint": False}, "read-only"),
        ({"read_only_hint": None}, "read-only"),
        (
            {"annotations": {"readOnlyHint": False, "destructiveHint": False}},
            "readOnlyHint",
        ),
        (
            {"annotations": {"readOnlyHint": True}},
            "destructiveHint",
        ),
        (
            {
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": True,
                }
            },
            "destructiveHint",
        ),
        (
            {
                "meta": {
                    "agentswitch": {
                        "permission": "write",
                        "risk": "READ",
                    }
                }
            },
            "permission",
        ),
        (
            {
                "meta": {
                    "agentswitch": {
                        "permission": "read",
                        "risk": "WRITE",
                    }
                }
            },
            "risk",
        ),
        (
            {
                "meta": {
                    "agentswitch": {
                        "permission": "read",
                        "risk": "DESTRUCTIVE",
                    }
                }
            },
            "risk",
        ),
        ({"meta": {}}, "AgentSwitch permission"),
        ({"meta": None}, "_meta"),
    ],
)
def test_missing_or_contradictory_safety_metadata_is_denied(
    updates: dict[str, Any],
    message: str,
) -> None:
    descriptor = replace(_tool("Party.list"), **updates)

    with pytest.raises(KeystoneOperationNotAllowedError, match=message):
        validate_resource_read_policy(
            _discovery(descriptor),
            "Party.list",
            {"limit": 1, "offset": 0},
        )


@pytest.mark.parametrize(
    ("input_schema", "output_schema", "message"),
    [
        ("not-an-object", None, "inputSchema"),
        (
            {
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}},
                "additionalProperties": True,
            },
            None,
            "additional properties",
        ),
        (
            {
                "type": "object",
                "properties": {"limit": {"type": "string"}, "offset": {"type": "integer"}},
                "additionalProperties": False,
            },
            None,
            "limit must be an integer",
        ),
        (None, None, "outputSchema"),
    ],
)
def test_malformed_discovered_schemas_are_rejected(
    input_schema: Any,
    output_schema: dict[str, Any] | None,
    message: str,
) -> None:
    descriptor = replace(
        _tool("Party.list"),
        input_schema=(
            _tool("Party.list").input_schema
            if input_schema is None and output_schema is None
            else input_schema
        ),
        output_schema=output_schema,
    )
    if input_schema is None and output_schema is None:
        descriptor = replace(descriptor, output_schema=None)

    with pytest.raises(KeystoneProtocolError, match=message):
        validate_resource_read_policy(
            _discovery(descriptor),
            "Party.list",
            {"limit": 1, "offset": 0},
        )


def test_output_pagination_fields_must_be_integers() -> None:
    descriptor = _tool("Party.list")
    output_schema = dict(descriptor.output_schema or {})
    properties = dict(output_schema["properties"])
    properties["total"] = {"type": "string"}
    output_schema["properties"] = properties
    descriptor = replace(descriptor, output_schema=output_schema)

    with pytest.raises(KeystoneProtocolError, match="total must be an integer"):
        validate_resource_read_policy(
            _discovery(descriptor),
            "Party.list",
            {"limit": 1, "offset": 0},
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"limit": 1, "offset": 0, "made_up": "x"}, "not allowed"),
        ({"limit": True, "offset": 0}, "wrong JSON type"),
        ({"limit": 1001, "offset": 0}, "maximum"),
        ({"limit": 20, "offset": -1}, "minimum"),
        ({"limit": 20, "offset": 0, "type": "person"}, "enum"),
        ({"limit": 20, "offset": 0, "email": "not-an-email"}, "format"),
        ({"limit": 20, "offset": 0, "company_id": ""}, "minimum"),
    ],
)
def test_arguments_are_validated_against_discovered_schema(
    arguments: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_resource_read_policy(
            _discovery(_tool("Party.list")),
            "Party.list",
            arguments,
        )


def test_any_of_validates_supported_boolean_or_integer_filters() -> None:
    descriptor = _tool("Party.list")

    for value in (True, 1):
        validated = validate_resource_read_policy(
            _discovery(descriptor),
            "Party.list",
            {"limit": 20, "offset": 0, "portal_enabled": value},
        )
        assert validated is descriptor

    with pytest.raises(ValueError, match="does not match any allowed schema"):
        validate_resource_read_policy(
            _discovery(descriptor),
            "Party.list",
            {"limit": 20, "offset": 0, "portal_enabled": "true"},
        )


def test_authorization_is_bound_to_the_discovery_made_by_the_same_client() -> None:
    descriptor = _tool("Party.list")
    client, discovery, session = _client_with_discovery(descriptor)
    authorization = client.authorize_read_request(
        "Party.list",
        {"limit": 1, "offset": 0},
    )

    assert isinstance(authorization, KeystoneReadAuthorization)
    assert client.validate_read_authorization(authorization) is discovery.tools[0]
    assert session.methods == ["tools/list"]


def test_client_can_authorize_from_unknown_catalog_completeness() -> None:
    class CursorlessToolsListSession(SyntheticToolsListSession):
        def post(self, url: str, **kwargs: Any) -> SyntheticToolsListResponse:
            result = super().post(url, **kwargs)
            result.payload["result"].pop("nextCursor")
            return result

    client = KeystoneClient(
        KeystoneConfig(token="offline-policy-test-token"),
        session=CursorlessToolsListSession((_tool("Party.list"),)),
    )  # type: ignore[arg-type]
    discovery = client.list_tools()

    assert discovery.completeness.status is CompletenessStatus.UNKNOWN
    authorization = client.authorize_read_request(
        "Party.list",
        {"limit": 1, "offset": 0},
    )
    assert client.validate_read_authorization(authorization).name == "Party.list"


def test_fabricated_or_unrelated_discovery_cannot_create_client_authorization() -> None:
    descriptor = _tool("Party.list")
    fabricated = _discovery(descriptor)
    client = KeystoneClient(KeystoneConfig(token="offline-policy-test-token"))

    with pytest.raises(KeystoneOperationNotAllowedError, match="no successful tools/list"):
        client.authorize_read_request("Party.list", _arguments("Party.list"))
    with pytest.raises(TypeError):
        client.authorize_read_request(
            "Party.list",
            _arguments("Party.list"),
            discovery=fabricated,  # type: ignore[call-arg]
        )

    first_client, _, _ = _client_with_discovery(descriptor)
    second_client, _, _ = _client_with_discovery(descriptor)
    with pytest.raises(KeystoneOperationNotAllowedError, match="not bound"):
        second_client.validate_read_authorization(
            first_client.authorize_read_request(
                "Party.list",
                _arguments("Party.list"),
            )
        )
    assert fabricated.tools == (descriptor,)


def test_descriptor_mutation_invalidates_client_discovery_provenance() -> None:
    client, discovery, _ = _client_with_discovery(_tool("Party.list"))
    discovery.tools[0].annotations["readOnlyHint"] = False

    with pytest.raises(KeystoneOperationNotAllowedError, match="changed after discovery"):
        client.authorize_read_request("Party.list", {"limit": 1, "offset": 0})


def test_discovery_refresh_invalidates_prior_authorizations() -> None:
    client, _, _ = _client_with_discovery(_tool("Party.list"))
    authorization = client.authorize_read_request(
        "Party.list",
        {"limit": 1, "offset": 0},
    )
    client.list_tools()

    with pytest.raises(KeystoneOperationNotAllowedError, match="not bound"):
        client.validate_read_authorization(authorization)


def test_reconstructed_authorization_capability_is_not_client_issued() -> None:
    client, _, _ = _client_with_discovery(_tool("Party.list"))
    authorization = client.authorize_read_request(
        "Party.list",
        {"limit": 1, "offset": 0},
    )

    with pytest.raises(KeystoneOperationNotAllowedError, match="not bound"):
        client.validate_read_authorization(replace(authorization))


def test_changing_client_session_invalidates_discovery_provenance() -> None:
    client, _, _ = _client_with_discovery(_tool("Party.list"))
    client._session = SyntheticToolsListSession((_tool("Party.list"),))  # type: ignore[assignment]

    with pytest.raises(KeystoneOperationNotAllowedError, match="session changed"):
        client.authorize_read_request("Party.list", {"limit": 1, "offset": 0})


def test_policy_approval_does_not_enable_client_transport() -> None:
    descriptor = _tool("Party.list")
    arguments = {"limit": 20, "offset": 0, "job_title": "Plant Manager"}
    client, _, session = _client_with_discovery(descriptor)
    authorization = client.authorize_read_request("Party.list", arguments)
    assert client.validate_read_authorization(authorization) == descriptor

    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_read("Party.list", arguments)
    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_authorized_read(authorization)

    assert session.methods == ["tools/list"]
    assert PRODUCTION_READ_ONLY_OPERATIONS == ()
