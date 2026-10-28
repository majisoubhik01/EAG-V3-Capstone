from __future__ import annotations

import importlib
import inspect
from io import BytesIO
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest
import requests
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import ReadTimeoutError
from urllib3.response import HTTPResponse
from requests.adapters import HTTPAdapter, Retry

from harnesses.keystone_harness import calendar_list_transport as transport_module
from harnesses.keystone_harness.calendar_list_pilot import (
    CalendarListSmokePilot,
)
from harnesses.keystone_harness.calendar_list_transport import (
    CalendarListPilotTransport,
)
from harnesses.keystone_harness.client import (
    DiscoveredTool,
    KeystoneClient,
    KeystoneAuthenticationError,
    KeystoneConfig,
    KeystoneHttpError,
    KeystoneJsonError,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
    KeystoneReadAuthorization,
    KeystoneRpcError,
    KeystoneTimeoutError,
    KeystoneToolError,
    KeystoneTransportError,
    ToolDiscoveryResult,
)
from harnesses.keystone_harness.discovery import (
    CollectionEvidence,
    CompletenessResult,
    CompletenessStatus,
)
from harnesses.keystone_harness import PRODUCTION_READ_ONLY_OPERATIONS
from harnesses.keystone_harness.mcp_response import McpContentKind


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
        input_schema=input_schema if input_schema is not None else {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 0, "maximum": 1000},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": [],
            "additionalProperties": False,
        },
        annotations=annotations if annotations is not None else {
            "readOnlyHint": True,
            "destructiveHint": False,
        },
        output_schema=output_schema if output_schema is not None else {
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
        },
        meta=meta if meta is not None else {
            "agentswitch": {"permission": "read", "risk": "READ"}
        },
    )


class _Response:
    def __init__(
        self,
        body: Any = None,
        *,
        status_code: int = 200,
        json_error: ValueError | None = None,
        raw_body: str | None = None,
    ) -> None:
        self._body = body
        self.status_code = status_code
        self._json_error = json_error
        self._raw_body = raw_body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._body


class _ToolsListOnlySession:
    def __init__(self, tools: tuple[DiscoveredTool, ...]) -> None:
        self.tools = tools
        self.methods: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.client: KeystoneClient | None = None
        self.business_response: Callable[[dict[str, Any]], _Response | BaseException] = (
            lambda request: _Response(
                _response(_page(), request_id=request["id"])
            )
        )
        self.business_entered: threading.Event | None = None
        self.release_business: threading.Event | None = None

    def post(self, url: str, **kwargs: Any) -> _Response:
        request = dict(kwargs["json"])
        self.calls.append({"url": url, **kwargs, "json": request})
        self.methods.append(request["method"])
        if request["method"] != "tools/list":
            raise AssertionError("business calls must use the offline pilot seam")
        return _Response({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "tools": [
                    {
                        "name": tool.name,
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

    def send_pilot(
        self,
        adapter: HTTPAdapter,
        prepared_request: requests.PreparedRequest,
        **kwargs: Any,
    ) -> requests.Response:
        request = json.loads(prepared_request.body)
        self.calls.append({
            "url": prepared_request.url,
            "json": request,
            "headers": dict(prepared_request.headers),
            "kwargs": kwargs,
        })
        self.methods.append(request["method"])
        if request["method"] != "tools/call":
            raise AssertionError("offline pilot seam received an unexpected request")
        if self.client is None or not self.client._calendar_list_attempted:
            raise AssertionError("pilot attempt was not consumed before transport")
        if adapter is not self.client._calendar_list_transport._adapter:
            raise AssertionError("pilot used an adapter other than its owned adapter")
        if self.business_entered is not None:
            self.business_entered.set()
        if self.release_business is not None and not self.release_business.wait(5):
            raise AssertionError("test did not release the synthetic business request")
        response = self.business_response(request)
        if isinstance(response, BaseException):
            raise response
        http_response = requests.Response()
        http_response.status_code = response.status_code
        http_response.url = prepared_request.url
        http_response.request = prepared_request
        if response._json_error is not None:
            http_response._content = (response._raw_body or "not-json").encode()
        else:
            http_response._content = json.dumps(response._body).encode()
        http_response.encoding = "utf-8"
        return http_response


_POOL_SPIES: dict[tuple[str, int | None], tuple[HTTPAdapter, _ToolsListOnlySession]] = {}


@pytest.fixture(autouse=True)
def _use_synthetic_https_pool(monkeypatch: pytest.MonkeyPatch):
    def synthetic_urlopen(
        pool: HTTPSConnectionPool,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: Any = None,
        **kwargs: Any,
    ) -> HTTPResponse:
        binding = _POOL_SPIES.get((pool.host, pool.port))
        if binding is None:
            raise AssertionError("unexpected HTTPS pool dispatch")
        adapter, spy = binding
        authority = pool.host if pool.port == 443 else f"{pool.host}:{pool.port}"
        prepared_request = requests.PreparedRequest()
        prepared_request.prepare(
            method=method,
            url=f"https://{authority}{url}",
            headers=dict(headers or {}),
            data=body,
        )
        try:
            response = spy.send_pilot(adapter, prepared_request, **kwargs)
        except requests.Timeout as exc:
            raise ReadTimeoutError(pool, url, str(exc)) from exc
        return HTTPResponse(
            body=BytesIO(response.content),
            status=response.status_code,
            headers={
                **response.headers,
                "Content-Length": str(len(response.content)),
            },
            preload_content=False,
            reason=response.reason or "",
        )

    monkeypatch.setattr(HTTPSConnectionPool, "urlopen", synthetic_urlopen)
    yield
    _POOL_SPIES.clear()


def _pilot(
    tools: tuple[DiscoveredTool, ...] | None = None,
) -> tuple[CalendarListSmokePilot, KeystoneClient, _ToolsListOnlySession]:
    discovered = tools or (
        _tool("Calendar.list"),
        _tool("Party.list"),
        _tool("CalendarEvent.list"),
        _tool("EventAttendee.list"),
        _tool("Calendar.create"),
        _tool("endpoint.calendar.free_busy"),
        _tool("endpoint.calendar.conference.conflicts"),
    )
    spy = _ToolsListOnlySession(discovered)
    session = requests.Session()
    session.post = spy.post  # type: ignore[method-assign]
    client = KeystoneClient(
        KeystoneConfig(token="offline-test-token"),
        session=session,
    )
    spy.client = client
    client.list_tools()
    assert client._calendar_list_transport is None
    transport = CalendarListPilotTransport(client)
    client._calendar_list_transport = transport
    pool = transport._adapter.poolmanager.connection_from_url(client.config.endpoint)
    _POOL_SPIES[(pool.host, pool.port)] = (transport._adapter, spy)
    return CalendarListSmokePilot(client), client, spy


def _grant(pilot: CalendarListSmokePilot):
    return pilot.authorize(
        "Calendar.list",
        {"limit": 1, "offset": 0},
    )


def _response(
    candidate: dict[str, Any],
    *,
    request_id: int = 17,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [],
            "structuredContent": candidate,
        },
    }


def _page(
    *,
    data: list[dict[str, Any]] | None = None,
    limit: int = 1,
    offset: int = 0,
) -> dict[str, Any]:
    return {
        "data": data if data is not None else [{"id": "calendar-1"}],
        "total": 8,
        "limit": limit,
        "offset": offset,
    }


def test_only_calendar_list_gets_a_pilot_authorization() -> None:
    pilot, client, session = _pilot()

    grant = _grant(pilot)

    assert isinstance(grant._client_authorization, KeystoneReadAuthorization)
    assert client.validate_read_authorization(grant._client_authorization).name == "Calendar.list"
    assert session.methods == ["tools/list"]


@pytest.mark.parametrize(
    "operation",
    [
        "Party.list",
        "CalendarEvent.list",
        "EventAttendee.list",
        "Calendar.create",
        "endpoint.calendar.free_busy",
        "endpoint.calendar.conference.conflicts",
        "Calendar.list.extra",
    ],
)
def test_pilot_rejects_every_other_operation(operation: str) -> None:
    pilot, _, session = _pilot()

    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.authorize(operation, {"limit": 1, "offset": 0})

    assert session.methods == ["tools/list"]


def test_pilot_rejects_whitespace_padded_discovered_tool_name() -> None:
    descriptor = replace(_tool("Calendar.list"), name=" Calendar.list ")

    with pytest.raises(KeystoneProtocolError, match="malformed tool metadata"):
        _pilot((descriptor,))


@pytest.mark.parametrize(
    "arguments",
    [
        {"limit": 2, "offset": 0},
        {"limit": 1, "offset": 1},
        {"limit": True, "offset": 0},
        {"limit": 1, "offset": False},
        {"limit": 1, "offset": 0, "search": "calendar"},
        {"limit": 1},
    ],
)
def test_pilot_rejects_non_fixed_or_malformed_arguments(
    arguments: dict[str, Any],
) -> None:
    pilot, _, session = _pilot()

    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.authorize("Calendar.list", arguments)

    assert session.methods == ["tools/list"]


@pytest.mark.parametrize(
    ("descriptor", "error"),
    [
        (_tool("Party.list"), KeystoneOperationNotAllowedError),
        (
            _tool(
                "Calendar.list",
                annotations={"readOnlyHint": False, "destructiveHint": False},
            ),
            KeystoneOperationNotAllowedError,
        ),
        (
            _tool("Calendar.list", annotations={}),
            KeystoneOperationNotAllowedError,
        ),
        (
            _tool(
                "Calendar.list",
                meta={"agentswitch": {"permission": "write", "risk": "READ"}},
            ),
            KeystoneOperationNotAllowedError,
        ),
        (_tool("Calendar.list", meta={}), KeystoneOperationNotAllowedError),
        (
            _tool(
                "Calendar.list",
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": [],
                    "additionalProperties": True,
                },
            ),
            KeystoneProtocolError,
        ),
    ],
)
def test_invalid_descriptor_metadata_or_schema_fails_closed(
    descriptor: DiscoveredTool,
    error: type[Exception],
) -> None:
    pilot, _, _ = _pilot((descriptor,))

    with pytest.raises(error):
        _grant(pilot)


def test_malformed_output_schema_fails_closed_during_authorization() -> None:
    descriptor = _tool("Calendar.list", output_schema={"type": "string"})
    pilot, _, _ = _pilot((descriptor,))

    with pytest.raises(KeystoneProtocolError):
        _grant(pilot)


def test_missing_output_schema_fails_closed_during_discovery() -> None:
    descriptor = replace(_tool("Calendar.list"), output_schema=None)

    with pytest.raises(KeystoneProtocolError, match="outputSchema"):
        _pilot((descriptor,))


def test_duplicate_calendar_list_descriptors_fail_closed_during_discovery() -> None:
    with pytest.raises(KeystoneProtocolError, match="duplicate tool names"):
        _pilot((_tool("Calendar.list"), _tool("Calendar.list")))


def test_fabricated_discovery_or_custom_reader_cannot_construct_pilot() -> None:
    catalog = ToolDiscoveryResult(
        tools=(_tool("Calendar.list"),),
        collection_evidence=CollectionEvidence(()),
        completeness=CompletenessResult(
            CompletenessStatus.UNKNOWN,
            observed_record_count=None,
            expected_record_count=None,
            reasons=("synthetic catalog only",),
        ),
    )

    with pytest.raises(TypeError, match="concrete KeystoneClient"):
        CalendarListSmokePilot(catalog)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="concrete KeystoneClient"):
        CalendarListSmokePilot(object())  # type: ignore[arg-type]


def test_pilot_authorization_is_bound_to_the_pilot_instance() -> None:
    first, _, _ = _pilot()
    second, _, _ = _pilot()
    grant = _grant(first)

    with pytest.raises(KeystoneOperationNotAllowedError, match="not issued"):
        second.parse_response(grant, _response(_page()), 17)


def test_pilot_response_is_schema_checked_but_withheld_from_goals() -> None:
    pilot, _, session = _pilot()

    parsed = pilot.parse_response(_grant(pilot), _response(_page()), 17)

    assert parsed.output_schema_validated
    assert parsed.structured_content is None
    assert not parsed.pagination_ready
    assert parsed.unresolved_contract is not None
    assert session.methods == ["tools/list"]


def test_pilot_accepts_structured_content_with_text_without_exposing_either() -> None:
    pilot, _, session = _pilot()
    page = _page(data=[{"id": "synthetic-sensitive-row"}])
    text = json.dumps(page, separators=(",", ":"))
    message = {
        "jsonrpc": "2.0",
        "id": 17,
        "result": {
            "content": [{"type": "text", "text": text}],
            "structuredContent": page,
        },
    }

    parsed = pilot.parse_response(_grant(pilot), message, 17)

    assert parsed.content_kind is McpContentKind.MIXED
    assert parsed.output_schema_validated
    assert parsed.structured_content is None
    assert parsed.text_content == ()
    assert not parsed.pagination_ready
    assert "synthetic-sensitive-row" not in repr(parsed)
    assert text not in repr(parsed)
    assert session.methods == ["tools/list"]


def test_pilot_rejects_text_only_json_even_when_it_matches_page_schema() -> None:
    pilot, _, session = _pilot()
    text = json.dumps(_page(), separators=(",", ":"))
    message = {
        "jsonrpc": "2.0",
        "id": 17,
        "result": {
            "content": [{"type": "text", "text": text}],
        },
    }

    with pytest.raises(
        KeystoneProtocolError,
        match="success must contain structuredContent",
    ) as caught:
        pilot.parse_response(_grant(pilot), message, 17)

    assert text not in str(caught.value)
    assert text not in repr(caught.value)
    assert session.methods == ["tools/list"]


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "1.0", "id": 17, "result": {"content": []}},
        {"jsonrpc": "2.0", "id": 18, "result": {"content": []}},
        {"jsonrpc": "2.0", "id": 17},
        {
            "jsonrpc": "2.0",
            "id": 17,
            "result": {"content": []},
            "extra": "not permitted",
        },
        {
            "jsonrpc": "2.0",
            "id": 17,
            "result": {"content": []},
            "error": {"code": -32000, "message": "ambiguous"},
        },
    ],
)
def test_invalid_request_ids_and_jsonrpc_envelopes_fail_closed(
    message: dict[str, Any],
) -> None:
    pilot, _, _ = _pilot()

    with pytest.raises(KeystoneProtocolError):
        pilot.parse_response(_grant(pilot), message, 17)


def test_boolean_request_ids_are_not_accepted_as_integers() -> None:
    pilot, _, _ = _pilot()

    with pytest.raises(KeystoneProtocolError):
        pilot.parse_response(_grant(pilot), _response(_page(), request_id=1), True)
    with pytest.raises(KeystoneProtocolError):
        pilot.parse_response(
            _grant(pilot),
            _response(_page(), request_id=1) | {"id": True},
            1,
        )


def test_jsonrpc_error_is_not_a_successful_read() -> None:
    pilot, _, _ = _pilot()
    message = {
        "jsonrpc": "2.0",
        "id": 17,
        "error": {"code": -32000, "message": "synthetic denial"},
    }

    with pytest.raises(KeystoneRpcError):
        pilot.parse_response(_grant(pilot), message, 17)


def test_mcp_tool_error_is_not_a_successful_read() -> None:
    pilot, _, _ = _pilot()
    message = {
        "jsonrpc": "2.0",
        "id": 17,
        "result": {
            "isError": True,
            "content": [{"type": "text", "text": "synthetic failure"}],
        },
    }

    with pytest.raises(KeystoneToolError):
        pilot.parse_response(_grant(pilot), message, 17)


@pytest.mark.parametrize(
    "candidate",
    [
        _page(data=[{"id": "calendar-1"}, {"id": "calendar-2"}]),
        _page(limit=2),
        _page(offset=1),
    ],
)
def test_pilot_rejects_response_that_exceeds_or_disagrees_with_request(
    candidate: dict[str, Any],
) -> None:
    pilot, _, _ = _pilot()

    with pytest.raises(KeystoneProtocolError):
        pilot.parse_response(_grant(pilot), _response(candidate), 17)


def test_client_transport_remains_disabled_after_pilot_approval() -> None:
    pilot, client, session = _pilot()
    grant = _grant(pilot)

    assert PRODUCTION_READ_ONLY_OPERATIONS == ()
    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_read("Calendar.list", {"limit": 1, "offset": 0})
    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_authorized_read(grant._client_authorization)

    assert session.methods == ["tools/list"]


def test_authorization_validation_after_discovery_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot, client, spy = _pilot()

    def no_refresh() -> None:
        raise AssertionError("authorization validation must not refresh discovery")

    monkeypatch.setattr(client, "list_tools", no_refresh)
    grant = _grant(pilot)

    assert client.validate_read_authorization(grant._client_authorization).name == "Calendar.list"
    assert spy.methods == ["tools/list"]


def test_missing_stale_and_foreign_authorizations_reject_before_transport() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.invoke(None)  # type: ignore[arg-type]

    foreign_pilot, _, foreign_spy = _pilot()
    with pytest.raises(KeystoneOperationNotAllowedError):
        foreign_pilot.invoke(grant)
    assert foreign_spy.methods == ["tools/list"]

    client.list_tools()
    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.invoke(grant)

    assert spy.methods == ["tools/list", "tools/list"]


def test_valid_invoke_uses_one_exact_request_and_returns_only_classification() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    transport = client._calendar_list_transport
    session = transport._session
    adapter = transport._adapter
    retry_policy = adapter.max_retries
    assert {
        "standard session": type(session) is requests.Session,
        "no environment": session.trust_env is False,
        "no auth": session.auth is None,
        "no cookies": not session.cookies,
        "no proxies": not session.proxies,
        "no hooks": session.hooks == {"response": []},
        "https adapter only": set(session.adapters) == {"https://"},
        "owned adapter mounted": session.adapters.get("https://") is adapter,
        "standard adapter": type(adapter) is HTTPAdapter,
        "retry categories zero": all(
            getattr(retry_policy, category) == 0
            for category in ("total", "connect", "read", "redirect", "status")
        ),
        "standard adapter send": HTTPAdapter.send
        is transport_module._STANDARD_HTTP_ADAPTER_SEND,
    } == {
        "standard session": True,
        "no environment": True,
        "no auth": True,
        "no cookies": True,
        "no proxies": True,
        "no hooks": True,
        "https adapter only": True,
        "owned adapter mounted": True,
        "standard adapter": True,
        "retry categories zero": True,
        "standard adapter send": True,
    }
    assert transport._is_bound_to(client, client.config)

    parsed = pilot.invoke(grant)

    assert parsed.content_kind is McpContentKind.STRUCTURED
    assert parsed.output_schema_validated
    assert parsed.structured_content is None
    assert parsed.text_content == ()
    assert not parsed.pagination_ready
    assert "calendar-1" not in repr(parsed)
    assert spy.methods == ["tools/list", "tools/call"]
    call = spy.calls[1]
    assert call["json"] == {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "Calendar.list",
            "arguments": {"limit": 1, "offset": 0},
        },
    }
    assert client._calendar_list_attempted
    assert client._next_request_id == 3


def test_direct_client_transport_bypass_never_returns_raw_response() -> None:
    pilot, client, spy = _pilot()
    authorization = client.authorize_read_request(
        "Calendar.list",
        {"limit": 1, "offset": 0},
    )

    parsed = client._call_calendar_list_pilot(authorization)

    assert parsed.content_kind is McpContentKind.STRUCTURED
    assert parsed.structured_content is None
    assert parsed.text_content == ()
    assert not parsed.pagination_ready
    assert "calendar-1" not in repr(parsed)
    assert spy.methods == ["tools/list", "tools/call"]


def test_validated_state_replacement_is_rejected_before_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    original_validate = client.validate_read_authorization
    validations = 0

    def replace_state_after_final_validation(authorization: Any) -> Any:
        nonlocal validations
        descriptor = original_validate(authorization)
        validations += 1
        if validations == 2:
            client.config = KeystoneConfig(token="offline-replacement-token")
            client._session = requests.Session()
            client._discovery_result = None
        return descriptor

    monkeypatch.setattr(
        client,
        "validate_read_authorization",
        replace_state_after_final_validation,
    )
    with pytest.raises(KeystoneOperationNotAllowedError, match="state changed"):
        pilot.invoke(grant)

    assert spy.methods == ["tools/list"]
    assert not client._calendar_list_attempted
    assert client._next_request_id == 2


def test_state_replacement_after_consumption_cannot_change_request_snapshot() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    original_endpoint = client.config.endpoint
    original_token = client.config.token

    def replace_client_state(request: dict[str, Any]) -> _Response:
        call = spy.calls[-1]
        assert call["url"] == original_endpoint
        assert call["headers"]["Authorization"] == f"Bearer {original_token}"
        client.config = KeystoneConfig.for_hosted_runner(
            base_url="https://offline-replacement.example.test",
            token="offline-replacement-token",
        )
        client._session = requests.Session()
        client._discovery_result = None
        return _Response(_response(_page(), request_id=request["id"]))

    spy.business_response = replace_client_state

    parsed = pilot.invoke(grant)

    assert parsed.structured_content is None
    assert spy.methods == ["tools/list", "tools/call"]
    assert spy.calls[-1]["url"] == original_endpoint


def test_concurrent_invocations_make_one_post_and_only_winner_allocates_id() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    spy.business_entered = threading.Event()
    spy.release_business = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(pilot.invoke, grant)
        try:
            assert spy.business_entered.wait(5)
            second = executor.submit(pilot.invoke, grant)
            with pytest.raises(KeystoneOperationNotAllowedError):
                second.result(timeout=5)
            assert [call["json"]["id"] for call in spy.calls] == [1, 2]
            assert client._calendar_list_attempted
        finally:
            spy.release_business.set()
        parsed = first.result(timeout=5)

    assert parsed.structured_content is None
    assert [call["json"]["id"] for call in spy.calls] == [1, 2]
    assert client._next_request_id == 3


def test_per_client_guard_spans_distinct_pilot_instances() -> None:
    pilot, client, spy = _pilot()
    first_grant = _grant(pilot)
    second_pilot = CalendarListSmokePilot(client)
    second_grant = _grant(second_pilot)

    pilot.invoke(first_grant)
    with pytest.raises(KeystoneOperationNotAllowedError, match="already been consumed"):
        second_pilot.invoke(second_grant)

    assert spy.methods == ["tools/list", "tools/call"]
    assert [call["json"]["id"] for call in spy.calls] == [1, 2]
    assert client._next_request_id == 3


def test_concurrent_distinct_pilots_share_one_failed_client_attempt() -> None:
    first_pilot, client, spy = _pilot()
    first_grant = _grant(first_pilot)
    second_pilot = CalendarListSmokePilot(client)
    second_grant = _grant(second_pilot)
    third_pilot = CalendarListSmokePilot(client)
    third_grant = _grant(third_pilot)
    spy.business_entered = threading.Event()
    spy.release_business = threading.Event()
    spy.business_response = lambda request: requests.ConnectionError(
        "synthetic failure after entry"
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_pilot.invoke, first_grant)
        try:
            assert spy.business_entered.wait(5)
            second = executor.submit(second_pilot.invoke, second_grant)
            with pytest.raises(KeystoneOperationNotAllowedError):
                second.result(timeout=5)
            assert [call["json"]["id"] for call in spy.calls] == [1, 2]
        finally:
            spy.release_business.set()

        with pytest.raises(KeystoneTransportError):
            first.result(timeout=5)

    with pytest.raises(KeystoneOperationNotAllowedError, match="already been consumed"):
        third_pilot.invoke(third_grant)
    assert spy.methods == ["tools/list", "tools/call"]
    assert [call["json"]["id"] for call in spy.calls] == [1, 2]
    assert client._calendar_list_attempted
    assert client._next_request_id == 3


def _failure_response(
    failure: str,
    request: dict[str, Any],
) -> _Response | BaseException:
    request_id = request["id"]
    if failure == "timeout":
        return requests.Timeout("synthetic timeout")
    if failure == "connection":
        return requests.ConnectionError("synthetic connection failure")
    if failure.startswith("http_"):
        return _Response(status_code=int(failure.removeprefix("http_")))
    if failure == "redirect":
        return _Response(status_code=302)
    if failure == "malformed_json":
        return _Response(json_error=ValueError("invalid synthetic JSON"))
    if failure == "rpc_error":
        return _Response({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": "synthetic error"},
        })
    if failure == "malformed_rpc":
        return _Response({
            "jsonrpc": "1.0",
            "id": request_id,
            "result": {},
        })
    if failure == "mismatched_id":
        return _Response({
            "jsonrpc": "2.0",
            "id": request_id + 1,
            "result": {"content": [], "structuredContent": _page()},
        })
    if failure == "both_result_error":
        return _Response({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [], "structuredContent": _page()},
            "error": {"code": -32000, "message": "synthetic error"},
        })
    if failure == "malformed_rpc_error":
        return _Response({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": True, "message": "synthetic error"},
        })
    if failure == "mcp_error":
        return _Response({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "synthetic failure"}],
            },
        })

    result: dict[str, Any] = {"content": [], "structuredContent": _page()}
    if failure == "text_only":
        result = {"content": [{"type": "text", "text": "not resource data"}]}
    elif failure == "unsupported_mixed":
        result["content"] = [
            {"type": "text", "text": "candidate plus text"},
            {"type": "image", "data": "opaque"},
        ]
    elif failure == "unsupported":
        result["content"] = [{"type": "image", "data": "opaque"}]
    elif failure == "missing_content":
        result.pop("content")
    elif failure == "missing_structured":
        result.pop("structuredContent")
    elif failure == "malformed_content":
        result["content"] = None
    elif failure == "unexpected_result":
        result["unexpected"] = True
    elif failure == "schema_mismatch":
        result["structuredContent"] = {"data": [], "total": 0}
    elif failure == "bad_total":
        result["structuredContent"] = _page(data=[{"id": "one"}]) | {"total": 0}
    elif failure == "bad_limit":
        result["structuredContent"] = _page(limit=2)
    elif failure == "bad_offset":
        result["structuredContent"] = _page(offset=1)
    elif failure == "too_many_records":
        result["structuredContent"] = _page(
            data=[{"id": "one"}, {"id": "two"}]
        )
    elif failure == "unexpected_page_field":
        result["structuredContent"]["extra"] = True
    elif failure == "malformed_structured":
        result["structuredContent"] = []
    elif failure == "bad_result_metadata":
        result["_meta"] = []
    elif failure == "nonempty_result_metadata":
        result["_meta"] = {"extension": True}
    elif failure == "unexpected_top_level":
        return _Response({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
            "unexpected": True,
        })
    else:
        raise AssertionError(f"unknown synthetic failure {failure}")
    return _Response({"jsonrpc": "2.0", "id": request_id, "result": result})


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("timeout", KeystoneTimeoutError),
        ("connection", KeystoneTransportError),
        ("http_401", KeystoneAuthenticationError),
        ("http_403", KeystoneHttpError),
        ("http_500", KeystoneHttpError),
        ("redirect", KeystoneHttpError),
        ("malformed_json", KeystoneJsonError),
        ("rpc_error", KeystoneRpcError),
        ("malformed_rpc", KeystoneProtocolError),
        ("mismatched_id", KeystoneProtocolError),
        ("both_result_error", KeystoneProtocolError),
        ("malformed_rpc_error", KeystoneProtocolError),
        ("mcp_error", KeystoneToolError),
        ("text_only", KeystoneProtocolError),
        ("unsupported_mixed", KeystoneProtocolError),
        ("unsupported", KeystoneProtocolError),
        ("missing_content", KeystoneProtocolError),
        ("missing_structured", KeystoneProtocolError),
        ("malformed_content", KeystoneProtocolError),
        ("unexpected_result", KeystoneProtocolError),
        ("schema_mismatch", KeystoneProtocolError),
        ("bad_total", KeystoneProtocolError),
        ("bad_limit", KeystoneProtocolError),
        ("bad_offset", KeystoneProtocolError),
        ("too_many_records", KeystoneProtocolError),
        ("unexpected_page_field", KeystoneProtocolError),
        ("malformed_structured", KeystoneProtocolError),
        ("bad_result_metadata", KeystoneProtocolError),
        ("nonempty_result_metadata", KeystoneProtocolError),
        ("unexpected_top_level", KeystoneProtocolError),
    ],
)
def test_every_transport_or_response_failure_consumes_attempt_without_retry(
    failure: str,
    error: type[Exception],
) -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    spy.business_response = lambda request: _failure_response(failure, request)

    with pytest.raises(error):
        pilot.invoke(grant)
    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.invoke(grant)

    assert client._calendar_list_attempted
    assert spy.methods == ["tools/list", "tools/call"]
    assert sum(method == "tools/call" for method in spy.methods) == 1
    assert [call["json"]["id"] for call in spy.calls] == [1, 2]


def test_pilot_rejects_success_envelope_missing_structured_content() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    spy.business_response = lambda request: _Response({
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {"content": []},
    })

    with pytest.raises(KeystoneProtocolError) as caught:
        pilot.invoke(grant)

    assert str(caught.value) == "Calendar.list success must contain structuredContent."
    assert client._calendar_list_attempted
    with pytest.raises(KeystoneOperationNotAllowedError, match="already been consumed"):
        pilot.invoke(grant)
    assert spy.methods == ["tools/list", "tools/call"]


def test_json_decode_failure_does_not_retain_raw_response_in_exception_chain() -> None:
    pilot, _, spy = _pilot()
    grant = _grant(pilot)
    raw_response = "Bearer offline-token and private-calendar-row"
    decoder_error = ValueError("synthetic decode failure")
    decoder_error.doc = raw_response  # type: ignore[attr-defined]
    decoder_error.pos = 12  # type: ignore[attr-defined]
    spy.business_response = lambda request: _Response(json_error=decoder_error)

    with pytest.raises(KeystoneJsonError) as caught:
        pilot.invoke(grant)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert raw_response not in str(caught.value)
    assert raw_response not in repr(caught.value)
    assert not hasattr(caught.value, "doc")


def test_pilot_fails_closed_when_no_offline_transport_is_configured() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    client._calendar_list_transport = None

    with pytest.raises(KeystoneOperationNotAllowedError, match="not configured"):
        pilot.invoke(grant)

    assert spy.methods == ["tools/list"]
    assert not client._calendar_list_attempted
    assert client._next_request_id == 2


def test_production_transport_owns_https_only_zero_retry_session() -> None:
    pilot, client, _ = _pilot()
    transport = client._calendar_list_transport
    assert isinstance(transport, CalendarListPilotTransport)
    assert client.config.is_approved_pilot_origin

    assert transport._session is not client._session
    assert type(transport._session) is requests.Session
    assert transport._session.trust_env is False
    assert transport._session.auth is None
    assert not transport._session.cookies
    assert not transport._session.proxies
    assert transport._session.hooks == {"response": []}
    assert set(transport._session.adapters) == {"https://"}
    assert transport._session.get_adapter(client.config.endpoint) is transport._adapter
    assert type(transport._adapter) is HTTPAdapter

    retry_policy = transport._adapter.max_retries
    assert type(retry_policy) is Retry
    assert all(
        getattr(retry_policy, category) == 0
        for category in ("total", "connect", "read", "redirect", "status")
    )
    if hasattr(retry_policy, "other"):
        assert retry_policy.other == 0
    assert not retry_policy.status_forcelist
    assert retry_policy.backoff_factor == 0
    assert tuple(inspect.signature(CalendarListPilotTransport).parameters) == ("client",)
    assert tuple(
        inspect.signature(CalendarListPilotTransport.send_once).parameters
    ) == ("self", "authorization")


@pytest.mark.parametrize(
    "mutation",
    [
        "session",
        "alternate_adapter",
        "retry_policy",
        "hooks",
        "environment_proxy",
        "instance_send",
        "adapter_send",
    ],
)
def test_transport_rejects_unexpected_private_configuration(
    mutation: str,
) -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    transport = client._calendar_list_transport

    if mutation == "session":
        class CustomSession(requests.Session):
            pass

        transport._session = CustomSession()
    elif mutation == "alternate_adapter":
        class CustomAdapter(HTTPAdapter):
            pass

        transport._session.mount("https://", CustomAdapter(max_retries=0))
    elif mutation == "retry_policy":
        transport._adapter.max_retries = Retry(
            total=1,
            connect=1,
            read=1,
            redirect=1,
            status=1,
            other=1,
        )
    elif mutation == "hooks":
        transport._session.hooks["response"].append(lambda *args, **kwargs: None)
    elif mutation == "environment_proxy":
        transport._session.trust_env = True
    elif mutation == "instance_send":
        transport._session.send = lambda *args, **kwargs: None
    elif mutation == "adapter_send":
        transport._adapter.send = lambda *args, **kwargs: None

    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.invoke(grant)

    assert spy.methods == ["tools/list"]
    assert not client._calendar_list_attempted
    assert client._next_request_id == 2


def test_transport_rejects_hosted_runner_destination() -> None:
    for base_url in (
        "https://runner.example.test",
        "https://runner.example.test/tenant/keystone",
        "https://class.agentswitch.theschoolofai.in:8443",
    ):
        client = KeystoneClient(
            KeystoneConfig.for_hosted_runner(
                base_url=base_url,
                token="offline-test-token",
            )
        )
        with pytest.raises(KeystoneOperationNotAllowedError):
            CalendarListPilotTransport(client)


def test_transport_rejects_replaced_adapter_send_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    standard_sender = transport_module._STANDARD_HTTP_ADAPTER_SEND
    called = False

    def replacement_sender(
        adapter: HTTPAdapter,
        prepared_request: requests.PreparedRequest,
        **kwargs: Any,
    ) -> requests.Response:
        nonlocal called
        called = True
        return standard_sender(adapter, prepared_request, **kwargs)

    monkeypatch.setattr(transport_module, "_STANDARD_HTTP_ADAPTER_SEND", replacement_sender)

    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.invoke(grant)

    assert not called
    assert spy.methods == ["tools/list"]
    assert not client._calendar_list_attempted


def test_transport_constructor_exposes_only_client_binding_and_zero_retries() -> None:
    pilot, client, _ = _pilot()
    transport = client._calendar_list_transport

    assert transport is not None
    assert tuple(inspect.signature(CalendarListPilotTransport).parameters) == ("client",)
    assert tuple(
        inspect.signature(CalendarListPilotTransport.send_once).parameters
    ) == ("self", "authorization")
    assert transport._session is not client._session
    assert type(transport._session) is requests.Session
    assert transport._session.trust_env is False
    assert transport._session.auth is None
    assert not transport._session.cookies
    assert not transport._session.proxies
    assert transport._session.hooks == {"response": []}
    assert set(transport._session.adapters) == {"https://"}
    assert transport._session.get_adapter(client.config.endpoint) is transport._adapter
    assert type(transport._adapter) is HTTPAdapter

    retry_policy = transport._adapter.max_retries
    assert type(retry_policy) is Retry
    assert all(
        getattr(retry_policy, category) == 0
        for category in ("total", "connect", "read", "redirect", "status")
    )
    if hasattr(retry_policy, "other"):
        assert retry_policy.other == 0
    assert not retry_policy.status_forcelist
    assert retry_policy.backoff_factor == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "custom_session",
        "alternate_adapter",
        "retry_policy",
        "response_hook",
        "environment_proxy",
        "session_send_override",
        "adapter_send_override",
    ],
)
def test_transport_rejects_custom_sessions_adapters_hooks_and_retries(
    mutation: str,
) -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)
    transport = client._calendar_list_transport

    if mutation == "custom_session":
        class CustomSession(requests.Session):
            pass

        transport._session = CustomSession()
    elif mutation == "alternate_adapter":
        class CustomAdapter(HTTPAdapter):
            pass

        transport._session.mount("https://", CustomAdapter(max_retries=0))
    elif mutation == "retry_policy":
        transport._adapter.max_retries = Retry(
            total=1,
            connect=1,
            read=1,
            redirect=1,
            status=1,
            other=1,
        )
    elif mutation == "response_hook":
        transport._session.hooks["response"].append(lambda *args, **kwargs: None)
    elif mutation == "environment_proxy":
        transport._session.trust_env = True
    elif mutation == "session_send_override":
        transport._session.send = lambda *args, **kwargs: None
    elif mutation == "adapter_send_override":
        transport._adapter.send = lambda *args, **kwargs: None

    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.invoke(grant)

    assert spy.methods == ["tools/list"]
    assert not client._calendar_list_attempted
    assert client._next_request_id == 2


@pytest.mark.parametrize(
    "operation",
    [
        "Party.list",
        "CalendarEvent.list",
        "EventAttendee.list",
        "Calendar.create",
        "CalendarEvent.update",
        "endpoint.calendar.free_busy",
        "endpoint.calendar.conference.conflicts",
    ],
)
def test_specialized_transport_and_pilot_reject_other_operations(
    operation: str,
) -> None:
    pilot, client, spy = _pilot()
    arguments = {"limit": 1, "offset": 0}

    with pytest.raises(KeystoneOperationNotAllowedError):
        pilot.authorize(operation, arguments)

    if operation in {"Party.list", "CalendarEvent.list", "EventAttendee.list"}:
        authorization = client.authorize_read_request(operation, arguments)
        with pytest.raises(KeystoneOperationNotAllowedError):
            client._call_calendar_list_pilot(authorization)
    elif operation == "endpoint.calendar.free_busy":
        with pytest.raises(KeystoneProtocolError, match="inputSchema"):
            client.authorize_read_request(operation, arguments)
    else:
        with pytest.raises(KeystoneOperationNotAllowedError):
            client.authorize_read_request(operation, arguments)

    assert spy.methods == ["tools/list"]
    assert not client._calendar_list_attempted


def test_generic_read_methods_still_deny_business_transport() -> None:
    pilot, client, spy = _pilot()
    grant = _grant(pilot)

    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_read("Calendar.list", {"limit": 1, "offset": 0})
    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_authorized_read(grant._client_authorization)

    assert spy.methods == ["tools/list"]
    assert PRODUCTION_READ_ONLY_OPERATIONS == ()


def test_valid_candidate_never_reaches_pagination_adapter_or_goal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harnesses.keystone_harness import calendar_adapter, pagination

    goal_modules = (
        importlib.import_module("calendar_agent.goals.find_30_minutes"),
        importlib.import_module("calendar_agent.goals.move_after_audit"),
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("pilot candidate reached a downstream consumer")

    monkeypatch.setattr(pagination, "collect_resource_pages", fail_if_called)
    monkeypatch.setattr(calendar_adapter, "KeystoneCalendarAdapter", fail_if_called)
    for goal_module in goal_modules:
        monkeypatch.setattr(goal_module, "plan", fail_if_called)

    pilot, _, spy = _pilot()
    parsed = pilot.invoke(_grant(pilot))

    assert parsed.structured_content is None
    assert not parsed.pagination_ready
    assert parsed.text_content == ()
    assert "calendar-1" not in repr(parsed)
    assert spy.methods == ["tools/list", "tools/call"]
