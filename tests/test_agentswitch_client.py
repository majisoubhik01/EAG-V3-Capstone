from __future__ import annotations

import pytest
import requests

from calendar_agent import (
    AgentSwitchClient,
    AgentSwitchConfig,
    AgentSwitchHttpError,
    AgentSwitchProtocolError,
    AgentSwitchRpcError,
    AgentSwitchTimeoutError,
    AgentSwitchToolError,
    CalendarEventClient,
)


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_client(response: FakeResponse | Exception) -> tuple[AgentSwitchClient, FakeSession]:
    session = FakeSession(response)
    client = AgentSwitchClient(
        AgentSwitchConfig(
            base_url="https://example.test",
            token="unit-test-token",
        ),
        session=session,  # type: ignore[arg-type]
    )
    return client, session


def test_calendar_event_list_builds_mcp_request_and_returns_result() -> None:
    client, session = make_client(
        FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"isError": False, "content": []},
            }
        )
    )

    result = CalendarEventClient(client).list(limit=1)

    assert result == {"isError": False, "content": []}
    request = session.calls[0]
    assert request["url"] == "https://example.test/api/mcp"
    assert request["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer unit-test-token",
    }
    assert request["json"] == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "CalendarEvent.list", "arguments": {"limit": 1}},
    }


def test_json_rpc_error_is_raised() -> None:
    client, _ = make_client(
        FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32602, "message": "invalid arguments"},
            }
        )
    )

    with pytest.raises(AgentSwitchRpcError, match="-32602"):
        client.call_tool("CalendarEvent.list", {"limit": 1})


def test_mcp_tool_error_is_raised() -> None:
    client, _ = make_client(
        FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"isError": True, "content": []},
            }
        )
    )

    with pytest.raises(AgentSwitchToolError):
        client.call_tool("CalendarEvent.list", {"limit": 1})


@pytest.mark.parametrize(
    "response",
    [
        {"jsonrpc": "1.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 99, "result": {}},
        {"jsonrpc": "2.0", "id": 1, "result": []},
    ],
)
def test_malformed_response_is_rejected(response: object) -> None:
    client, _ = make_client(FakeResponse(response))

    with pytest.raises(AgentSwitchProtocolError):
        client.call_tool("CalendarEvent.list", {"limit": 1})


def test_http_error_is_raised_without_response_body() -> None:
    client, _ = make_client(FakeResponse({"ignored": True}, status_code=503))

    with pytest.raises(AgentSwitchHttpError, match="503"):
        client.call_tool("CalendarEvent.list", {"limit": 1})


def test_timeout_is_raised() -> None:
    client, _ = make_client(requests.Timeout())

    with pytest.raises(AgentSwitchTimeoutError):
        client.call_tool("CalendarEvent.list", {"limit": 1})