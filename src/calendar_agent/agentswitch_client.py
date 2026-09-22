"""Small JSON-RPC MCP client for the AgentSwitch service."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any

import requests

from .config import AgentSwitchConfig


class AgentSwitchError(RuntimeError):
    """Base class for safe, expected AgentSwitch client failures."""


class AgentSwitchTimeoutError(AgentSwitchError):
    """Raised when AgentSwitch does not respond before the configured timeout."""


class AgentSwitchHttpError(AgentSwitchError):
    """Raised when AgentSwitch returns a non-success HTTP status."""


class AgentSwitchProtocolError(AgentSwitchError):
    """Raised when the response is not a valid JSON-RPC/MCP result."""


class AgentSwitchRpcError(AgentSwitchError):
    """Raised when the JSON-RPC response contains an error object."""


class AgentSwitchToolError(AgentSwitchError):
    """Raised when an MCP tool reports ``isError``."""


@dataclass
class AgentSwitchClient:
    """Invoke AgentSwitch MCP tools without exposing HTTP details to agents."""

    config: AgentSwitchConfig
    session: requests.Session | None = None
    _request_ids: Any = field(default_factory=lambda: count(1), init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        if self.session is None:
            self.session = requests.Session()

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call one MCP tool and return its validated result object."""

        if not name.strip():
            raise ValueError("tool name cannot be empty")
        if arguments is not None and not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a dictionary")

        request_id = next(self._request_ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }

        try:
            response = self.session.post(
                self.config.endpoint,
                headers=self.config.auth_headers(),
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise AgentSwitchTimeoutError("AgentSwitch request timed out") from exc
        except requests.RequestException as exc:
            raise AgentSwitchError("AgentSwitch request failed") from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise AgentSwitchHttpError(
                f"AgentSwitch returned HTTP {response.status_code}"
            ) from exc

        try:
            message = response.json()
        except ValueError as exc:
            raise AgentSwitchProtocolError("AgentSwitch returned invalid JSON") from exc

        return self._parse_response(message, request_id)

    @staticmethod
    def _parse_response(message: Any, request_id: int) -> dict[str, Any]:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise AgentSwitchProtocolError("Response is not a JSON-RPC 2.0 object")
        if message.get("id") != request_id:
            raise AgentSwitchProtocolError("Response id does not match the request")

        if "error" in message:
            error = message["error"]
            if not isinstance(error, dict):
                raise AgentSwitchProtocolError("JSON-RPC error is malformed")
            code = error.get("code", "unknown")
            detail = error.get("message", "unknown error")
            raise AgentSwitchRpcError(f"AgentSwitch JSON-RPC error {code}: {detail}")

        result = message.get("result")
        if not isinstance(result, dict):
            raise AgentSwitchProtocolError("JSON-RPC result is missing or malformed")
        if result.get("isError") is True:
            raise AgentSwitchToolError("AgentSwitch MCP tool reported an error")
        return result