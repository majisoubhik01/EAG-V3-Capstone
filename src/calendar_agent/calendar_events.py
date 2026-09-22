"""Read-only CalendarEvent tool wrappers."""

from __future__ import annotations

from typing import Any

from .agentswitch_client import AgentSwitchClient


class CalendarEventClient:
    """Expose the read-only CalendarEvent operations used by the Calendar Agent."""

    def __init__(self, client: AgentSwitchClient) -> None:
        self._client = client

    def list(self, *, limit: int = 1) -> dict[str, Any]:
        """List a small number of calendar events without mutating AgentSwitch."""

        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        return self._client.call_tool("CalendarEvent.list", {"limit": limit})