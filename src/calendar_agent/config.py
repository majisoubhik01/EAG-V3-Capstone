"""Configuration for authenticated AgentSwitch access."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://agentswitch.theschoolofai.in"


class ConfigurationError(ValueError):
    """Raised when the local AgentSwitch configuration is incomplete."""


@dataclass(frozen=True)
class AgentSwitchConfig:
    """Non-secret connection settings and bearer-token authentication."""

    base_url: str = DEFAULT_BASE_URL
    token: str | None = None
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "AgentSwitchConfig":
        """Build configuration from environment variables without reading files."""

        token = os.getenv("AGENTSWITCH_TOKEN")
        try:
            timeout_seconds = float(os.getenv("AGENTSWITCH_TIMEOUT_SECONDS", "10"))
        except ValueError as exc:
            raise ConfigurationError(
                "AGENTSWITCH_TIMEOUT_SECONDS must be a positive number"
            ) from exc

        config = cls(
            base_url=os.getenv("AGENTSWITCH_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            token=token,
            timeout_seconds=timeout_seconds,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate settings needed to make an authenticated request."""

        if not self.base_url:
            raise ConfigurationError("AGENTSWITCH_BASE_URL cannot be empty")
        if not self.token:
            raise ConfigurationError("Set AGENTSWITCH_TOKEN for authenticated access")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/mcp"

    def auth_headers(self) -> dict[str, str]:
        """Return request headers while keeping auth out of the client API."""

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers