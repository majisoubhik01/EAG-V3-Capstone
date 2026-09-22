"""Read-only live AgentSwitch smoke test for Seat 19."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calendar_agent import AgentSwitchClient, AgentSwitchConfig, CalendarEventClient


def main() -> None:
    config = AgentSwitchConfig.from_env()
    result = CalendarEventClient(AgentSwitchClient(config)).list(limit=1)

    content = result.get("content")
    content_items = len(content) if isinstance(content, list) else 0
    print("AgentSwitch read-only smoke test passed")
    print("tool=CalendarEvent.list")
    print(f"result_keys={sorted(result)}")
    print(f"content_items={content_items}")
    print(f"is_error={result.get('isError', False)}")


if __name__ == "__main__":
    main()