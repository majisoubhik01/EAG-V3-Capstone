"""Seat 19 Calendar Agent integration components."""

from .agentswitch_client import (
    AgentSwitchClient,
    AgentSwitchError,
    AgentSwitchHttpError,
    AgentSwitchProtocolError,
    AgentSwitchRpcError,
    AgentSwitchTimeoutError,
    AgentSwitchToolError,
)
from .calendar_events import CalendarEventClient
from .config import AgentSwitchConfig, ConfigurationError
from .agent import CalendarAgent
from .models import (
    AgentResult,
    AgentTask,
    AuditReference,
    CandidateSlot,
    CalendarEventRef,
    EventMove,
    MutationDisabledError,
    ResultStatus,
    ToolCallRecord,
    VerificationResult,
)
from .tools import AgentSwitchCalendarTools, CalendarTool

__all__ = [
    "AgentSwitchClient",
    "AgentSwitchConfig",
    "AgentSwitchError",
    "AgentSwitchHttpError",
    "AgentSwitchProtocolError",
    "AgentSwitchRpcError",
    "AgentSwitchTimeoutError",
    "AgentSwitchToolError",
    "AgentSwitchCalendarTools",
    "CalendarAgent",
    "CalendarEventClient",
    "CalendarEventRef",
    "CalendarTool",
    "AgentResult",
    "AgentTask",
    "AuditReference",
    "CandidateSlot",
    "ConfigurationError",
    "EventMove",
    "MutationDisabledError",
    "ResultStatus",
    "ToolCallRecord",
    "VerificationResult",
]