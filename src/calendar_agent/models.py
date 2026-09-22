"""Small typed models shared by the Calendar Agent and harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResultStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentTask:
    goal: str
    instruction: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    mutating: bool = False
    succeeded: bool = True
    error: str | None = None


@dataclass(frozen=True)
class CandidateSlot:
    start_at: str
    end_at: str
    timezone: str | None
    availability_confirmed: bool
    reason: str
    party_id: str | None = None
    calendar_id: str | None = None


@dataclass(frozen=True)
class CalendarEventRef:
    id: str
    title: str
    start_at: str
    end_at: str
    calendar_id: str | None = None
    status: str | None = None
    timezone: str | None = None


@dataclass(frozen=True)
class AuditReference:
    candidates: tuple[CalendarEventRef, ...]
    search_terms: tuple[str, ...]


@dataclass(frozen=True)
class EventMove:
    """Read-only proposal for moving one existing event."""

    event_id: str
    title: str
    original_start_at: str
    original_end_at: str
    proposed_start_at: str
    proposed_end_at: str
    duration_seconds: float
    calendar_id: str | None = None
    timezone: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    outcome_ok: bool
    integrity_ok: bool
    details: tuple[str, ...] = ()


@dataclass
class AgentResult:
    goal: str
    status: ResultStatus
    summary: str
    ambiguities: list[str] = field(default_factory=list)
    candidate_slots: list[CandidateSlot] = field(default_factory=list)
    audit_reference: AuditReference | None = None
    rescheduling_plan: list[EventMove] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    claimed_outcome: dict[str, Any] = field(default_factory=dict)


class MutationDisabledError(RuntimeError):
    """Raised when a goal attempts a mutation before mutation execution is enabled."""