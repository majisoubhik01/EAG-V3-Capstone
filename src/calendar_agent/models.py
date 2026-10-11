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


class RiskLevel(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    POSTING = "POSTING"
    EXTERNAL = "EXTERNAL"
    DESTRUCTIVE = "DESTRUCTIVE"


class AuthorityFailureKind(str, Enum):
    TOOL_MISSING = "tool_missing"
    AGENT_UNAUTHORIZED = "agent_unauthorized"
    PROVIDER_FAILURE = "provider_failure"
    TOOL_PERMISSION_DENIED = "tool_permission_denied"


_ATTENDEE_RESPONSES = {"accepted", "declined", "tentative", "needs_action"}
_ATTENDEE_ROLES = {"required", "optional", "organizer", "chair"}


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
class PartyRecord:
    id: str
    name: str
    email: str
    job_title: str = ""


@dataclass(frozen=True)
class PartyPage:
    records: tuple[PartyRecord, ...]
    page: int
    page_size: int
    total_count: int
    has_more: bool


@dataclass(frozen=True)
class EventAttendee:
    event_id: str
    party_id: str | None
    email: str
    role: str
    response_status: str

    def __post_init__(self) -> None:
        if self.role not in _ATTENDEE_ROLES:
            raise ValueError(f"unsupported attendee role: {self.role}")
        if self.response_status not in _ATTENDEE_RESPONSES:
            raise ValueError(f"unsupported attendee response: {self.response_status}")

    @property
    def required(self) -> bool:
        return self.role in {"required", "organizer", "chair"}


@dataclass(frozen=True)
class FreeBusyInterval:
    start_at: str
    end_at: str
    source: str = "calendar"


@dataclass(frozen=True)
class MeetingBooking:
    id: str
    event_id: str | None
    title: str
    start_at: str
    end_at: str
    attendee_count: int | None = None
    status: str = "confirmed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulingWorkflowAction:
    action: str
    risk: RiskLevel
    enabled: bool = True


@dataclass(frozen=True)
class CalendarSubscription:
    id: str
    calendar_id: str
    status: str
    last_sync_at: str | None
    feed_error: str | None = None


@dataclass(frozen=True)
class AgentToolPolicy:
    read_only: bool = True
    risk_mode: RiskLevel = RiskLevel.READ
    write_domains: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    denied_domains: tuple[str, ...] = ()
    allowed_entities: tuple[str, ...] = ()
    denied_entities: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    denied_actions: tuple[str, ...] = ()
    max_records_per_query: int = 100

    def __post_init__(self) -> None:
        if self.max_records_per_query <= 0:
            raise ValueError("max_records_per_query must be positive")

    def allows(
        self,
        *,
        action: str,
        domain: str,
        entity: str,
        risk: RiskLevel,
    ) -> bool:
        if action in self.denied_actions or domain in self.denied_domains or entity in self.denied_entities:
            return False
        if risk is not RiskLevel.READ and self.read_only:
            return False
        if risk is not RiskLevel.READ and risk.value not in {self.risk_mode.value}:
            return False
        if self.allowed_actions and action not in self.allowed_actions:
            return False
        if self.allowed_domains and domain not in self.allowed_domains:
            return False
        if self.allowed_entities and entity not in self.allowed_entities:
            return False
        if risk is not RiskLevel.READ and domain not in self.write_domains:
            return False
        return True


@dataclass(frozen=True)
class AuthorityFailure:
    kind: AuthorityFailureKind
    message: str
    tool: str | None = None
    provider_status: int | None = None


@dataclass(frozen=True)
class AgentMessage:
    message_id: str
    role: str
    content: str
    step_id: str | None = None


@dataclass(frozen=True)
class AgentJobStep:
    step_id: str
    action: str
    tool_input: dict[str, Any]
    status: str
    tool_output: Any = None
    error: str | None = None
    authority_failure: AuthorityFailure | None = None
    retry_count: int = 0
    latency_ms: float | None = None
    token_usage: dict[str, int] | None = None


@dataclass(frozen=True)
class AgentJob:
    job_id: str
    status: str
    steps: tuple[AgentJobStep, ...] = ()
    messages: tuple[AgentMessage, ...] = ()
    evaluation_decision: str | None = None
    approval_state: str | None = None
    retry_count: int = 0
    token_usage: dict[str, int] | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class ProviderFailure:
    status_code: int
    error: str
    invocation_attempted: bool = True
    completion_occurred: bool = False
    tool_calls: int = 0
    resource_reads: int = 0
    mutations: int = 0
    token_usage: dict[str, int] | None = None
    cost: float | None = None


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