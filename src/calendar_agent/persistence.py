"""Versioned, secret-safe persistence for runtime evidence."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .models import (
    AgentResult,
    AgentTask,
    AuditReference,
    CalendarEventRef,
    CandidateSlot,
    EventMove,
    ResultStatus,
    ToolCallRecord,
)
from .runtime import (
    Clarify,
    Fail,
    RawRun,
    SelectGoal,
    TerminationReason,
)


JOURNAL_VERSION = 1
MANIFEST_VERSION = 1
ARTIFACT_TYPE = "calendar_agent.raw_run"
_TYPE_KEY = "__calendar_agent_type__"
_REDACTED = "[REDACTED]"
_BEARER_PATTERN = re.compile(r"(\bbearer\s+)[^\s,;]+", re.IGNORECASE)
_SECRET_KEY_PARTS = (
    "token",
    "apikey",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "secret",
    "credential",
    "bearer",
)


class ArtifactError(ValueError):
    """Base error for invalid or unsafe persisted artifacts."""


class MalformedArtifactError(ArtifactError):
    """Raised when an artifact is not valid JSON or has the wrong shape."""


class MissingArtifactFieldError(MalformedArtifactError):
    """Raised when a required artifact field is absent."""


class InvalidArtifactValueError(MalformedArtifactError):
    """Raised when an artifact field has an invalid value or type."""


class IncompatibleArtifactVersionError(ArtifactError):
    """Raised when an artifact was written by an unsupported version."""


class ArtifactSerializationError(ArtifactError):
    """Raised when runtime evidence cannot be represented safely as JSON."""


@dataclass(frozen=True)
class EvaluationManifest:
    """Metadata describing how a persisted run may be evaluated."""

    manifest_version: int = MANIFEST_VERSION
    task_identifier: str | None = None
    task_version: str | None = None
    model_identity: str | None = None
    model_configuration: Mapping[str, Any] | None = None
    prompt_ref: str | None = None
    prompt_version: str | None = None
    policy_ref: str | None = None
    policy_version: str | None = None
    seed: int | str | None = None
    timeout_seconds: float | None = None
    environment: str | Mapping[str, Any] | None = None
    tool_set_identity: str | None = None
    scorer_version: str | None = None
    run_id: str | None = None

    @classmethod
    def for_run(cls, raw_run: RawRun) -> EvaluationManifest:
        task_identifier = raw_run.task.goal if raw_run.task is not None else None
        prompt_ref = raw_run.policy_metadata.get("prompt_ref")
        if not isinstance(prompt_ref, str):
            prompt_ref = None
        return cls(
            task_identifier=task_identifier,
            prompt_ref=prompt_ref,
            run_id=raw_run.run_id,
        )


@dataclass(frozen=True)
class PersistedRun:
    """A loaded raw run and the manifest persisted alongside it."""

    raw_run: RawRun
    manifest: EvaluationManifest


def _is_secret_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _sanitize(value: Any, *, key: object | None = None) -> Any:
    if _is_secret_key(key):
        return _REDACTED
    if isinstance(value, str):
        return _BEARER_PATTERN.sub(r"\1" + _REDACTED, value)
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item_value, key=item_key) for item_key, item_value in value.items()}
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _sanitize(getattr(value, item.name), key=item.name)
            for item in fields(value)
        }
    return value


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return {_TYPE_KEY: "datetime", "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {_TYPE_KEY: "timedelta", "seconds": value.total_seconds()}
    if isinstance(value, Enum):
        return {_TYPE_KEY: "enum", "value": value.value}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ArtifactSerializationError("JSON object keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [_encode(item) for item in value]}
    if is_dataclass(value):
        return {
            _TYPE_KEY: "dataclass",
            "class": type(value).__name__,
            "fields": {item.name: _encode(getattr(value, item.name)) for item in fields(value)},
        }
    raise ArtifactSerializationError(f"Unsupported evidence value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    marker = value.get(_TYPE_KEY)
    if marker == "datetime":
        raw_value = value.get("value")
        if not isinstance(raw_value, str):
            raise InvalidArtifactValueError("datetime value must be a string")
        try:
            return datetime.fromisoformat(raw_value)
        except ValueError as exc:
            raise InvalidArtifactValueError("invalid datetime value") from exc
    if marker == "timedelta":
        seconds = value.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise InvalidArtifactValueError("timedelta seconds must be numeric")
        return timedelta(seconds=seconds)
    if marker == "enum":
        return value.get("value")
    if marker == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise InvalidArtifactValueError("tuple items must be a list")
        return tuple(_decode(item) for item in items)
    if marker == "dataclass":
        fields_value = value.get("fields")
        if not isinstance(fields_value, Mapping):
            raise InvalidArtifactValueError("dataclass fields must be an object")
        return {str(key): _decode(item) for key, item in fields_value.items()}
    if marker is not None:
        raise InvalidArtifactValueError(f"unknown encoded value type: {marker!r}")
    return {str(key): _decode(item) for key, item in value.items()}


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise MissingArtifactFieldError(f"Missing required field: {key}")
    return data[key]


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidArtifactValueError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise InvalidArtifactValueError(f"{name} must be a string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise InvalidArtifactValueError(f"{name} must be a string or null")
    return value


def _task_to_data(task: AgentTask | None) -> dict[str, Any] | None:
    if task is None:
        return None
    return {"goal": task.goal, "instruction": task.instruction, "context": task.context}


def _task_from_data(value: Any) -> AgentTask | None:
    if value is None:
        return None
    data = _object(value, "task")
    context = _object(_decode(_required(data, "context")), "task.context")
    return AgentTask(
        goal=_string(_required(data, "goal"), "task.goal"),
        instruction=_string(_required(data, "instruction"), "task.instruction"),
        context=dict(context),
    )


def _tool_call_to_data(call: ToolCallRecord) -> dict[str, Any]:
    return {
        "tool": call.tool,
        "arguments": call.arguments,
        "mutating": call.mutating,
        "succeeded": call.succeeded,
        "error": call.error,
    }


def _tool_call_from_data(value: Any, name: str = "tool call") -> ToolCallRecord:
    data = _object(value, name)
    arguments = _object(_decode(_required(data, "arguments")), f"{name}.arguments")
    mutating = _required(data, "mutating")
    succeeded = _required(data, "succeeded")
    if not isinstance(mutating, bool) or not isinstance(succeeded, bool):
        raise InvalidArtifactValueError(f"{name} flags must be boolean")
    error = _optional_string(_required(data, "error"), f"{name}.error")
    return ToolCallRecord(
        tool=_string(_required(data, "tool"), f"{name}.tool"),
        arguments=dict(arguments),
        mutating=mutating,
        succeeded=succeeded,
        error=error,
    )


def _candidate_to_data(slot: CandidateSlot) -> dict[str, Any]:
    return {
        "start_at": slot.start_at,
        "end_at": slot.end_at,
        "timezone": slot.timezone,
        "availability_confirmed": slot.availability_confirmed,
        "reason": slot.reason,
        "party_id": slot.party_id,
        "calendar_id": slot.calendar_id,
    }


def _candidate_from_data(value: Any) -> CandidateSlot:
    data = _object(value, "candidate slot")
    confirmed = _required(data, "availability_confirmed")
    if not isinstance(confirmed, bool):
        raise InvalidArtifactValueError("candidate slot availability_confirmed must be boolean")
    return CandidateSlot(
        start_at=_string(_required(data, "start_at"), "candidate slot.start_at"),
        end_at=_string(_required(data, "end_at"), "candidate slot.end_at"),
        timezone=_optional_string(_required(data, "timezone"), "candidate slot.timezone"),
        availability_confirmed=confirmed,
        reason=_string(_required(data, "reason"), "candidate slot.reason"),
        party_id=_optional_string(_required(data, "party_id"), "candidate slot.party_id"),
        calendar_id=_optional_string(_required(data, "calendar_id"), "candidate slot.calendar_id"),
    )


def _event_ref_to_data(reference: CalendarEventRef) -> dict[str, Any]:
    return {
        "id": reference.id,
        "title": reference.title,
        "start_at": reference.start_at,
        "end_at": reference.end_at,
        "calendar_id": reference.calendar_id,
        "status": reference.status,
        "timezone": reference.timezone,
    }


def _event_ref_from_data(value: Any) -> CalendarEventRef:
    data = _object(value, "calendar event reference")
    return CalendarEventRef(
        id=_string(_required(data, "id"), "event reference.id"),
        title=_string(_required(data, "title"), "event reference.title"),
        start_at=_string(_required(data, "start_at"), "event reference.start_at"),
        end_at=_string(_required(data, "end_at"), "event reference.end_at"),
        calendar_id=_optional_string(_required(data, "calendar_id"), "event reference.calendar_id"),
        status=_optional_string(_required(data, "status"), "event reference.status"),
        timezone=_optional_string(_required(data, "timezone"), "event reference.timezone"),
    )


def _audit_to_data(reference: AuditReference | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    return {
        "candidates": [_event_ref_to_data(item) for item in reference.candidates],
        "search_terms": list(reference.search_terms),
    }


def _audit_from_data(value: Any) -> AuditReference | None:
    if value is None:
        return None
    data = _object(value, "audit reference")
    candidates = _required(data, "candidates")
    search_terms = _required(data, "search_terms")
    if not isinstance(candidates, list) or not isinstance(search_terms, list):
        raise InvalidArtifactValueError("audit reference collections must be lists")
    return AuditReference(
        candidates=tuple(_event_ref_from_data(item) for item in candidates),
        search_terms=tuple(_string(item, "audit search term") for item in search_terms),
    )


def _move_to_data(move: EventMove) -> dict[str, Any]:
    return {
        "event_id": move.event_id,
        "title": move.title,
        "original_start_at": move.original_start_at,
        "original_end_at": move.original_end_at,
        "proposed_start_at": move.proposed_start_at,
        "proposed_end_at": move.proposed_end_at,
        "duration_seconds": move.duration_seconds,
        "calendar_id": move.calendar_id,
        "timezone": move.timezone,
    }


def _move_from_data(value: Any) -> EventMove:
    data = _object(value, "event move")
    duration = _required(data, "duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise InvalidArtifactValueError("event move duration_seconds must be numeric")
    return EventMove(
        event_id=_string(_required(data, "event_id"), "event move.event_id"),
        title=_string(_required(data, "title"), "event move.title"),
        original_start_at=_string(_required(data, "original_start_at"), "event move.original_start_at"),
        original_end_at=_string(_required(data, "original_end_at"), "event move.original_end_at"),
        proposed_start_at=_string(_required(data, "proposed_start_at"), "event move.proposed_start_at"),
        proposed_end_at=_string(_required(data, "proposed_end_at"), "event move.proposed_end_at"),
        duration_seconds=float(duration),
        calendar_id=_optional_string(_required(data, "calendar_id"), "event move.calendar_id"),
        timezone=_optional_string(_required(data, "timezone"), "event move.timezone"),
    )


def _result_to_data(result: AgentResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "goal": result.goal,
        "status": result.status.value,
        "summary": result.summary,
        "ambiguities": result.ambiguities,
        "candidate_slots": [_candidate_to_data(item) for item in result.candidate_slots],
        "audit_reference": _audit_to_data(result.audit_reference),
        "rescheduling_plan": [_move_to_data(item) for item in result.rescheduling_plan],
        "tool_calls": [_tool_call_to_data(item) for item in result.tool_calls],
        "claimed_outcome": result.claimed_outcome,
    }


def _result_from_data(value: Any) -> AgentResult | None:
    if value is None:
        return None
    data = _object(value, "result")
    raw_status = _string(_required(data, "status"), "result.status")
    try:
        status = ResultStatus(raw_status)
    except ValueError as exc:
        raise InvalidArtifactValueError(f"invalid result.status: {raw_status!r}") from exc
    ambiguities = _required(data, "ambiguities")
    candidate_slots = _required(data, "candidate_slots")
    rescheduling_plan = _required(data, "rescheduling_plan")
    tool_calls = _required(data, "tool_calls")
    claimed_outcome = _object(_decode(_required(data, "claimed_outcome")), "result.claimed_outcome")
    if not all(isinstance(items, list) for items in (ambiguities, candidate_slots, rescheduling_plan, tool_calls)):
        raise InvalidArtifactValueError("result collections must be lists")
    return AgentResult(
        goal=_string(_required(data, "goal"), "result.goal"),
        status=status,
        summary=_string(_required(data, "summary"), "result.summary"),
        ambiguities=[_string(item, "result ambiguity") for item in ambiguities],
        candidate_slots=[_candidate_from_data(item) for item in candidate_slots],
        audit_reference=_audit_from_data(_required(data, "audit_reference")),
        rescheduling_plan=[_move_from_data(item) for item in rescheduling_plan],
        tool_calls=[_tool_call_from_data(item, "result tool call") for item in tool_calls],
        claimed_outcome=dict(claimed_outcome),
    )


def _decision_to_data(decision: object) -> dict[str, Any]:
    if isinstance(decision, SelectGoal):
        return {
            "type": "SelectGoal",
            "goal": decision.goal,
            "instruction": decision.instruction,
            "context": decision.context,
        }
    if isinstance(decision, Clarify):
        return {"type": "Clarify", "reason": decision.reason}
    if isinstance(decision, Fail):
        return {"type": "Fail", "reason": decision.reason}
    return {"type": "raw", "value": decision}


def _decision_from_data(value: Any) -> object:
    data = _object(value, "model decision")
    decision_type = _string(_required(data, "type"), "model decision.type")
    if decision_type == "SelectGoal":
        context = _object(_decode(_required(data, "context")), "decision.context")
        return SelectGoal(
            goal=_string(_required(data, "goal"), "decision.goal"),
            instruction=_string(_required(data, "instruction"), "decision.instruction"),
            context=dict(context),
        )
    if decision_type == "Clarify":
        return Clarify(_string(_required(data, "reason"), "decision.reason"))
    if decision_type == "Fail":
        return Fail(_string(_required(data, "reason"), "decision.reason"))
    if decision_type == "raw":
        return _decode(_required(data, "value"))
    raise InvalidArtifactValueError(f"invalid model decision type: {decision_type!r}")


def _raw_run_to_data(raw_run: RawRun) -> dict[str, Any]:
    return {
        "run_id": raw_run.run_id,
        "user_instruction": raw_run.user_instruction,
        "supplied_context": raw_run.supplied_context,
        "policy_metadata": raw_run.policy_metadata,
        "model_decisions": [_decision_to_data(item) for item in raw_run.model_decisions],
        "task": _task_to_data(raw_run.task),
        "result": _result_to_data(raw_run.result),
        "tool_trace": [_tool_call_to_data(item) for item in raw_run.tool_trace],
        "error": raw_run.error,
        "steps_used": raw_run.steps_used,
        "retries_used": raw_run.retries_used,
        "termination_reason": raw_run.termination_reason.value,
    }


def _raw_run_from_data(value: Any) -> RawRun:
    data = _object(value, "raw_run")
    termination_value = _string(_required(data, "termination_reason"), "raw_run.termination_reason")
    try:
        termination_reason = TerminationReason(termination_value)
    except ValueError as exc:
        raise InvalidArtifactValueError(f"invalid raw_run.termination_reason: {termination_value!r}") from exc
    supplied_context = _object(_decode(_required(data, "supplied_context")), "raw_run.supplied_context")
    policy_metadata = _object(_decode(_required(data, "policy_metadata")), "raw_run.policy_metadata")
    decisions = _required(data, "model_decisions")
    tool_trace = _required(data, "tool_trace")
    if not isinstance(decisions, list) or not isinstance(tool_trace, list):
        raise InvalidArtifactValueError("raw_run decisions and tool_trace must be lists")
    steps_used = _required(data, "steps_used")
    retries_used = _required(data, "retries_used")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (steps_used, retries_used)):
        raise InvalidArtifactValueError("raw_run step counts must be non-negative integers")
    error = _optional_string(_required(data, "error"), "raw_run.error")
    return RawRun(
        run_id=_string(_required(data, "run_id"), "raw_run.run_id"),
        user_instruction=_string(_required(data, "user_instruction"), "raw_run.user_instruction"),
        supplied_context=dict(supplied_context),
        policy_metadata=dict(policy_metadata),
        model_decisions=tuple(_decision_from_data(item) for item in decisions),
        task=_task_from_data(_required(data, "task")),
        result=_result_from_data(_required(data, "result")),
        tool_trace=tuple(_tool_call_from_data(item) for item in tool_trace),
        error=error,
        steps_used=steps_used,
        retries_used=retries_used,
        termination_reason=termination_reason,
    )


def _manifest_to_data(manifest: EvaluationManifest) -> dict[str, Any]:
    return {item.name: getattr(manifest, item.name) for item in fields(manifest)}


def _manifest_from_data(value: Any) -> EvaluationManifest:
    data = _object(value, "evaluation_manifest")
    version = _required(data, "manifest_version")
    if version != MANIFEST_VERSION:
        raise IncompatibleArtifactVersionError(
            f"Unsupported evaluation manifest version: {version!r}"
        )
    model_configuration = _decode(_required(data, "model_configuration"))
    if model_configuration is not None and not isinstance(model_configuration, Mapping):
        raise InvalidArtifactValueError("manifest model_configuration must be an object or null")
    environment = _decode(_required(data, "environment"))
    if environment is not None and not isinstance(environment, (str, Mapping)):
        raise InvalidArtifactValueError("manifest environment must be a string, object, or null")
    timeout = _required(data, "timeout_seconds")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))):
        raise InvalidArtifactValueError("manifest timeout_seconds must be numeric or null")
    return EvaluationManifest(
        manifest_version=version,
        task_identifier=_optional_string(_required(data, "task_identifier"), "manifest.task_identifier"),
        task_version=_optional_string(_required(data, "task_version"), "manifest.task_version"),
        model_identity=_optional_string(_required(data, "model_identity"), "manifest.model_identity"),
        model_configuration=model_configuration,
        prompt_ref=_optional_string(_required(data, "prompt_ref"), "manifest.prompt_ref"),
        prompt_version=_optional_string(_required(data, "prompt_version"), "manifest.prompt_version"),
        policy_ref=_optional_string(_required(data, "policy_ref"), "manifest.policy_ref"),
        policy_version=_optional_string(_required(data, "policy_version"), "manifest.policy_version"),
        seed=_required(data, "seed"),
        timeout_seconds=float(timeout) if timeout is not None else None,
        environment=environment,
        tool_set_identity=_optional_string(_required(data, "tool_set_identity"), "manifest.tool_set_identity"),
        scorer_version=_optional_string(_required(data, "scorer_version"), "manifest.scorer_version"),
        run_id=_optional_string(_required(data, "run_id"), "manifest.run_id"),
    )


def _resolved_manifest(raw_run: RawRun, manifest: EvaluationManifest | None) -> EvaluationManifest:
    resolved_manifest = manifest or EvaluationManifest.for_run(raw_run)
    if resolved_manifest.run_id is None:
        resolved_manifest = replace(resolved_manifest, run_id=raw_run.run_id)
    if resolved_manifest.run_id != raw_run.run_id:
        raise ArtifactSerializationError("manifest.run_id must match raw_run.run_id")
    return resolved_manifest


def _document(raw_run: RawRun, manifest: EvaluationManifest | None) -> dict[str, Any]:
    resolved_manifest = _resolved_manifest(raw_run, manifest)
    raw_document = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": JOURNAL_VERSION,
        "raw_run": _raw_run_to_data(raw_run),
        "evaluation_manifest": _manifest_to_data(resolved_manifest),
    }
    return _encode(_sanitize(raw_document))


def _load_document(document: Any) -> PersistedRun:
    root = _object(document, "artifact")
    if _required(root, "artifact_type") != ARTIFACT_TYPE:
        raise InvalidArtifactValueError("invalid artifact_type")
    version = _required(root, "artifact_version")
    if version != JOURNAL_VERSION:
        raise IncompatibleArtifactVersionError(f"Unsupported journal version: {version!r}")
    raw_run = _raw_run_from_data(_required(root, "raw_run"))
    manifest = _manifest_from_data(_required(root, "evaluation_manifest"))
    if manifest.run_id is not None and manifest.run_id != raw_run.run_id:
        raise InvalidArtifactValueError("manifest.run_id does not match raw_run.run_id")
    return PersistedRun(raw_run=raw_run, manifest=manifest)


class RawRunJournal:
    """Read and write one versioned JSON journal artifact."""

    @staticmethod
    def dumps(raw_run: RawRun, manifest: EvaluationManifest | None = None) -> str:
        try:
            return json.dumps(_document(raw_run, manifest), sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactSerializationError("raw run could not be serialized") from exc

    @staticmethod
    def loads(serialized: str) -> PersistedRun:
        try:
            document = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as exc:
            raise MalformedArtifactError("artifact is not valid JSON") from exc
        return _load_document(document)

    @staticmethod
    def write(
        destination: str | Path,
        raw_run: RawRun,
        manifest: EvaluationManifest | None = None,
    ) -> PersistedRun:
        path = Path(destination)
        try:
            path.write_text(RawRunJournal.dumps(raw_run, manifest), encoding="utf-8")
        except OSError as exc:
            raise ArtifactError(f"could not write artifact: {path}") from exc
        return PersistedRun(raw_run=raw_run, manifest=_resolved_manifest(raw_run, manifest))

    @staticmethod
    def load(source: str | Path) -> PersistedRun:
        path = Path(source)
        try:
            serialized = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ArtifactError(f"could not read artifact: {path}") from exc
        return RawRunJournal.loads(serialized)


def persist_raw_run(
    destination: str | Path,
    raw_run: RawRun,
    manifest: EvaluationManifest | None = None,
) -> PersistedRun:
    return RawRunJournal.write(destination, raw_run, manifest)


def load_raw_run(source: str | Path) -> PersistedRun:
    return RawRunJournal.load(source)