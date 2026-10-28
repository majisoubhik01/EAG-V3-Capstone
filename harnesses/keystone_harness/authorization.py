"""Offline policy validation for the narrow Keystone resource-read candidates."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from urllib.parse import urlsplit
from typing import Any

from .client import (
    DiscoveredTool,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
    ToolDiscoveryResult,
)


RESOURCE_READ_OPERATIONS = frozenset(
    {
        "Party.list",
        "Calendar.list",
        "CalendarEvent.list",
        "EventAttendee.list",
    }
)
GOAL_ONE_LIST_READ_OPERATIONS = frozenset(
    {"SchedulingPreferences.list", "AvailabilityRule.list"}
)
GOAL_ONE_ENDPOINT_READ_OPERATIONS = frozenset({"endpoint.calendar.free_busy"})
DISPOSABLE_CALENDAR_MUTATION_OPERATIONS = frozenset(
    {
        "CalendarEvent.create",
        "CalendarEvent.update",
    }
)

_ANNOTATION_KEYWORDS = frozenset(
    {
        "$schema",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
_SCHEMA_KEYWORDS = _ANNOTATION_KEYWORDS | frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "oneOf",
        "allOf",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    }
)
_JSON_TYPES = frozenset(
    {"null", "boolean", "object", "array", "number", "integer", "string"}
)
_SUPPORTED_FORMATS = frozenset(
    {"date", "date-time", "email", "uri-reference"}
)


def validate_resource_read_policy(
    discovery: ToolDiscoveryResult,
    operation: str,
    arguments: Mapping[str, Any],
) -> DiscoveredTool:
    """Validate a local candidate read request; this does not enable transport."""

    if not isinstance(discovery, ToolDiscoveryResult):
        raise TypeError("discovery must be a ToolDiscoveryResult")
    if not isinstance(operation, str) or operation not in (
        RESOURCE_READ_OPERATIONS
        | GOAL_ONE_LIST_READ_OPERATIONS
        | GOAL_ONE_ENDPOINT_READ_OPERATIONS
    ):
        raise KeystoneOperationNotAllowedError(
            "Keystone read operation is not in the exact read candidate set."
        )

    matches = [tool for tool in discovery.tools if tool.name == operation]
    if len(matches) != 1:
        raise KeystoneOperationNotAllowedError(
            f"{operation} must appear exactly once in the observed tool catalog."
        )
    tool = matches[0]
    _validate_safety_metadata(operation, tool)
    if operation in RESOURCE_READ_OPERATIONS | GOAL_ONE_LIST_READ_OPERATIONS:
        _validate_descriptor_schemas(operation, tool)
    else:
        _validate_goal_one_endpoint_schema(operation, tool)
    _validate_arguments(tool.input_schema, arguments)
    return tool


def _validate_goal_one_endpoint_schema(
    operation: str,
    tool: DiscoveredTool,
) -> None:
    if operation != "endpoint.calendar.free_busy":
        raise KeystoneOperationNotAllowedError(
            "Keystone endpoint read is not in the exact Goal 1 allowlist."
        )
    if not isinstance(tool.input_schema, Mapping):
        raise KeystoneProtocolError(f"{operation} has a malformed inputSchema.")
    properties = tool.input_schema.get("properties")
    required = tool.input_schema.get("required")
    if (
        tool.input_schema.get("type") != "object"
        or tool.input_schema.get("additionalProperties") is not False
        or not isinstance(properties, Mapping)
        or not isinstance(required, list)
        or not {"start", "end"}.issubset(required)
        or any(
            not isinstance(properties.get(field), Mapping)
            or properties[field].get("type") != "string"
            for field in ("start", "end")
        )
    ):
        raise KeystoneProtocolError(
            f"{operation} inputSchema does not match the supported read contract."
        )
    _validate_schema_definition(tool.input_schema)

    output_schema = tool.output_schema
    output_properties = (
        output_schema.get("properties")
        if isinstance(output_schema, Mapping)
        else None
    )
    output_required = (
        output_schema.get("required")
        if isinstance(output_schema, Mapping)
        else None
    )
    if (
        not isinstance(output_schema, Mapping)
        or output_schema.get("type") != "object"
        or not isinstance(output_properties, Mapping)
        or not isinstance(output_required, list)
        or not {"status", "result"}.issubset(output_required)
        or not isinstance(output_properties.get("status"), Mapping)
        or output_properties["status"].get("type") != "string"
        or not isinstance(output_properties.get("result"), Mapping)
        or output_properties["result"].get("type") != "object"
    ):
        raise KeystoneProtocolError(
            f"{operation} outputSchema does not match the supported read contract."
        )
    _validate_schema_definition(output_schema)


def validate_disposable_calendar_mutation_policy(
    discovery: ToolDiscoveryResult,
    operation: str,
    arguments: Mapping[str, Any],
) -> DiscoveredTool:
    """Validate one explicitly opted-in Calendar create/update for a disposable copy."""

    if not isinstance(discovery, ToolDiscoveryResult):
        raise TypeError("discovery must be a ToolDiscoveryResult")
    if (
        not isinstance(operation, str)
        or operation not in DISPOSABLE_CALENDAR_MUTATION_OPERATIONS
    ):
        raise KeystoneOperationNotAllowedError(
            "Calendar mutation is not in the exact disposable-copy allowlist."
        )

    matches = [tool for tool in discovery.tools if tool.name == operation]
    if len(matches) != 1:
        raise KeystoneOperationNotAllowedError(
            f"{operation} must appear exactly once in the observed tool catalog."
        )
    tool = matches[0]
    annotations = tool.annotations
    if (
        type(tool.read_only_hint) is not bool
        or tool.read_only_hint is not False
        or not isinstance(annotations, Mapping)
        or annotations.get("readOnlyHint") is not False
        or annotations.get("destructiveHint") is not False
    ):
        raise KeystoneOperationNotAllowedError(
            f"{operation} lacks the expected non-read-only mutation descriptor."
        )
    if (
        not isinstance(tool.input_schema, Mapping)
        or tool.input_schema.get("type") != "object"
        or tool.input_schema.get("additionalProperties") is not False
    ):
        raise KeystoneProtocolError(
            f"{operation} inputSchema must be an object that rejects additional properties."
        )
    if not isinstance(tool.output_schema, Mapping):
        raise KeystoneProtocolError(f"{operation} has no declared outputSchema.")
    output_properties = tool.output_schema.get("properties")
    if (
        tool.output_schema.get("type") != "object"
        or not isinstance(output_properties, Mapping)
        or "id" not in tool.output_schema.get("required", [])
        or not isinstance(output_properties.get("id"), Mapping)
        or output_properties["id"].get("type") != "string"
    ):
        raise KeystoneProtocolError(
            f"{operation} outputSchema must return a string event ID."
        )

    if operation == "CalendarEvent.update":
        allowed_fields = {"id", "start_at", "end_at"}
        if set(arguments) - allowed_fields:
            raise KeystoneOperationNotAllowedError(
                "Disposable CalendarEvent.update may change only event ID and times."
            )
        if not {"id", "start_at", "end_at"}.issubset(arguments):
            raise KeystoneProtocolError(
                "Disposable CalendarEvent.update requires ID, start_at, and end_at."
            )
        start = arguments.get("start_at")
        end = arguments.get("end_at")
        if not isinstance(start, str) or not isinstance(end, str):
            raise KeystoneProtocolError(
                "Disposable CalendarEvent.update requires string timestamps."
            )
        try:
            parsed_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
            parsed_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            raise KeystoneProtocolError(
                "Disposable CalendarEvent.update timestamps are invalid."
            ) from None
        if (
            parsed_start.tzinfo is None
            or parsed_start.utcoffset() is None
            or parsed_end.tzinfo is None
            or parsed_end.utcoffset() is None
            or parsed_end <= parsed_start
        ):
            raise KeystoneProtocolError(
                "Disposable CalendarEvent.update timestamps must be aware and ordered."
            )
    elif set(arguments) - {"calendar_id", "title", "start_at", "end_at", "timezone"}:
        raise KeystoneOperationNotAllowedError(
            "Disposable CalendarEvent.create accepts only a simple timed test event."
        )

    validate_json_schema_instance(
        tool.input_schema,
        dict(arguments),
        location="disposable Calendar mutation arguments",
    )
    return tool


def _validate_safety_metadata(operation: str, tool: DiscoveredTool) -> None:
    annotations = tool.annotations
    meta = tool.meta
    if not isinstance(annotations, Mapping):
        raise KeystoneOperationNotAllowedError(
            f"{operation} has no valid annotations object."
        )
    if type(tool.read_only_hint) is not bool or tool.read_only_hint is not True:
        raise KeystoneOperationNotAllowedError(
            f"{operation} is not explicitly marked read-only."
        )
    if (
        type(annotations.get("readOnlyHint")) is not bool
        or annotations.get("readOnlyHint") is not True
    ):
        raise KeystoneOperationNotAllowedError(
            f"{operation} has missing or contradictory readOnlyHint metadata."
        )
    if (
        type(annotations.get("destructiveHint")) is not bool
        or annotations.get("destructiveHint") is not False
    ):
        raise KeystoneOperationNotAllowedError(
            f"{operation} has missing or contradictory destructiveHint metadata."
        )
    if not isinstance(meta, Mapping):
        raise KeystoneOperationNotAllowedError(
            f"{operation} has no valid _meta object."
        )
    agentswitch = meta.get("agentswitch")
    if not isinstance(agentswitch, Mapping):
        raise KeystoneOperationNotAllowedError(
            f"{operation} has no valid AgentSwitch permission metadata."
        )
    allowed_permissions = {"read"}
    if operation == "endpoint.calendar.free_busy":
        allowed_permissions.add("CalendarEvent.read")
    if agentswitch.get("permission") not in allowed_permissions:
        raise KeystoneOperationNotAllowedError(
            f"{operation} does not declare read permission."
        )
    if agentswitch.get("risk") != "READ":
        raise KeystoneOperationNotAllowedError(
            f"{operation} risk metadata is missing or is not READ."
        )


def _validate_descriptor_schemas(operation: str, tool: DiscoveredTool) -> None:
    if tool.name != operation:
        raise KeystoneOperationNotAllowedError(
            "The discovered descriptor name does not exactly match the requested operation."
        )
    if not isinstance(tool.input_schema, Mapping):
        raise KeystoneProtocolError(f"{operation} has a malformed inputSchema.")
    if tool.input_schema.get("type") != "object":
        raise KeystoneProtocolError(f"{operation} inputSchema must describe an object.")
    if tool.input_schema.get("additionalProperties") is not False:
        raise KeystoneProtocolError(
            f"{operation} inputSchema must explicitly reject additional properties."
        )
    properties = tool.input_schema.get("properties")
    required = tool.input_schema.get("required", [])
    if (
        not isinstance(properties, Mapping)
        or not isinstance(required, list)
        or any(not isinstance(field, str) for field in required)
    ):
        raise KeystoneProtocolError(f"{operation} inputSchema is malformed.")
    if not {"limit", "offset"}.issubset(properties):
        raise KeystoneProtocolError(
            f"{operation} inputSchema must declare limit and offset."
        )
    for field_name in ("limit", "offset"):
        field_schema = properties[field_name]
        if not isinstance(field_schema, Mapping) or field_schema.get("type") != "integer":
            raise KeystoneProtocolError(
                f"{operation} inputSchema {field_name} must be an integer."
            )
    _validate_schema_definition(tool.input_schema)

    output_schema = tool.output_schema
    if not isinstance(output_schema, Mapping):
        raise KeystoneProtocolError(f"{operation} has no declared outputSchema.")
    output_properties = output_schema.get("properties")
    output_required = output_schema.get("required")
    if (
        output_schema.get("type") != "object"
        or not isinstance(output_properties, Mapping)
        or not isinstance(output_required, list)
        or any(not isinstance(field, str) for field in output_required)
        or not {"data", "total", "limit", "offset"}.issubset(output_required)
    ):
        raise KeystoneProtocolError(
            f"{operation} outputSchema does not declare the resource pagination contract."
        )
    for field_name in ("total", "limit", "offset"):
        field_schema = output_properties.get(field_name)
        if not isinstance(field_schema, Mapping) or field_schema.get("type") != "integer":
            raise KeystoneProtocolError(
                f"{operation} outputSchema {field_name} must be an integer."
            )
    for field_name in ("total", "limit", "offset"):
        field_schema = output_properties.get(field_name)
        if not isinstance(field_schema, Mapping) or field_schema.get("type") != "integer":
            raise KeystoneProtocolError(
                f"{operation} outputSchema {field_name} must be an integer."
            )
    data_schema = output_properties.get("data")
    if not isinstance(data_schema, Mapping) or data_schema.get("type") != "array":
        raise KeystoneProtocolError(
            f"{operation} outputSchema data must be an array."
        )
    items = data_schema.get("items")
    if not isinstance(items, Mapping):
        raise KeystoneProtocolError(
            f"{operation} outputSchema does not declare record objects."
        )
    item_properties = items.get("properties")
    item_required = items.get("required")
    if (
        items.get("type") != "object"
        or not isinstance(item_properties, Mapping)
        or not isinstance(item_required, list)
        or "id" not in item_required
        or not isinstance(item_properties.get("id"), Mapping)
        or item_properties["id"].get("type") != "string"
        or type(items.get("additionalProperties")) is not bool
    ):
        raise KeystoneProtocolError(
            f"{operation} outputSchema must guarantee a string record ID."
        )


def _validate_schema_definition(schema: Mapping[str, Any]) -> None:
    for keyword, value in schema.items():
        if keyword not in _SCHEMA_KEYWORDS:
            raise KeystoneProtocolError(
                f"Input schema uses unsupported JSON Schema keyword {keyword!r}."
            )
        if keyword == "type":
            types = value if isinstance(value, list) else [value]
            if not types or any(
                not isinstance(item, str) or item not in _JSON_TYPES
                for item in types
            ):
                raise KeystoneProtocolError("Input schema contains an invalid type.")
        elif keyword in {"properties"}:
            if not isinstance(value, Mapping):
                raise KeystoneProtocolError(f"Input schema {keyword} must be an object.")
            for child in value.values():
                if not isinstance(child, Mapping):
                    raise KeystoneProtocolError("Input schema property is malformed.")
                _validate_schema_definition(child)
        elif keyword in {"anyOf", "oneOf", "allOf"}:
            if not isinstance(value, list) or (
                keyword != "allOf" and not value
            ):
                raise KeystoneProtocolError(f"Input schema {keyword} must be an array.")
            for child in value:
                if not isinstance(child, Mapping):
                    raise KeystoneProtocolError(f"Input schema {keyword} entry is malformed.")
                _validate_schema_definition(child)
        elif keyword == "items":
            if not isinstance(value, (Mapping, bool)):
                raise KeystoneProtocolError("Input schema items must be a schema.")
            if isinstance(value, Mapping):
                _validate_schema_definition(value)
        elif keyword == "additionalProperties":
            if not isinstance(value, (Mapping, bool)):
                raise KeystoneProtocolError(
                    "Input schema additionalProperties must be boolean or a schema."
                )
            if isinstance(value, Mapping):
                _validate_schema_definition(value)
        elif keyword == "required":
            if not isinstance(value, list) or any(
                not isinstance(field, str) for field in value
            ):
                raise KeystoneProtocolError("Input schema required must be an array of strings.")
        elif keyword == "enum":
            if not isinstance(value, list):
                raise KeystoneProtocolError("Input schema enum must be an array.")
        elif keyword in {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        }:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KeystoneProtocolError(
                    f"Input schema {keyword} must be numeric."
                )
            if not math.isfinite(value):
                raise KeystoneProtocolError(
                    f"Input schema {keyword} must be finite."
                )
        elif keyword in {"minLength", "maxLength", "minItems", "maxItems",
                         "minProperties", "maxProperties"}:
            if type(value) is not int or value < 0:
                raise KeystoneProtocolError(
                    f"Input schema {keyword} must be a non-negative integer."
                )
        elif keyword in {"pattern", "format"}:
            if not isinstance(value, str):
                raise KeystoneProtocolError(
                    f"Input schema {keyword} must be a string."
                )
            if keyword == "pattern":
                try:
                    re.compile(value)
                except re.error:
                    raise KeystoneProtocolError(
                        "Input schema contains an invalid regular expression."
                    ) from None
            elif value not in _SUPPORTED_FORMATS:
                raise KeystoneProtocolError(
                    f"Input schema uses unsupported format {value!r}."
                )


def _validate_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    errors = _schema_errors(schema, arguments, "arguments")
    if errors:
        raise ValueError("; ".join(errors))


def validate_json_schema_instance(
    schema: Mapping[str, Any],
    value: Any,
    *,
    location: str = "value",
) -> None:
    """Validate an instance using the same fail-closed schema subset as policy."""

    if not isinstance(schema, Mapping):
        raise KeystoneProtocolError("JSON Schema must be an object.")
    _validate_schema_definition(schema)
    errors = _schema_errors(schema, value, location)
    if errors:
        raise KeystoneProtocolError(
            f"{location} does not satisfy its declared JSON Schema: "
            + "; ".join(errors)
        )


def _schema_errors(
    schema: Mapping[str, Any],
    value: Any,
    path: str,
) -> list[str]:
    errors: list[str] = []
    for keyword in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(keyword)
        if variants is None:
            continue
        branch_errors = [
            _schema_errors(variant, value, path) for variant in variants
        ]
        successful = sum(not branch for branch in branch_errors)
        if keyword == "anyOf" and successful == 0:
            errors.append(f"{path} does not match any allowed schema.")
        elif keyword == "oneOf" and successful != 1:
            errors.append(f"{path} does not match exactly one allowed schema.")
        elif keyword == "allOf":
            for branch in branch_errors:
                errors.extend(branch)

    declared_type = schema.get("type")
    if declared_type is not None:
        types = declared_type if isinstance(declared_type, list) else [declared_type]
        if not any(_matches_type(value, expected) for expected in types):
            errors.append(f"{path} has the wrong JSON type.")
            return errors

    if "enum" in schema and not any(
        _json_equal(value, option) for option in schema["enum"]
    ):
        errors.append(f"{path} is not one of the declared enum values.")
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path} does not match the declared constant.")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return errors + [f"{path} has an invalid object schema."]
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}.{required} is required.")
        minimum = schema.get("minProperties")
        maximum = schema.get("maxProperties")
        if minimum is not None and len(value) < minimum:
            errors.append(f"{path} has fewer than {minimum} properties.")
        if maximum is not None and len(value) > maximum:
            errors.append(f"{path} has more than {maximum} properties.")
        additional = schema.get("additionalProperties", True)
        for key, child_value in value.items():
            if key in properties:
                errors.extend(
                    _schema_errors(properties[key], child_value, f"{path}.{key}")
                )
            elif additional is False:
                errors.append(f"{path}.{key} is not allowed.")
            elif isinstance(additional, Mapping):
                errors.extend(
                    _schema_errors(additional, child_value, f"{path}.{key}")
                )
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            errors.append(f"{path} has fewer than {minimum} items.")
        if maximum is not None and len(value) > maximum:
            errors.append(f"{path} has more than {maximum} items.")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                errors.extend(_schema_errors(items, item, f"{path}[{index}]"))
        elif items is False and value:
            errors.append(f"{path} does not allow items.")
    elif isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value) < minimum:
            errors.append(f"{path} is shorter than the declared minimum.")
        if maximum is not None and len(value) > maximum:
            errors.append(f"{path} is longer than the declared maximum.")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            errors.append(f"{path} does not match the declared pattern.")
        format_name = schema.get("format")
        if format_name is not None and not _matches_format(value, format_name):
            errors.append(f"{path} does not match the declared format.")
    elif _is_number(value):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path} is below the declared minimum.")
        if maximum is not None and value > maximum:
            errors.append(f"{path} is above the declared maximum.")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            errors.append(f"{path} is not above the exclusive minimum.")
        if exclusive_maximum is not None and value >= exclusive_maximum:
            errors.append(f"{path} is not below the exclusive maximum.")
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None and value % multiple_of != 0:
            errors.append(f"{path} is not a multiple of the declared value.")
    return errors


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return type(value) is bool
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return _is_number(value)
    if expected == "string":
        return isinstance(value, str)
    return False


def _is_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_number(left) and _is_number(right):
        return left == right
    return type(left) is type(right) and left == right


def _matches_format(value: str, format_name: str) -> bool:
    if format_name == "email":
        return re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value) is not None
    if format_name == "date":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if format_name == "date-time":
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    if format_name == "uri-reference":
        if any(character.isspace() or ord(character) < 32 for character in value):
            return False
        try:
            urlsplit(value)
        except ValueError:
            return False
        return True
    return False
