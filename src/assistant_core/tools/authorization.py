from __future__ import annotations

from dataclasses import replace
from typing import Any

from assistant_core.domain.policy import Capability
from assistant_core.domain.tools import ToolCallRequest, ToolSpec
from assistant_core.tools.events import SYSTEM_DIAGNOSTICS_CAPABILITIES


def with_effective_working_directory(
    request: ToolCallRequest,
    spec: ToolSpec,
) -> ToolCallRequest:
    if request.working_directory is not None or spec.capability not in {
        Capability.TOOL_SHELL_READ,
        *SYSTEM_DIAGNOSTICS_CAPABILITIES,
    }:
        return request
    cwd = request.arguments.get("cwd")
    if isinstance(cwd, str) and cwd:
        return replace(request, working_directory=cwd)
    return request


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
    schema = spec.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = sorted(required - set(arguments))
    if missing:
        return f"missing required arguments: {missing}"
    if not schema.get("additionalProperties", True):
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return f"unexpected arguments: {unexpected}"
    for key, value in arguments.items():
        property_schema = properties.get(key, {})
        expected = property_schema.get("type")
        if expected is not None and not matches_type(value, expected):
            return f"invalid type for argument: {key}"
        if expected == "array":
            min_items = property_schema.get("minItems")
            if isinstance(min_items, int) and len(value) < min_items:
                return f"array argument has too few items: {key}"
            max_items = property_schema.get("maxItems")
            if isinstance(max_items, int) and len(value) > max_items:
                return f"array argument has too many items: {key}"
            item_schema = property_schema.get("items", {})
            item_type = item_schema.get("type")
            if isinstance(item_type, str) and not all(
                matches_type(item, item_type) for item in value
            ):
                return f"invalid array item type for argument: {key}"
            item_max_length = item_schema.get("maxLength", property_schema.get("maxLength"))
            if isinstance(item_max_length, int) and any(
                isinstance(item, str) and len(item) > item_max_length for item in value
            ):
                return f"array item is too long: {key}"
        max_length = property_schema.get("maxLength", schema.get("maxLength"))
        if isinstance(value, str) and isinstance(max_length, int) and len(value) > max_length:
            return f"argument is too long: {key}"
    return None


def matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False
