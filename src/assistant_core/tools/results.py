from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import (
    ToolCallRequest,
    ToolInvocationResult,
    ToolObservation,
    ToolObservationStatus,
    ToolParseStatus,
)
from assistant_core.privacy.redaction import (
    redact_content,
    redact_structured_content,
)
from assistant_core.tools.events import tool_output_sensitivity
from assistant_core.tools.registry import ToolAdapter


def empty_observation(
    request: ToolCallRequest,
    status: ToolObservationStatus,
    started_at: datetime,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    sensitivity: Sensitivity | None = None,
) -> ToolObservation:
    completed_at = datetime.now(UTC)
    return ToolObservation(
        tool_call_id=tool_call_id or str(uuid4()),
        tool_name=tool_name or request.tool_name,
        status=status,
        content="",
        content_type="text/plain",
        sensitivity=sensitivity or request.sensitivity,
        truncated=False,
        output_bytes=0,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
        error=error,
        metadata=metadata or {},
        parse_status=ToolParseStatus.NOT_APPLICABLE,
    )


def completed_observation(
    *,
    request: ToolCallRequest,
    adapter: ToolAdapter,
    tool_call_id: str,
    started_at: datetime,
    result: Any,
    max_output_bytes: int,
) -> ToolObservation:
    (
        content,
        content_type,
        metadata,
        result_truncated,
        result_output_bytes,
        structured_content,
        structured_schema,
        structured_schema_version,
        parse_status,
        parse_warnings,
    ) = serialize_content(
        result,
        adapter.content_type,
    )
    content = redact_content(content, content_type=content_type)
    encoded = content.encode("utf-8")
    output_bytes = result_output_bytes if result_output_bytes is not None else len(encoded)
    gateway_truncated = len(encoded) > max_output_bytes
    truncated = result_truncated or gateway_truncated
    if gateway_truncated:
        content = encoded[:max_output_bytes].decode("utf-8", errors="ignore")
    completed_at = datetime.now(UTC)
    return ToolObservation(
        tool_call_id=tool_call_id,
        tool_name=request.tool_name,
        status=ToolObservationStatus.COMPLETED,
        content=content,
        content_type=content_type,
        sensitivity=tool_output_sensitivity(request, adapter.spec),
        truncated=truncated,
        output_bytes=output_bytes,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
        error=None,
        metadata=metadata,
        structured_content=redact_structured_content(structured_content),
        structured_schema=structured_schema,
        structured_schema_version=structured_schema_version,
        parse_status=parse_status,
        parse_warnings=parse_warnings,
    )


def serialize_content(
    result: Any,
    content_type: str,
) -> tuple[
    str,
    str,
    dict[str, Any],
    bool,
    int | None,
    Any | None,
    str | None,
    int | None,
    ToolParseStatus,
    tuple[str, ...],
]:
    if isinstance(result, ToolInvocationResult):
        return (
            result.content,
            result.content_type,
            result.metadata,
            result.truncated,
            result.output_bytes,
            result.structured_content,
            result.structured_schema,
            result.structured_schema_version,
            result.parse_status,
            result.parse_warnings,
        )
    if isinstance(result, str):
        return result, content_type, {}, False, None, None, None, None, ToolParseStatus.NOT_APPLICABLE, ()
    return (
        json.dumps(result, sort_keys=True),
        "application/json",
        {},
        False,
        None,
        None,
        None,
        None,
        ToolParseStatus.NOT_APPLICABLE,
        (),
    )


def effective_timeout(request: ToolCallRequest, spec) -> float:
    if request.timeout_seconds is None:
        return spec.default_timeout_seconds
    return min(request.timeout_seconds, spec.default_timeout_seconds)


def effective_max_output(request: ToolCallRequest, spec) -> int:
    if request.max_output_bytes is None:
        return spec.max_output_bytes
    return min(request.max_output_bytes, spec.max_output_bytes)
