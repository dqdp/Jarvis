from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistant_core.domain.events import EventType
from assistant_core.domain.requests import RequestStatus
from assistant_core.ports.event_log import EventFilter


@dataclass(frozen=True)
class RequestStreamEvent:
    event_type: str
    data: dict[str, Any]


TERMINAL_REQUEST_STATUSES = {
    RequestStatus.COMPLETED,
    RequestStatus.FAILED,
    RequestStatus.CANCELLED,
}
TERMINAL_EVENT_TYPES = {
    EventType.REQUEST_PROCESSING_COMPLETED.value,
    EventType.REQUEST_PROCESSING_FAILED.value,
    EventType.REQUEST_PROCESSING_CANCELLED.value,
}
STREAM_REPLAY_EVENT_TYPES = {
    EventType.REQUEST_PROCESSING_STARTED.value,
    EventType.LOOP_SELECTION_STARTED.value,
    EventType.LOOP_SELECTION_COMPLETED.value,
    EventType.LOOP_SELECTION_FAILED.value,
    EventType.CONTEXT_ASSEMBLY_STARTED.value,
    EventType.MEMORY_RETRIEVED.value,
    EventType.MEMORY_RETRIEVAL_FAILED.value,
    EventType.CONTENT_RETRIEVED.value,
    EventType.CONTEXT_ASSEMBLED.value,
    EventType.MODEL_REQUEST_CREATED.value,
    EventType.MODEL_RESPONSE_RECEIVED.value,
    EventType.ASSISTANT_MESSAGE_CREATED.value,
    EventType.REQUEST_PROCESSING_COMPLETED.value,
    EventType.REQUEST_PROCESSING_FAILED.value,
    EventType.REQUEST_PROCESSING_CANCELLED.value,
    EventType.APPROVAL_REQUIRED.value,
    EventType.APPROVAL_GRANTED.value,
    EventType.APPROVAL_DENIED.value,
    EventType.APPROVAL_EXPIRED.value,
    EventType.APPROVAL_CANCELLED.value,
    EventType.TOOL_CALL_STARTED.value,
    EventType.TOOL_CALL_COMPLETED.value,
    EventType.TOOL_CALL_DENIED.value,
    EventType.TOOL_CALL_FAILED.value,
    EventType.TOOL_CALL_TIMEOUT.value,
    EventType.TOOL_CALL_CANCELLED.value,
    EventType.TOOL_SHELL_STARTED.value,
    EventType.TOOL_SHELL_COMPLETED.value,
    EventType.TOOL_SHELL_DENIED.value,
    EventType.TOOL_SHELL_FAILED.value,
    EventType.TOOL_SHELL_TIMEOUT.value,
    EventType.TOOL_SHELL_OUTPUT_TRUNCATED.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_STARTED.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_FAILED.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_TIMEOUT.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_OUTPUT_TRUNCATED.value,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE.value,
}
PUBLIC_STREAM_FIELDS = {
    "token": ("request_id", "delta"),
    EventType.REQUEST_PROCESSING_STARTED.value: ("request_id", "event_id"),
    EventType.LOOP_SELECTION_STARTED.value: (
        "request_id",
        "event_id",
        "requested_mode",
    ),
    EventType.LOOP_SELECTION_COMPLETED.value: (
        "request_id",
        "event_id",
        "requested_mode",
        "selected_loop_strategy",
        "selected_model_profile",
        "request_plan_status",
        "request_plan_reason_code",
        "agent_tool_policy",
        "agent_allowed_tool_count",
        "agent_allowed_tool_names",
    ),
    EventType.LOOP_SELECTION_FAILED.value: (
        "request_id",
        "event_id",
        "requested_mode",
        "selected_loop_strategy",
        "selected_model_profile",
        "request_plan_status",
        "request_plan_reason_code",
    ),
    EventType.CONTEXT_ASSEMBLY_STARTED.value: ("request_id", "event_id"),
    EventType.MEMORY_RETRIEVED.value: ("request_id", "event_id"),
    EventType.MEMORY_RETRIEVAL_FAILED.value: ("request_id", "event_id", "error"),
    EventType.CONTENT_RETRIEVED.value: (
        "request_id",
        "event_id",
        "hit_count",
        "full_content_stored",
    ),
    EventType.CONTEXT_ASSEMBLED.value: (
        "request_id",
        "event_id",
        "context_manifest_id",
        "degraded",
        "token_estimate",
    ),
    EventType.MODEL_REQUEST_CREATED.value: ("request_id", "event_id", "context_manifest_id"),
    EventType.MODEL_RESPONSE_RECEIVED.value: ("request_id", "event_id", "context_manifest_id"),
    EventType.ASSISTANT_MESSAGE_CREATED.value: (
        "request_id",
        "event_id",
        "message_id",
        "content_hash",
    ),
    EventType.REQUEST_PROCESSING_COMPLETED.value: (
        "request_id",
        "event_id",
        "assistant_message_id",
    ),
    EventType.REQUEST_PROCESSING_FAILED.value: ("request_id", "event_id", "error"),
    EventType.REQUEST_PROCESSING_CANCELLED.value: ("request_id", "event_id", "error"),
    EventType.APPROVAL_REQUIRED.value: (
        "request_id",
        "event_id",
        "approval_id",
        "status",
        "capability",
        "risk_classes",
        "tool_name",
        "redacted_summary",
        "expires_at",
    ),
    EventType.APPROVAL_GRANTED.value: ("request_id", "event_id", "approval_id", "status"),
    EventType.APPROVAL_DENIED.value: ("request_id", "event_id", "approval_id", "status"),
    EventType.APPROVAL_EXPIRED.value: ("request_id", "event_id", "approval_id", "status"),
    EventType.APPROVAL_CANCELLED.value: ("request_id", "event_id", "approval_id", "status"),
    EventType.TOOL_CALL_STARTED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "risk_classes",
    ),
    EventType.TOOL_CALL_COMPLETED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "output_bytes",
        "truncated",
    ),
    EventType.TOOL_CALL_DENIED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
        "policy_outcome",
    ),
    EventType.TOOL_CALL_FAILED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
    ),
    EventType.TOOL_CALL_TIMEOUT.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
    ),
    EventType.TOOL_CALL_CANCELLED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
    ),
    EventType.TOOL_SHELL_STARTED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "risk_classes",
        "argv",
    ),
    EventType.TOOL_SHELL_COMPLETED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "argv",
        "exit_code",
        "output_bytes",
        "truncated",
        "duration_ms",
    ),
    EventType.TOOL_SHELL_DENIED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
        "policy_outcome",
    ),
    EventType.TOOL_SHELL_FAILED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
    ),
    EventType.TOOL_SHELL_TIMEOUT.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
    ),
    EventType.TOOL_SHELL_OUTPUT_TRUNCATED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "output_bytes",
        "truncated",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_STARTED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "risk_classes",
        "argv",
        "family",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "argv",
        "family",
        "exit_code",
        "output_bytes",
        "truncated",
        "duration_ms",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
        "policy_outcome",
        "family",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_FAILED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
        "family",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_TIMEOUT.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "error_code",
        "family",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_OUTPUT_TRUNCATED.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "output_bytes",
        "truncated",
        "family",
    ),
    EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE.value: (
        "request_id",
        "event_id",
        "tool_name",
        "capability",
        "source",
        "family",
        "unavailable",
    ),
}


async def event_log_stream(event_log, request_record):
    yielded_terminal = False
    events = [
        event
        for event in await event_log.query(EventFilter(request_id=request_record.request_id))
        if event.event_type.value in STREAM_REPLAY_EVENT_TYPES
    ]
    started_event = next(
        (
            event
            for event in events
            if event.event_type.value == EventType.REQUEST_PROCESSING_STARTED.value
        ),
        None,
    )
    if started_event is not None:
        events = [started_event, *(event for event in events if event is not started_event)]
    for event in events:
        if event.event_type.value in TERMINAL_EVENT_TYPES:
            yielded_terminal = True
        yield event_stream_event(event)
    if not yielded_terminal:
        yield terminal_stream_event(request_record)


async def terminal_stream_event_from_log(event_log, request_record) -> RequestStreamEvent:
    for event in reversed(await event_log.query(EventFilter(request_id=request_record.request_id))):
        if event.event_type.value in TERMINAL_EVENT_TYPES:
            return event_stream_event(event)
    return terminal_stream_event(request_record)


def terminal_stream_event(request_record) -> RequestStreamEvent:
    if request_record.status == RequestStatus.COMPLETED:
        return RequestStreamEvent(
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {
                "request_id": request_record.request_id,
                "assistant_message_id": request_record.assistant_message_id,
            },
        )
    if request_record.status == RequestStatus.CANCELLED:
        return RequestStreamEvent(
            EventType.REQUEST_PROCESSING_CANCELLED.value,
            {
                "request_id": request_record.request_id,
                "error": {
                    "code": "cancelled",
                    "message": "request cancelled",
                    "request_id": request_record.request_id,
                    "details": {},
                },
            },
        )
    return RequestStreamEvent(
        EventType.REQUEST_PROCESSING_FAILED.value,
        {
            "request_id": request_record.request_id,
            "error": {
                "code": request_record.error_code,
                "message": request_record.error_message,
                "request_id": request_record.request_id,
                "details": {},
            },
        },
    )


def event_stream_event(event) -> RequestStreamEvent:
    return RequestStreamEvent(
        event.event_type.value,
        public_stream_data(
            event.event_type.value,
            {"request_id": event.request_id, "event_id": event.event_id, **event.payload},
        ),
    )


def public_stream_data(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    public_data = dict(data)
    if event_type == EventType.REQUEST_PROCESSING_FAILED.value and "error" not in public_data:
        public_data["error"] = {
            "code": public_data.get("error_code") or public_data.get("error_type"),
            "message": "request failed",
            "request_id": public_data.get("request_id"),
            "details": {},
        }
    if event_type == EventType.MEMORY_RETRIEVAL_FAILED.value and "error" not in public_data:
        public_data["error"] = {
            "code": public_data.get("error_code") or public_data.get("error_type"),
            "message": "memory retrieval failed",
            "request_id": public_data.get("request_id"),
            "details": {},
        }
    if event_type == EventType.REQUEST_PROCESSING_CANCELLED.value and "error" not in public_data:
        public_data["error"] = {
            "code": "cancelled",
            "message": "request cancelled",
            "request_id": public_data.get("request_id"),
            "details": {},
        }
    if event_type == EventType.CONTENT_RETRIEVED.value and "hit_count" not in public_data:
        content_refs = public_data.get("retrieved_content_refs")
        if isinstance(content_refs, list):
            public_data["hit_count"] = len(content_refs)

    fields = PUBLIC_STREAM_FIELDS.get(event_type, ("request_id", "event_id"))
    return {
        field: public_data[field]
        for field in fields
        if field in public_data and public_data[field] is not None
    }


def has_terminal_event(events: list[tuple[str, dict[str, Any]]]) -> bool:
    return any(event_type in TERMINAL_EVENT_TYPES for event_type, _ in events)
