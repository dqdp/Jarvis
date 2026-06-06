from __future__ import annotations

from typing import Any

from assistant_core.domain.events import EventEnvelope
from assistant_core.domain.loops import LoopExecutionRequest


def failed_stream_payload(
    request: LoopExecutionRequest,
    failed_event: EventEnvelope | None,
) -> dict[str, Any]:
    if failed_event is None:
        return {
            "request_id": request.request_id,
            "event_id": None,
            "error": {
                "code": "tool_loop_failed",
                "message": "tool loop failed",
                "request_id": request.request_id,
                "details": {},
            },
        }
    error = failed_event.payload.get("error")
    if not isinstance(error, dict):
        error = {
            "code": failed_event.payload.get("error_code") or failed_event.payload.get("error_type"),
            "message": "tool loop failed",
            "request_id": request.request_id,
            "details": {},
        }
    return {
        "request_id": request.request_id,
        "event_id": failed_event.event_id,
        "error": error,
    }
