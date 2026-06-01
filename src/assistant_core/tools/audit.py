from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import (
    ToolCallRequest,
    ToolObservation,
    safe_tool_observation_error_code,
)


class ToolInvocationAuditRecorder:
    def __init__(self, event_log) -> None:
        self._event_log = event_log

    async def record_observation(
        self,
        request: ToolCallRequest,
        observation: ToolObservation,
        *,
        policy_decision_id: str | None,
    ) -> None:
        await self.record_event(
            EventType.TOOL_OBSERVATION_RECORDED,
            request,
            tool_call_id=observation.tool_call_id,
            payload={
                "tool_name": observation.tool_name,
                "policy_decision_id": policy_decision_id,
                "status": observation.status.value,
                "truncated": observation.truncated,
                "output_bytes": observation.output_bytes,
                "error_code": safe_tool_observation_error_code(
                    observation.error.get("code") if observation.error else None,
                ),
                "structured_schema": observation.structured_schema,
                "structured_schema_version": observation.structured_schema_version,
                "parse_status": observation.parse_status.value,
                "parse_warnings": list(observation.parse_warnings),
            },
            sensitivity=observation.sensitivity,
        )

    async def record_event(
        self,
        event_type: EventType,
        request: ToolCallRequest,
        *,
        tool_call_id: str,
        payload: dict[str, Any],
        sensitivity: Sensitivity | None = None,
    ) -> None:
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=event_type,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                correlation_id=request.correlation_id or request.request_id,
                causation_id=request.causation_event_id,
                parent_event_id=None,
                actor_type=ActorType.TOOL,
                actor_id=request.user_id,
                source_component="tool_gateway",
                source_node=None,
                sensitivity=sensitivity or request.sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=request.idempotency_key,
                payload={"tool_call_id": tool_call_id, "step_id": request.step_id, **payload},
                metadata={},
            ),
        )
