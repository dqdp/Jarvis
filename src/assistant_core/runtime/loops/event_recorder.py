from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.loops import AgentLoopState, AgentLoopStep, LoopExecutionRequest
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventLogPort


class LoopEventRecorder:
    def __init__(self, *, event_log: EventLogPort, source_component: str) -> None:
        self._event_log = event_log
        self._source_component = source_component

    async def append(
        self,
        event_type: EventType,
        request: LoopExecutionRequest,
        *,
        payload: dict[str, Any],
        causation_id: str | None = None,
        sensitivity: Sensitivity = Sensitivity.PROJECT,
        state: AgentLoopState | None = None,
        step: AgentLoopStep | None = None,
    ) -> EventEnvelope:
        event_payload = dict(payload)
        if state is not None:
            event_payload.setdefault("agent_state", state.value)
        if step is not None:
            event_payload.setdefault("agent_step", step.value)
        now = datetime.now(UTC)
        return await self._event_log.append(
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
                causation_id=causation_id,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component=self._source_component,
                source_node=None,
                sensitivity=sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload=event_payload,
                metadata={},
            ),
        )
