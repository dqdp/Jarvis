from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from assistant_core.context_assembly.manifest import max_sensitivity
from assistant_core.domain.context import AssembledContext, ContextAssemblyRequest
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.memory import MemoryHit
from assistant_core.domain.policy import PolicyDecision
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventLogPort


CONTEXT_ASSEMBLER_SOURCE = "context_assembler"


class ContextAssemblyAuditRecorder:
    def __init__(self, event_log: EventLogPort) -> None:
        self._event_log = event_log

    async def record_memory_retrieved(
        self,
        request: ContextAssemblyRequest,
        memory_hits: list[MemoryHit],
    ) -> EventEnvelope:
        return await self._event_log.append(
            _event(
                request,
                event_type=EventType.MEMORY_RETRIEVED,
                sensitivity=max_sensitivity(request, [], memory_hits, [], []),
                payload={
                    "used_memory_ids": [hit.memory.id for hit in memory_hits],
                    "scores": {hit.memory.id: hit.score for hit in memory_hits},
                    "full_memory_content_stored": False,
                },
            ),
        )

    async def record_memory_retrieval_failed(
        self,
        request: ContextAssemblyRequest,
    ) -> EventEnvelope:
        return await self._event_log.append(
            _event(
                request,
                event_type=EventType.MEMORY_RETRIEVAL_FAILED,
                sensitivity=Sensitivity.PROJECT,
                payload={"error_code": "memory_retrieval_failed"},
            ),
        )

    async def record_policy_decision(
        self,
        request: ContextAssemblyRequest,
        *,
        source_ref: str,
        decision: PolicyDecision,
        sensitivity: Sensitivity,
    ) -> None:
        await self._event_log.append(
            _event(
                request,
                event_type=EventType.POLICY_DECISION_RECORDED,
                sensitivity=sensitivity,
                payload={
                    "source_ref": source_ref,
                    "allowed": decision.allowed,
                    "code": decision.code,
                    "reason": decision.reason,
                },
            ),
        )

    async def record_context_assembled(
        self,
        context: AssembledContext,
        *,
        causation_id: str | None,
    ) -> None:
        manifest = context.manifest
        await self._event_log.append(
            _context_event(
                manifest,
                event_type=EventType.CONTEXT_ASSEMBLED,
                causation_id=causation_id,
                sensitivity=manifest.max_sensitivity,
                payload={
                    "context_manifest_id": manifest.context_manifest_id,
                    "section_names": manifest.section_names,
                    "used_message_ids": manifest.used_message_ids,
                    "used_memory_ids": manifest.used_memory_ids,
                    "used_content_refs": [
                        {
                            "source_id": ref.source_id,
                            "chunk_id": ref.chunk_id,
                            "citation": ref.citation,
                            "score": ref.score,
                            "sensitivity": ref.sensitivity.value,
                            "content_hash": ref.content_hash,
                        }
                        for ref in manifest.used_content_refs
                    ],
                    "token_estimate": manifest.token_estimate,
                    "dropped_refs": [
                        {"kind": ref.kind, "ref_id": ref.ref_id, "reason": ref.reason}
                        for ref in manifest.dropped_refs
                    ],
                    "active_namespaces": manifest.active_namespaces,
                    "retrieval_parameters": manifest.retrieval_parameters,
                    "sources_by_sensitivity": manifest.sources_by_sensitivity,
                    "degraded": manifest.degraded,
                    "full_prompt_stored": manifest.full_prompt_stored,
                },
            ),
        )


def _event(
    request: ContextAssemblyRequest,
    *,
    event_type: EventType,
    sensitivity: Sensitivity,
    payload: dict,
) -> EventEnvelope:
    return _context_event(
        request,
        event_type=event_type,
        causation_id=request.causation_event_id,
        sensitivity=sensitivity,
        payload=payload,
    )


def _context_event(
    source,
    *,
    event_type: EventType,
    causation_id: str | None,
    sensitivity: Sensitivity,
    payload: dict,
) -> EventEnvelope:
    now = datetime.now(UTC)
    return EventEnvelope(
        event_id=str(uuid4()),
        event_seq=0,
        event_type=event_type,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=source.conversation_id,
        request_id=source.request_id,
        correlation_id=source.request_id,
        causation_id=causation_id,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component=CONTEXT_ASSEMBLER_SOURCE,
        source_node=None,
        sensitivity=sensitivity,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload=payload,
        metadata={},
    )
