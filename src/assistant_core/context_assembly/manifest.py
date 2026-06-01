from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from assistant_core.domain.context import (
    ContextContentHitRef,
    ContextDroppedRef,
    ContextManifest,
    ContextSection,
)
from assistant_core.domain.conversations import ConversationMessage
from assistant_core.domain.content_retrieval import ContentHit
from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.memory import MemoryHit
from assistant_core.domain.sensitivity import Sensitivity


SENSITIVITY_ORDER = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.PROJECT: 1,
    Sensitivity.PERSONAL: 2,
    Sensitivity.INFRA: 3,
    Sensitivity.SECRET: 4,
}


def manifest(
    request,
    sections: list[ContextSection],
    recent_messages: list[ConversationMessage],
    memory_hits: list[MemoryHit],
    content_hits: list[ContentHit],
    tool_observation_refs: list[ToolObservationRef],
    dropped_refs: list[ContextDroppedRef],
    token_estimate: int,
    active_namespaces: list[str],
    retrieval_parameters: dict[str, Any],
    degraded: bool,
) -> ContextManifest:
    used_message_ids = [message.message_id for message in recent_messages]
    used_memory_ids = [hit.memory.id for hit in memory_hits]
    content_refs = [content_hit_ref(hit) for hit in content_hits]
    tool_observation_ids = [ref.tool_call_id for ref in tool_observation_refs]
    sources_by_sensitivity = sources_by_sensitivity_map(
        request,
        recent_messages,
        memory_hits,
        content_hits,
        tool_observation_refs,
    )
    return ContextManifest(
        context_manifest_id=str(
            uuid5(
                NAMESPACE_URL,
                context_manifest_seed(
                    request.request_id,
                    getattr(request, "purpose", None),
                    getattr(request, "output_contract", None),
                    used_message_ids,
                    used_memory_ids,
                    [ref.chunk_id for ref in content_refs],
                    tool_observation_ids,
                ),
            ),
        ),
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        loop_strategy=request.loop_strategy,
        model_profile=request.model_profile,
        section_names=[section.name for section in sections],
        used_message_ids=used_message_ids,
        used_memory_ids=used_memory_ids,
        dropped_refs=dropped_refs,
        token_estimate=token_estimate,
        active_namespaces=active_namespaces,
        retrieval_parameters=retrieval_parameters,
        max_sensitivity=max_sensitivity(
            request,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
        ),
        sources_by_sensitivity=sources_by_sensitivity,
        degraded=degraded,
        full_prompt_stored=False,
        used_content_refs=content_refs,
    )


def sources_by_sensitivity_map(
    request,
    messages: list[ConversationMessage],
    memory_hits: list[MemoryHit],
    content_hits: list[ContentHit],
    tool_observation_refs: list[ToolObservationRef],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {
        request.current_message_sensitivity.value: ["current_user_message"],
    }
    for message in messages:
        sources.setdefault(message.sensitivity.value, []).append(message.message_id)
    for hit in memory_hits:
        sources.setdefault(hit.memory.sensitivity.value, []).append(hit.memory.id)
    for hit in content_hits:
        sources.setdefault(hit.sensitivity.value, []).append(hit.chunk_id)
    for ref in tool_observation_refs:
        sources.setdefault(ref.sensitivity.value, []).append(ref.tool_call_id)
    return sources


def context_manifest_seed(
    request_id: str,
    purpose: str | None,
    output_contract: str | None,
    used_message_ids: list[str],
    used_memory_ids: list[str],
    used_content_chunk_ids: list[str],
    tool_observation_ids: list[str],
) -> str:
    seed = (
        f"jarvis-context:{request_id}:purpose={purpose or 'default'}:"
        f"output_contract={output_contract or 'default'}:{used_message_ids}:{used_memory_ids}"
    )
    if used_content_chunk_ids:
        seed = f"{seed}:{used_content_chunk_ids}"
    if tool_observation_ids:
        return f"{seed}:{tool_observation_ids}"
    return seed


def content_hit_ref(hit: ContentHit) -> ContextContentHitRef:
    return ContextContentHitRef(
        source_id=hit.source_id,
        chunk_id=hit.chunk_id,
        citation=hit.citation.format(),
        score=hit.score,
        sensitivity=hit.sensitivity,
        content_hash=hit.content_hash,
    )


def max_sensitivity(
    request,
    messages: list[ConversationMessage],
    memory_hits: list[MemoryHit],
    content_hits: list[ContentHit],
    tool_observation_refs: list[ToolObservationRef],
) -> Sensitivity:
    values = [request.current_message_sensitivity]
    values.extend(message.sensitivity for message in messages)
    values.extend(hit.memory.sensitivity for hit in memory_hits)
    values.extend(hit.sensitivity for hit in content_hits)
    values.extend(ref.sensitivity for ref in tool_observation_refs)
    return max(values, key=lambda value: SENSITIVITY_ORDER[value])
