from __future__ import annotations

from assistant_core.context_assembly.rendering import (
    build_sections,
    chat_message,
    context_token_estimate,
)
from assistant_core.domain.context import ContextAssemblyRequest, ContextDroppedRef, ContextSection
from assistant_core.domain.conversations import ConversationMessage
from assistant_core.domain.content_retrieval import ContentHit
from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.memory import MemoryHit
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart


def exclude_current_user_message(
    request: ContextAssemblyRequest,
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    if request.current_user_message_id is None:
        return messages
    return [
        message
        for message in messages
        if message.message_id != request.current_user_message_id
    ]


def apply_message_count_limit(
    messages: list[ConversationMessage],
    max_messages: int,
    dropped_refs: list[ContextDroppedRef],
) -> list[ConversationMessage]:
    if len(messages) <= max_messages:
        return messages
    dropped = messages[: len(messages) - max_messages]
    for message in dropped:
        dropped_refs.append(
            ContextDroppedRef(kind="message", ref_id=message.message_id, reason="max_messages"),
        )
    return messages[-max_messages:]


def apply_token_budget(
    request: ContextAssemblyRequest,
    recent_messages: list[ConversationMessage],
    memory_hits: list[MemoryHit],
    content_hits: list[ContentHit],
    tool_observation_refs: list[ToolObservationRef],
    sections: list[ContextSection],
    dropped_refs: list[ContextDroppedRef],
) -> tuple[list[ConversationMessage], list[ContentHit], list[ContextSection], list[ToolObservationRef]]:
    if request.max_input_tokens is None:
        return recent_messages, content_hits, sections, tool_observation_refs

    current_messages = current_conversation_messages(request, recent_messages)
    if content_hits and context_token_estimate(sections, current_messages) > request.max_input_tokens:
        for hit in content_hits:
            dropped_refs.append(
                ContextDroppedRef(
                    kind="content",
                    ref_id=hit.chunk_id,
                    reason="token_budget",
                ),
            )
        content_hits = []
        sections = build_sections(
            request,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
        )
    if (
        tool_observation_refs
        and context_token_estimate(sections, current_messages) > request.max_input_tokens
    ):
        for ref in tool_observation_refs:
            dropped_refs.append(
                ContextDroppedRef(
                    kind="tool_observation",
                    ref_id=ref.tool_call_id,
                    reason="token_budget",
                ),
            )
        tool_observation_refs = []
        sections = build_sections(
            request,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
        )
    while recent_messages and context_token_estimate(sections, current_messages) > request.max_input_tokens:
        dropped = recent_messages.pop(0)
        dropped_refs.append(
            ContextDroppedRef(kind="message", ref_id=dropped.message_id, reason="token_budget"),
        )
        sections = build_sections(
            request,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
        )
        current_messages = current_conversation_messages(request, recent_messages)
    return recent_messages, content_hits, sections, tool_observation_refs


def current_conversation_messages(
    request: ContextAssemblyRequest,
    recent_messages: list[ConversationMessage],
) -> list[ChatMessage]:
    messages = [chat_message(message) for message in recent_messages]
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content=[TextPart(text=request.current_user_message)],
            sensitivity=request.current_message_sensitivity,
        ),
    )
    return messages
