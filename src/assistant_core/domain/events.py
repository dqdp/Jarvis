from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from assistant_core.domain.sensitivity import Sensitivity


class EventType(StrEnum):
    USER_MESSAGE_CREATED = "user.message.created"
    ASSISTANT_MESSAGE_CREATED = "assistant.message.created"
    REQUEST_PROCESSING_STARTED = "request.processing.started"
    REQUEST_PROCESSING_COMPLETED = "request.processing.completed"
    REQUEST_PROCESSING_FAILED = "request.processing.failed"
    REQUEST_PROCESSING_CANCELLED = "request.processing.cancelled"
    CONTEXT_ASSEMBLY_STARTED = "context.assembly.started"
    CONTEXT_ASSEMBLED = "context.assembled"
    CONTEXT_ASSEMBLY_FAILED = "context.assembly.failed"
    CONTEXT_ASSEMBLY_TRUNCATED = "context.assembly.truncated"
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_RETRIEVAL_FAILED = "memory.retrieval.failed"
    MEMORY_EMBEDDING_CREATED = "memory.embedding.created"
    MEMORY_EMBEDDING_FAILED = "memory.embedding.failed"
    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_ARCHIVED = "memory.archived"
    MEMORY_SUPERSEDED = "memory.superseded"
    MODEL_REQUEST_CREATED = "model.request.created"
    MODEL_RESPONSE_RECEIVED = "model.response.received"
    MODEL_REQUEST_FAILED = "model.request.failed"
    MODEL_REQUEST_DENIED = "model.request.denied"
    POLICY_DECISION_RECORDED = "policy.decision.recorded"
    RUNTIME_ERROR = "runtime.error"


class ActorType(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    MODEL = "model"
    TOOL = "tool"
    SCHEDULER = "scheduler"


class EventVisibility(StrEnum):
    INTERNAL = "internal"
    USER_VISIBLE = "user_visible"
    DEBUG = "debug"


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    event_seq: int
    event_type: EventType
    event_version: int
    occurred_at: datetime
    recorded_at: datetime
    conversation_id: str | None
    request_id: str | None
    correlation_id: str | None
    causation_id: str | None
    parent_event_id: str | None
    actor_type: ActorType
    actor_id: str | None
    source_component: str
    source_node: str | None
    sensitivity: Sensitivity
    visibility: EventVisibility
    idempotency_key: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
