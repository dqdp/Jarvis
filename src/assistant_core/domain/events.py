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
    AGENT_LOOP_STARTED = "agent.loop.started"
    AGENT_LOOP_COMPLETED = "agent.loop.completed"
    AGENT_LOOP_FAILED = "agent.loop.failed"
    AGENT_LOOP_CANCELLED = "agent.loop.cancelled"
    AGENT_STEP_STARTED = "agent.step.started"
    AGENT_STEP_COMPLETED = "agent.step.completed"
    AGENT_STEP_FAILED = "agent.step.failed"
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
    POLICY_CAPABILITY_DECISION_RECORDED = "policy.capability.decision.recorded"
    TOOL_CALL_REQUESTED = "tool.call.requested"
    TOOL_CALL_APPROVED = "tool.call.approved"
    TOOL_CALL_DENIED = "tool.call.denied"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_CALL_TIMEOUT = "tool.call.timeout"
    TOOL_CALL_CANCELLED = "tool.call.cancelled"
    TOOL_SHELL_CLASSIFIED = "tool.shell.classified"
    TOOL_SHELL_DENIED = "tool.shell.denied"
    TOOL_SHELL_STARTED = "tool.shell.started"
    TOOL_SHELL_COMPLETED = "tool.shell.completed"
    TOOL_SHELL_FAILED = "tool.shell.failed"
    TOOL_SHELL_TIMEOUT = "tool.shell.timeout"
    TOOL_SHELL_OUTPUT_TRUNCATED = "tool.shell.output_truncated"
    TOOL_SYSTEM_DIAGNOSTICS_CLASSIFIED = "tool.system.diagnostics.classified"
    TOOL_SYSTEM_DIAGNOSTICS_DENIED = "tool.system.diagnostics.denied"
    TOOL_SYSTEM_DIAGNOSTICS_STARTED = "tool.system.diagnostics.started"
    TOOL_SYSTEM_DIAGNOSTICS_COMPLETED = "tool.system.diagnostics.completed"
    TOOL_SYSTEM_DIAGNOSTICS_FAILED = "tool.system.diagnostics.failed"
    TOOL_SYSTEM_DIAGNOSTICS_TIMEOUT = "tool.system.diagnostics.timeout"
    TOOL_SYSTEM_DIAGNOSTICS_OUTPUT_TRUNCATED = "tool.system.diagnostics.output_truncated"
    TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE = "tool.system.diagnostics.unavailable"
    CONTENT_SOURCE_DISCOVERED = "content.source.discovered"
    CONTENT_SOURCE_INGESTED = "content.source.ingested"
    CONTENT_SOURCE_UPDATED = "content.source.updated"
    CONTENT_SOURCE_DELETED = "content.source.deleted"
    CONTENT_CHUNK_CREATED = "content.chunk.created"
    CONTENT_CHUNK_STALE = "content.chunk.stale"
    TOOL_OBSERVATION_RECORDED = "tool.observation.recorded"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_EXPIRED = "approval.expired"
    APPROVAL_CANCELLED = "approval.cancelled"
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
