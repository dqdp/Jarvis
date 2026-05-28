from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any

from assistant_core.domain.messages import ChatMessage
from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class ChatModelRequest:
    profile: str
    messages: list[ChatMessage]
    sensitivity: Sensitivity
    request_id: str | None = None
    conversation_id: str | None = None
    context_manifest_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatModelResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredModelRequest:
    profile: str
    messages: list[ChatMessage]
    schema: dict[str, Any]
    sensitivity: Sensitivity


@dataclass(frozen=True)
class StructuredModelResponse:
    value: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingRequest:
    profile: str
    texts: list[str]
    sensitivity: Sensitivity


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStreamEvent:
    event_type: str
    delta: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ModelStream = AsyncIterator[ModelStreamEvent]
