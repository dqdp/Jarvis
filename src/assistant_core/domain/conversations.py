from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from assistant_core.domain.messages import MessageRole
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    user_id: str
    title: str | None
    active_project_namespace: str | None
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationMessage:
    message_id: str
    conversation_id: str
    request_id: str | None
    event_id: str | None
    client_message_id: str | None
    role: MessageRole
    content: str
    content_hash: str
    sensitivity: Sensitivity
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantRequest:
    request_id: str
    conversation_id: str
    user_message_id: str
    assistant_message_id: str | None
    status: RequestStatus
    client_message_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_message: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MessageSubmission:
    user_message: ConversationMessage
    request: AssistantRequest
    idempotent_replay: bool = False


@dataclass(frozen=True)
class AssistantResponseCompletion:
    message: ConversationMessage
    request: AssistantRequest


@dataclass(frozen=True)
class CreateConversationCommand:
    user_id: str
    title: str | None = None
    active_project_namespace: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AppendMessageCommand:
    conversation_id: str
    role: MessageRole
    content: str
    sensitivity: Sensitivity
    message_id: str | None = None
    request_id: str | None = None
    event_id: str | None = None
    client_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompleteAssistantResponseCommand:
    request_id: str
    conversation_id: str
    content: str
    sensitivity: Sensitivity
    message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecentMessagesQuery:
    conversation_id: str
    limit: int


@dataclass(frozen=True)
class ListConversationsQuery:
    user_id: str
    limit: int = 20


@dataclass(frozen=True)
class MessageSubmissionCommand:
    conversation_id: str
    client_message_id: str
    content: str
    sensitivity: Sensitivity
    message_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CreateAssistantRequestCommand:
    conversation_id: str
    user_message_id: str
    client_message_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateAssistantRequestStatusCommand:
    request_id: str
    status: RequestStatus
    assistant_message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
