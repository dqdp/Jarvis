from __future__ import annotations

from typing import Protocol

from assistant_core.domain.conversations import (
    AppendMessageCommand,
    AssistantResponseCompletion,
    AssistantRequest,
    CompleteAssistantResponseCommand,
    Conversation,
    ConversationMessage,
    CreateAssistantRequestCommand,
    CreateConversationCommand,
    ListConversationsQuery,
    MessageSubmission,
    MessageSubmissionCommand,
    RecentMessagesQuery,
    UpdateAssistantRequestStatusCommand,
)


class ConversationStoreError(Exception):
    """Base error for conversation store contract violations."""


class ClientMessageIdConflict(ConversationStoreError):
    """Raised when a client_message_id is reused with different content."""


class InvalidRequestStatusTransition(ConversationStoreError):
    """Raised when an assistant request status transition is not allowed."""


class ConversationStorePort(Protocol):
    async def create_conversation(
        self,
        command: CreateConversationCommand,
    ) -> Conversation: ...

    async def get_conversation(self, conversation_id: str) -> Conversation | None: ...

    async def list_conversations(
        self,
        query: ListConversationsQuery,
    ) -> list[Conversation]: ...

    async def append_message(
        self,
        command: AppendMessageCommand,
    ) -> ConversationMessage: ...

    async def complete_assistant_response(
        self,
        command: CompleteAssistantResponseCommand,
    ) -> AssistantResponseCompletion: ...

    async def load_recent_messages(
        self,
        query: RecentMessagesQuery,
    ) -> list[ConversationMessage]: ...

    async def submit_user_message(
        self,
        command: MessageSubmissionCommand,
    ) -> MessageSubmission: ...

    async def create_assistant_request(
        self,
        command: CreateAssistantRequestCommand,
    ) -> AssistantRequest: ...

    async def get_assistant_request(self, request_id: str) -> AssistantRequest | None: ...

    async def update_assistant_request_status(
        self,
        command: UpdateAssistantRequestStatusCommand,
    ) -> AssistantRequest: ...
