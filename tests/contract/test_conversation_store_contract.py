from __future__ import annotations

import asyncio
import os
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.domain.conversations import (
    AppendMessageCommand,
    CompleteAssistantResponseCommand,
    CreateAssistantRequestCommand,
    CreateConversationCommand,
    ConversationStatus,
    ListConversationsQuery,
    MessageSubmissionCommand,
    RecentMessagesQuery,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.messages import MessageRole
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.conversation_store import (
    ClientMessageIdConflict,
    ConversationStoreError,
    InvalidRequestStatusTransition,
)
from assistant_core.ports.event_log import EventFilter
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-conversation-contract:{label}"))


async def _truncate_storage(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table assistant_requests, messages, conversations, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


@pytest.fixture
def store():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_storage(database_url))
    engine = create_database_engine(database_url)
    try:
        yield PostgresConversationStore(engine)
    finally:
        asyncio.run(engine.dispose())


async def _conversation(store: PostgresConversationStore, label: str = "conv"):
    return await store.create_conversation(
        CreateConversationCommand(
            conversation_id=_id(label),
            user_id="user-1",
            title="Phase 1",
            active_project_namespace="project.jarvis",
            metadata={"source": "contract"},
        ),
    )


def test_create_conversation(store) -> None:
    conversation = asyncio.run(_conversation(store))

    assert conversation.conversation_id == _id("conv")
    assert conversation.status == ConversationStatus.ACTIVE
    assert conversation.active_project_namespace == "project.jarvis"
    assert conversation.metadata == {"source": "contract"}


def test_list_conversations_returns_most_recent_first(store) -> None:
    async def scenario():
        older = await _conversation(store, "older")
        newer = await _conversation(store, "newer")
        conversations = await store.list_conversations(
            ListConversationsQuery(user_id="user-1", limit=10),
        )
        return older, newer, conversations

    older, newer, conversations = asyncio.run(scenario())

    assert [conversation.conversation_id for conversation in conversations[:2]] == [
        newer.conversation_id,
        older.conversation_id,
    ]


def test_conversation_updated_at_moves_when_message_is_appended(store) -> None:
    async def scenario():
        conversation = await _conversation(store, "touch-on-message")
        await store.append_message(
            AppendMessageCommand(
                message_id=_id("touch-message"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="touch conversation",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        refreshed = await store.get_conversation(conversation.conversation_id)
        return conversation, refreshed

    conversation, refreshed = asyncio.run(scenario())

    assert refreshed is not None
    assert refreshed.updated_at > conversation.updated_at


def test_append_user_message(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        return await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-user"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="hello",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-1",
                metadata={"channel": "test"},
            ),
        )

    message = asyncio.run(scenario())

    assert message.role == MessageRole.USER
    assert message.content == "hello"
    assert message.client_message_id == "client-1"
    assert message.content_hash.startswith("sha256:")


def test_append_assistant_message(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        user_message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-assistant-user"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="request please",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-assistant-user",
            ),
        )
        request = await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-1"),
                conversation_id=conversation.conversation_id,
                user_message_id=user_message.message_id,
                client_message_id="client-assistant-user",
            ),
        )
        return await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-assistant"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content="hello back",
                sensitivity=Sensitivity.PROJECT,
                request_id=request.request_id,
            ),
        )

    message = asyncio.run(scenario())

    assert message.role == MessageRole.ASSISTANT
    assert message.request_id == _id("req-1")


def test_append_message_rejects_unknown_request_id(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-orphan"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content="orphan",
                sensitivity=Sensitivity.PROJECT,
                request_id=_id("req-missing"),
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_load_messages_ordered(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        first = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-1"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="first",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        second = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-2"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content="second",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=conversation.conversation_id, limit=10),
        )
        return messages, [first, second]

    messages, expected = asyncio.run(scenario())

    assert messages == expected


def test_load_recent_messages_returns_latest_limited_window_in_chronological_order(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        created = []
        for index in range(4):
            created.append(
                await store.append_message(
                    AppendMessageCommand(
                        message_id=_id(f"recent-{index}"),
                        conversation_id=conversation.conversation_id,
                        role=MessageRole.USER,
                        content=f"message {index}",
                        sensitivity=Sensitivity.PROJECT,
                    ),
                ),
            )
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=conversation.conversation_id, limit=2),
        )
        return created, messages

    created, messages = asyncio.run(scenario())

    assert messages == created[-2:]


def test_client_message_id_idempotency_same_content(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        command = MessageSubmissionCommand(
            conversation_id=conversation.conversation_id,
            client_message_id="client-repeat",
            content="same content",
            sensitivity=Sensitivity.PROJECT,
        )
        first = await store.submit_user_message(command)
        second = await store.submit_user_message(command)
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=conversation.conversation_id, limit=10),
        )
        return first, second, messages

    first, second, messages = asyncio.run(scenario())

    assert second.idempotent_replay is True
    assert second.request.request_id == first.request.request_id
    assert second.user_message.message_id == first.user_message.message_id
    assert messages == [first.user_message]


def test_concurrent_client_message_id_replay_returns_single_request(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        command = MessageSubmissionCommand(
            conversation_id=conversation.conversation_id,
            client_message_id="client-concurrent",
            content="same concurrent content",
            sensitivity=Sensitivity.PROJECT,
        )
        submissions = await asyncio.gather(
            *(store.submit_user_message(command) for _ in range(8)),
        )
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=conversation.conversation_id, limit=10),
        )
        return submissions, messages

    submissions, messages = asyncio.run(scenario())

    assert len({submission.request.request_id for submission in submissions}) == 1
    assert len({submission.user_message.message_id for submission in submissions}) == 1
    assert len(messages) == 1


def test_client_message_id_conflict_different_content(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-conflict",
                content="original",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-conflict",
                content="changed",
                sensitivity=Sensitivity.PROJECT,
            ),
        )

    with pytest.raises(ClientMessageIdConflict):
        asyncio.run(scenario())


def test_client_message_id_conflict_different_sensitivity(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-conflict-sensitivity",
                content="same content",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-conflict-sensitivity",
                content="same content",
                sensitivity=Sensitivity.SECRET,
            ),
        )

    with pytest.raises(ClientMessageIdConflict):
        asyncio.run(scenario())


def test_client_message_id_conflict_different_request_metadata(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-conflict-metadata",
                content="same content",
                sensitivity=Sensitivity.PROJECT,
                request_metadata={"loop_strategy": "memory_augmented_answer"},
            ),
        )
        await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-conflict-metadata",
                content="same content",
                sensitivity=Sensitivity.PROJECT,
                request_metadata={"loop_strategy": "tool_react_loop"},
            ),
        )

    with pytest.raises(ClientMessageIdConflict):
        asyncio.run(scenario())


def test_client_message_id_is_copied_to_event_idempotency_key(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        submission = await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-event",
                content="event-linked",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        event_log = PostgresEventLog(store.engine)
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        return submission, events

    submission, events = asyncio.run(scenario())

    assert len(events) == 1
    assert events[0].event_type == EventType.USER_MESSAGE_CREATED
    assert events[0].idempotency_key == "client-event"
    assert submission.user_message.event_id == events[0].event_id


def test_create_assistant_request(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-request"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="request please",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-request",
            ),
        )
        return await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-create"),
                conversation_id=conversation.conversation_id,
                user_message_id=message.message_id,
                client_message_id="client-request",
            ),
        )

    request = asyncio.run(scenario())

    assert request.status == RequestStatus.ACCEPTED
    assert request.user_message_id == _id("msg-request")


def test_one_assistant_request_per_user_message(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-one-request"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="request once",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-one-request",
            ),
        )
        await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-one"),
                conversation_id=conversation.conversation_id,
                user_message_id=message.message_id,
                client_message_id="client-one-request",
            ),
        )
        await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-two"),
                conversation_id=conversation.conversation_id,
                user_message_id=message.message_id,
                client_message_id="client-one-request",
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_assistant_request_rejects_user_message_from_other_conversation(store) -> None:
    async def scenario():
        first = await _conversation(store, label="foreign-message-source")
        second = await _conversation(store, label="foreign-message-target")
        message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-foreign-request"),
                conversation_id=first.conversation_id,
                role=MessageRole.USER,
                content="belongs elsewhere",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-foreign-request",
            ),
        )
        await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-foreign"),
                conversation_id=second.conversation_id,
                user_message_id=message.message_id,
                client_message_id="client-foreign-request",
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_assistant_request_rejects_non_user_message(store) -> None:
    async def scenario():
        conversation = await _conversation(store, label="non-user-request")
        message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-non-user-request"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.ASSISTANT,
                content="assistant is not request source",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-non-user"),
                conversation_id=conversation.conversation_id,
                user_message_id=message.message_id,
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_append_message_rejects_request_from_other_conversation(store) -> None:
    async def scenario():
        first = await _conversation(store, label="append-request-source")
        second = await _conversation(store, label="append-request-target")
        user_message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-request-source"),
                conversation_id=first.conversation_id,
                role=MessageRole.USER,
                content="request source",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-request-source",
            ),
        )
        request = await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-append-foreign"),
                conversation_id=first.conversation_id,
                user_message_id=user_message.message_id,
                client_message_id="client-request-source",
            ),
        )
        await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-append-foreign"),
                conversation_id=second.conversation_id,
                request_id=request.request_id,
                role=MessageRole.ASSISTANT,
                content="wrong conversation",
                sensitivity=Sensitivity.PROJECT,
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_complete_request_rejects_assistant_message_from_other_conversation(store) -> None:
    async def scenario():
        first = await _conversation(store, label="complete-request-source")
        second = await _conversation(store, label="complete-request-target")
        user_message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-complete-source"),
                conversation_id=first.conversation_id,
                role=MessageRole.USER,
                content="complete source",
                sensitivity=Sensitivity.PROJECT,
                client_message_id="client-complete-source",
            ),
        )
        request = await store.create_assistant_request(
            CreateAssistantRequestCommand(
                request_id=_id("req-complete-foreign"),
                conversation_id=first.conversation_id,
                user_message_id=user_message.message_id,
                client_message_id="client-complete-source",
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        assistant_message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-complete-foreign"),
                conversation_id=second.conversation_id,
                role=MessageRole.ASSISTANT,
                content="wrong conversation",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.COMPLETED,
                assistant_message_id=assistant_message.message_id,
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_complete_request_rejects_non_assistant_message_as_response(store) -> None:
    async def scenario():
        conversation = await _conversation(store, label="complete-non-assistant")
        submission = await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-complete-non-assistant",
                content="complete non assistant",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        user_message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-complete-non-assistant"),
                conversation_id=conversation.conversation_id,
                role=MessageRole.USER,
                content="not an assistant response",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.COMPLETED,
                assistant_message_id=user_message.message_id,
            ),
        )

    with pytest.raises(ConversationStoreError):
        asyncio.run(scenario())


def test_request_status_transitions(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        request = await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-status",
                content="run it",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        running = await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        waiting_approval = await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request.request_id,
                status=RequestStatus.WAITING_APPROVAL,
            ),
        )
        resumed = await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        assistant_message = await store.append_message(
            AppendMessageCommand(
                message_id=_id("msg-completed"),
                conversation_id=conversation.conversation_id,
                request_id=request.request.request_id,
                role=MessageRole.ASSISTANT,
                content="done",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        completed = await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request.request_id,
                status=RequestStatus.COMPLETED,
                assistant_message_id=assistant_message.message_id,
            ),
        )
        with pytest.raises(InvalidRequestStatusTransition):
            await store.update_assistant_request_status(
                UpdateAssistantRequestStatusCommand(
                    request_id=completed.request_id,
                    status=RequestStatus.RUNNING,
                ),
            )
        return running, waiting_approval, resumed, completed

    running, waiting_approval, resumed, completed = asyncio.run(scenario())

    assert running.status == RequestStatus.RUNNING
    assert running.started_at is not None
    assert waiting_approval.status == RequestStatus.WAITING_APPROVAL
    assert waiting_approval.started_at == running.started_at
    assert resumed.status == RequestStatus.RUNNING
    assert resumed.started_at == running.started_at
    assert completed.status == RequestStatus.COMPLETED
    assert completed.completed_at is not None


def test_complete_assistant_response_appends_message_and_completes_request_atomically(
    store,
) -> None:
    async def scenario():
        conversation = await _conversation(store, label="complete-response")
        submission = await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-complete-response",
                content="complete response atomically",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        completion = await store.complete_assistant_response(
            CompleteAssistantResponseCommand(
                request_id=submission.request.request_id,
                conversation_id=conversation.conversation_id,
                content="done atomically",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=conversation.conversation_id, limit=10),
        )
        return completion, messages

    completion, messages = asyncio.run(scenario())

    assert completion.request.status == RequestStatus.COMPLETED
    assert completion.request.assistant_message_id == completion.message.message_id
    assert completion.message.request_id == completion.request.request_id
    assert [message.role for message in messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


def test_request_can_transition_from_running_to_cancelled(store) -> None:
    async def scenario():
        conversation = await _conversation(store)
        submission = await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-cancel-status",
                content="cancel it",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        return await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.CANCELLED,
                error_code="cancelled",
                error_message="request cancelled",
            ),
        )

    cancelled = asyncio.run(scenario())

    assert cancelled.status == RequestStatus.CANCELLED
    assert cancelled.completed_at is not None
