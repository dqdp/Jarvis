from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from assistant_core.domain.conversations import (
    AppendMessageCommand,
    AssistantResponseCompletion,
    AssistantRequest,
    CompleteAssistantResponseCommand,
    Conversation,
    ConversationMessage,
    ConversationStatus,
    CreateAssistantRequestCommand,
    CreateConversationCommand,
    ListConversationsQuery,
    MessageSubmission,
    MessageSubmissionCommand,
    RecentMessagesQuery,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.messages import MessageRole
from assistant_core.domain.requests import RequestStatus, is_request_status_transition_allowed
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.conversation_store import (
    ClientMessageIdConflict,
    ConversationStoreError,
    InvalidRequestStatusTransition,
)
from assistant_core.storage.event_log import insert_event


_metadata = sa.MetaData()

_conversations = sa.Table(
    "conversations",
    _metadata,
    sa.Column("conversation_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("user_id", sa.Text(), nullable=False),
    sa.Column("title", sa.Text(), nullable=True),
    sa.Column("active_project_namespace", sa.Text(), nullable=True),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)

_messages = sa.Table(
    "messages",
    _metadata,
    sa.Column("message_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column(
        "request_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("assistant_requests.request_id", ondelete="SET NULL"),
        nullable=True,
    ),
    sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("client_message_id", sa.Text(), nullable=True),
    sa.Column("role", sa.Text(), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("sensitivity", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)

_assistant_requests = sa.Table(
    "assistant_requests",
    _metadata,
    sa.Column("request_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column("user_message_id", postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column("assistant_message_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("client_message_id", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("error_code", sa.Text(), nullable=True),
    sa.Column("error_message", sa.Text(), nullable=True),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)


class PostgresConversationStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def create_conversation(
        self,
        command: CreateConversationCommand,
    ) -> Conversation:
        now = _now()
        statement = (
            sa.insert(_conversations)
            .values(
                {
                    "conversation_id": _uuid(command.conversation_id or _new_id()),
                    "user_id": command.user_id,
                    "title": command.title,
                    "active_project_namespace": command.active_project_namespace,
                    "status": ConversationStatus.ACTIVE.value,
                    "created_at": now,
                    "updated_at": now,
                    "metadata": command.metadata,
                },
            )
            .returning(*_conversations.c)
        )

        async with self.engine.begin() as connection:
            row = (await connection.execute(statement)).mappings().one()

        return _row_to_conversation(row)

    async def health_check(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                row = (
                    await connection.execute(
                        sa.text(
                            "select "
                            "to_regclass('public.conversations'), "
                            "to_regclass('public.messages'), "
                            "to_regclass('public.assistant_requests'), "
                            "to_regclass('public.events'), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'messages_request_fk'"
                            "), "
                            "exists ("
                            "select 1 from pg_class c "
                            "join pg_index i on i.indexrelid = c.oid "
                            "where c.relname = 'assistant_requests_user_message_idx' "
                            "and i.indisunique"
                            ")",
                        ),
                    )
                ).one()
        except Exception:
            return False
        return all(value is not None for value in row[:4]) and all(bool(value) for value in row[4:])

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(_conversations).where(
                        _conversations.c.conversation_id == _uuid(conversation_id),
                    ),
                )
            ).mappings().first()
        if row is None:
            return None
        return _row_to_conversation(row)

    async def list_conversations(
        self,
        query: ListConversationsQuery,
    ) -> list[Conversation]:
        statement = (
            sa.select(_conversations)
            .where(_conversations.c.user_id == query.user_id)
            .order_by(_conversations.c.updated_at.desc(), _conversations.c.conversation_id.desc())
            .limit(query.limit)
        )
        async with self.engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return [_row_to_conversation(row) for row in rows]

    async def append_message(self, command: AppendMessageCommand) -> ConversationMessage:
        try:
            async with self.engine.begin() as connection:
                return await _append_message(connection, command)
        except IntegrityError as exc:
            raise ConversationStoreError("message violates conversation integrity") from exc

    async def complete_assistant_response(
        self,
        command: CompleteAssistantResponseCommand,
    ) -> AssistantResponseCompletion:
        try:
            async with self.engine.begin() as connection:
                current = await _select_request(connection, command.request_id, for_update=True)
                current_request = _row_to_request(current)
                if current_request.conversation_id != command.conversation_id:
                    raise ConversationStoreError(
                        "assistant response request must belong to the same conversation",
                    )
                if not is_request_status_transition_allowed(
                    current_request.status,
                    RequestStatus.COMPLETED,
                ):
                    raise InvalidRequestStatusTransition(
                        f"{current_request.status.value} -> {RequestStatus.COMPLETED.value} is not allowed",
                    )
                message = await _append_message(
                    connection,
                    AppendMessageCommand(
                        message_id=command.message_id,
                        conversation_id=command.conversation_id,
                        request_id=command.request_id,
                        role=MessageRole.ASSISTANT,
                        content=command.content,
                        sensitivity=command.sensitivity,
                        metadata=command.metadata,
                    ),
                )
                row = (
                    await connection.execute(
                        sa.update(_assistant_requests)
                        .where(_assistant_requests.c.request_id == _uuid(command.request_id))
                        .values(
                            {
                                "status": RequestStatus.COMPLETED.value,
                                "assistant_message_id": _uuid(message.message_id),
                                "completed_at": _now(),
                                "error_code": None,
                                "error_message": None,
                            },
                        )
                        .returning(*_assistant_requests.c),
                    )
                ).mappings().one()
        except IntegrityError as exc:
            raise ConversationStoreError("assistant response violates conversation integrity") from exc
        return AssistantResponseCompletion(message=message, request=_row_to_request(row))

    async def load_recent_messages(
        self,
        query: RecentMessagesQuery,
    ) -> list[ConversationMessage]:
        statement = (
            sa.select(_messages)
            .where(_messages.c.conversation_id == _uuid(query.conversation_id))
            .order_by(_messages.c.created_at.desc(), _messages.c.message_id.desc())
            .limit(query.limit)
        )

        async with self.engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()

        return list(reversed([_row_to_message(row) for row in rows]))

    async def get_message(self, message_id: str) -> ConversationMessage | None:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(_messages).where(_messages.c.message_id == _uuid(message_id)),
                )
            ).mappings().first()
        if row is None:
            return None
        return _row_to_message(row)

    async def submit_user_message(
        self,
        command: MessageSubmissionCommand,
    ) -> MessageSubmission:
        try:
            async with self.engine.begin() as connection:
                existing_message = await _select_message_by_client_id(
                    connection,
                    command.conversation_id,
                    command.client_message_id,
                )
                content_hash = _content_hash(command.content)
                if existing_message is not None:
                    return await _idempotent_submission_from_existing(
                        connection,
                        existing_message,
                        content_hash,
                        command.sensitivity,
                        command.request_metadata,
                    )

                request_id = command.request_id or _new_id()
                message = await _append_message(
                    connection,
                    AppendMessageCommand(
                        message_id=command.message_id,
                        conversation_id=command.conversation_id,
                        request_id=None,
                        role=MessageRole.USER,
                        content=command.content,
                        sensitivity=command.sensitivity,
                        client_message_id=command.client_message_id,
                        metadata=command.metadata,
                    ),
                )
                request = await _insert_assistant_request(
                    connection,
                    CreateAssistantRequestCommand(
                        request_id=request_id,
                        conversation_id=command.conversation_id,
                        user_message_id=message.message_id,
                        client_message_id=command.client_message_id,
                        metadata=command.request_metadata,
                    ),
                )
                event = await insert_event(
                    connection,
                    _user_message_created_event(message, request, command.client_message_id),
                )
                message = await _set_message_event_id(connection, message.message_id, event.event_id)
        except IntegrityError:
            async with self.engine.begin() as connection:
                existing_message = await _select_message_by_client_id(
                    connection,
                    command.conversation_id,
                    command.client_message_id,
                )
                if existing_message is None:
                    raise
                return await _idempotent_submission_from_existing(
                    connection,
                    existing_message,
                    _content_hash(command.content),
                    command.sensitivity,
                    command.request_metadata,
                )

        return MessageSubmission(user_message=message, request=request)

    async def get_submission_by_client_message_id(
        self,
        conversation_id: str,
        client_message_id: str,
    ) -> MessageSubmission | None:
        async with self.engine.connect() as connection:
            existing_message = await _select_message_by_client_id(
                connection,
                conversation_id,
                client_message_id,
            )
            if existing_message is None:
                return None
            message = _row_to_message(existing_message)
            request = await _select_request_by_user_message(connection, message.message_id)
            return MessageSubmission(
                user_message=message,
                request=_row_to_request(request),
                idempotent_replay=True,
            )

    async def create_assistant_request(
        self,
        command: CreateAssistantRequestCommand,
    ) -> AssistantRequest:
        try:
            async with self.engine.begin() as connection:
                return await _insert_assistant_request(connection, command)
        except IntegrityError as exc:
            raise ConversationStoreError("assistant request violates conversation integrity") from exc

    async def get_assistant_request(self, request_id: str) -> AssistantRequest | None:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(_assistant_requests).where(
                        _assistant_requests.c.request_id == _uuid(request_id),
                    ),
                )
            ).mappings().first()
        if row is None:
            return None
        return _row_to_request(row)

    async def update_assistant_request_status(
        self,
        command: UpdateAssistantRequestStatusCommand,
    ) -> AssistantRequest:
        async with self.engine.begin() as connection:
            current = await _select_request(connection, command.request_id, for_update=True)
            current_request = _row_to_request(current)
            if not is_request_status_transition_allowed(
                current_request.status,
                command.status,
            ):
                raise InvalidRequestStatusTransition(
                    f"{current_request.status.value} -> {command.status.value} is not allowed",
                )

            values: dict[str, Any] = {
                "status": command.status.value,
                "assistant_message_id": _optional_uuid(
                    command.assistant_message_id or current_request.assistant_message_id,
                ),
                "error_code": command.error_code,
                "error_message": command.error_message,
            }
            now = _now()
            if command.status == RequestStatus.RUNNING and current_request.started_at is None:
                values["started_at"] = now
            if command.status in {
                RequestStatus.COMPLETED,
                RequestStatus.FAILED,
                RequestStatus.CANCELLED,
            }:
                values["completed_at"] = now

            assistant_message_id = command.assistant_message_id or current_request.assistant_message_id
            if assistant_message_id is not None:
                await _validate_assistant_message_for_request(
                    connection,
                    current_request,
                    assistant_message_id,
                )

            statement = (
                sa.update(_assistant_requests)
                .where(_assistant_requests.c.request_id == _uuid(command.request_id))
                .values(values)
                .returning(*_assistant_requests.c)
            )
            row = (await connection.execute(statement)).mappings().one()

        return _row_to_request(row)


async def _append_message(
    connection: AsyncConnection,
    command: AppendMessageCommand,
) -> ConversationMessage:
    content_hash = _content_hash(command.content)
    if command.request_id is not None:
        try:
            request_row = await _select_request(connection, command.request_id, for_update=True)
        except KeyError as exc:
            raise ConversationStoreError("message request does not exist") from exc
        request = _row_to_request(request_row)
        if request.conversation_id != command.conversation_id:
            raise ConversationStoreError("message request must belong to the same conversation")
        if (
            command.role == MessageRole.ASSISTANT
            and request.status
            in {RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED}
        ):
            raise ConversationStoreError("assistant message cannot be appended to a terminal request")

    if command.client_message_id is not None:
        existing = await _select_message_by_client_id(
            connection,
            command.conversation_id,
            command.client_message_id,
        )
        if existing is not None:
            message = _row_to_message(existing)
            if message.content_hash != content_hash:
                raise ClientMessageIdConflict(
                    "client_message_id was already used with different content",
                )
            return message

    statement = (
        sa.insert(_messages)
        .values(
            {
                "message_id": _uuid(command.message_id or _new_id()),
                "conversation_id": _uuid(command.conversation_id),
                "request_id": _optional_uuid(command.request_id),
                "event_id": _optional_uuid(command.event_id),
                "client_message_id": command.client_message_id,
                "role": command.role.value,
                "content": command.content,
                "content_hash": content_hash,
                "sensitivity": command.sensitivity.value,
                "created_at": _now(),
                "metadata": command.metadata,
            },
        )
        .returning(*_messages.c)
    )
    row = (await connection.execute(statement)).mappings().one()
    await _touch_conversation(connection, command.conversation_id)
    return _row_to_message(row)


async def _idempotent_submission_from_existing(
    connection: AsyncConnection,
    existing_message: Mapping[str, Any],
    content_hash: str,
    sensitivity: Sensitivity,
    request_metadata: dict[str, Any],
) -> MessageSubmission:
    message = _row_to_message(existing_message)
    if message.content_hash != content_hash:
        raise ClientMessageIdConflict(
            "client_message_id was already used with different content",
        )
    if message.sensitivity != sensitivity:
        raise ClientMessageIdConflict(
            "client_message_id was already used with different sensitivity",
        )
    request = await _select_request_by_user_message(connection, message.message_id)
    if dict(request["metadata"]) != request_metadata:
        raise ClientMessageIdConflict(
            "client_message_id was already used with different runtime options",
        )
    return MessageSubmission(
        user_message=message,
        request=_row_to_request(request),
        idempotent_replay=True,
    )


async def _insert_assistant_request(
    connection: AsyncConnection,
    command: CreateAssistantRequestCommand,
) -> AssistantRequest:
    request_id = command.request_id or _new_id()
    source_message = await _select_message(connection, command.user_message_id, for_update=True)
    if (
        source_message["conversation_id"] != _uuid(command.conversation_id)
        or source_message["role"] != MessageRole.USER.value
    ):
        raise ConversationStoreError("assistant request requires a user message in the same conversation")
    now = _now()
    statement = (
        sa.insert(_assistant_requests)
        .values(
            {
                "request_id": _uuid(request_id),
                "conversation_id": _uuid(command.conversation_id),
                "user_message_id": _uuid(command.user_message_id),
                "assistant_message_id": None,
                "status": RequestStatus.ACCEPTED.value,
                "client_message_id": command.client_message_id,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "error_code": None,
                "error_message": None,
                "metadata": command.metadata,
            },
        )
        .returning(*_assistant_requests.c)
    )
    row = (await connection.execute(statement)).mappings().one()

    await connection.execute(
        sa.update(_messages)
        .where(_messages.c.message_id == _uuid(command.user_message_id))
        .values(request_id=_uuid(request_id)),
    )
    return _row_to_request(row)


async def _select_message(
    connection: AsyncConnection,
    message_id: str,
    *,
    for_update: bool = False,
) -> Mapping[str, Any]:
    statement = sa.select(_messages).where(_messages.c.message_id == _uuid(message_id))
    if for_update:
        statement = statement.with_for_update()
    row = (await connection.execute(statement)).mappings().first()
    if row is None:
        raise ConversationStoreError(f"message {message_id} not found")
    return row


async def _select_message_by_client_id(
    connection: AsyncConnection,
    conversation_id: str,
    client_message_id: str,
) -> Mapping[str, Any] | None:
    statement = (
        sa.select(_messages)
        .where(
            _messages.c.conversation_id == _uuid(conversation_id),
            _messages.c.client_message_id == client_message_id,
        )
        .limit(1)
    )
    return (await connection.execute(statement)).mappings().first()


async def _select_request_by_user_message(
    connection: AsyncConnection,
    user_message_id: str,
) -> Mapping[str, Any]:
    statement = (
        sa.select(_assistant_requests)
        .where(_assistant_requests.c.user_message_id == _uuid(user_message_id))
        .limit(1)
    )
    row = (await connection.execute(statement)).mappings().first()
    if row is None:
        raise KeyError(f"assistant request for message {user_message_id} not found")
    return row


async def _select_request(
    connection: AsyncConnection,
    request_id: str,
    *,
    for_update: bool = False,
) -> Mapping[str, Any]:
    statement = sa.select(_assistant_requests).where(
        _assistant_requests.c.request_id == _uuid(request_id),
    )
    if for_update:
        statement = statement.with_for_update()
    row = (await connection.execute(statement)).mappings().first()
    if row is None:
        raise KeyError(f"assistant request {request_id} not found")
    return row


async def _validate_assistant_message_for_request(
    connection: AsyncConnection,
    request: AssistantRequest,
    assistant_message_id: str,
) -> None:
    message = await _select_message(connection, assistant_message_id, for_update=True)
    if (
        message["conversation_id"] != _uuid(request.conversation_id)
        or message["role"] != MessageRole.ASSISTANT.value
        or message["request_id"] != _uuid(request.request_id)
    ):
        raise ConversationStoreError(
            "assistant_message_id must reference an assistant message for the same request",
        )


async def _set_message_event_id(
    connection: AsyncConnection,
    message_id: str,
    event_id: str,
) -> ConversationMessage:
    statement = (
        sa.update(_messages)
        .where(_messages.c.message_id == _uuid(message_id))
        .values(event_id=_uuid(event_id))
        .returning(*_messages.c)
    )
    row = (await connection.execute(statement)).mappings().one()
    return _row_to_message(row)


async def _touch_conversation(connection: AsyncConnection, conversation_id: str) -> None:
    await connection.execute(
        sa.update(_conversations)
        .where(_conversations.c.conversation_id == _uuid(conversation_id))
        .values(updated_at=_now()),
    )


def _user_message_created_event(
    message: ConversationMessage,
    request: AssistantRequest,
    client_message_id: str,
) -> EventEnvelope:
    now = _now()
    return EventEnvelope(
        event_id=_new_id(),
        event_seq=0,
        event_type=EventType.USER_MESSAGE_CREATED,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=message.conversation_id,
        request_id=request.request_id,
        correlation_id=request.request_id,
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.USER,
        actor_id=None,
        source_component="conversation_store",
        source_node=None,
        sensitivity=message.sensitivity,
        visibility=EventVisibility.USER_VISIBLE,
        idempotency_key=client_message_id,
        payload={
            "message_id": message.message_id,
            "content_hash": message.content_hash,
        },
        metadata={},
    )


def _row_to_conversation(row: Mapping[str, Any]) -> Conversation:
    return Conversation(
        conversation_id=str(row["conversation_id"]),
        user_id=row["user_id"],
        title=row["title"],
        active_project_namespace=row["active_project_namespace"],
        status=ConversationStatus(row["status"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        metadata=dict(row["metadata"]),
    )


def _row_to_message(row: Mapping[str, Any]) -> ConversationMessage:
    return ConversationMessage(
        message_id=str(row["message_id"]),
        conversation_id=str(row["conversation_id"]),
        request_id=_optional_string(row["request_id"]),
        event_id=_optional_string(row["event_id"]),
        client_message_id=row["client_message_id"],
        role=MessageRole(row["role"]),
        content=row["content"],
        content_hash=row["content_hash"],
        sensitivity=Sensitivity(row["sensitivity"]),
        created_at=_datetime(row["created_at"]),
        metadata=dict(row["metadata"]),
    )


def _row_to_request(row: Mapping[str, Any]) -> AssistantRequest:
    return AssistantRequest(
        request_id=str(row["request_id"]),
        conversation_id=str(row["conversation_id"]),
        user_message_id=str(row["user_message_id"]),
        assistant_message_id=_optional_string(row["assistant_message_id"]),
        status=RequestStatus(row["status"]),
        client_message_id=row["client_message_id"],
        created_at=_datetime(row["created_at"]),
        started_at=_optional_datetime(row["started_at"]),
        completed_at=_optional_datetime(row["completed_at"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        metadata=dict(row["metadata"]),
    )


def _content_hash(content: str) -> str:
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"


def _new_id() -> str:
    return str(uuid4())


def _uuid(value: str) -> UUID:
    return UUID(value)


def _optional_uuid(value: str | None) -> UUID | None:
    if value is None:
        return None
    return _uuid(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _now() -> datetime:
    return datetime.now(UTC)


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime, got {type(value).__name__}")


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)
