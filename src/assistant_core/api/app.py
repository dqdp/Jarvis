from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from assistant_core.api.content_authorization import (
    authorize_content_operation as _authorize_content_operation,
)
from assistant_core.api.errors import (
    error_response as _error_response,
    json_response as _json_response,
    validation_errors as _validation_errors,
)
from assistant_core.api.health import health_payload as _health_payload
from assistant_core.api.presenters import (
    approval_payload as _approval_payload,
    content_ingestion_payload as _content_ingestion_payload,
    content_source_payload as _content_source_payload,
    content_status_summary as _content_status_summary,
    conversation_payload as _conversation_payload,
    memory_lifecycle_payload as _memory_lifecycle_payload,
    memory_payload as _memory_payload,
    message_payload as _message_payload,
    request_payload as _request_payload,
    runtime_status_payload as _runtime_status_payload,
)
from assistant_core.api.sse import sse_stream
from assistant_core.config.settings import Settings
from assistant_core.domain.approvals import ApprovalConflict, ApprovalNotFound
from assistant_core.domain.conversations import (
    CreateConversationCommand,
    ListConversationsQuery,
    MessageSubmissionCommand,
    RecentMessagesQuery,
)
from assistant_core.domain.memory import ArchiveMemoryCommand, CreateMemoryCommand, MemoryType
from assistant_core.domain.policy import Capability
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.conversation_store import ClientMessageIdConflict
from assistant_core.ports.memory import MemoryStoreError
from assistant_core.runtime.request_execution import RequestExecutionManager
from assistant_core.runtime.request_metadata import runtime_request_metadata as _runtime_request_metadata


class _StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationCreateBody(_StrictBody):
    title: str | None = None
    active_project_namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageCreateBody(_StrictBody):
    client_message_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    sensitivity: Sensitivity = Sensitivity.PERSONAL
    model_profile: str | None = Field(default=None, min_length=1)
    loop_strategy: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryCreateBody(_StrictBody):
    namespace: str = Field(min_length=1)
    memory_type: MemoryType
    content: str = Field(min_length=1)
    summary: str | None = None
    sensitivity: Sensitivity | None = None
    confidence: float = 1.0
    importance: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecisionBody(_StrictBody):
    reason: str | None = None


def create_app(
    *,
    conversation_store,
    memory_store,
    memory_write=None,
    content_store=None,
    content_retrieval=None,
    settings: Settings,
    runtime=None,
    event_log=None,
    approval_store=None,
    inference_health=None,
    content_ingestion=None,
    policy=None,
    lifespan=None,
) -> FastAPI:
    app = FastAPI(title="Jarvis Assistant Core", version="0.0.0", lifespan=lifespan)
    memory_write_service = memory_write or memory_store
    content_retrieval_service = content_retrieval or content_store
    execution_manager = (
        RequestExecutionManager(
            runtime=runtime,
            conversation_store=conversation_store,
            event_log=event_log,
            settings=settings,
            approval_store=approval_store,
        )
        if runtime is not None and event_log is not None
        else None
    )
    app.state.request_execution_manager = execution_manager

    @app.exception_handler(ClientMessageIdConflict)
    async def _client_message_conflict(_request: Request, exc: ClientMessageIdConflict):
        return _error_response(409, "conflict", str(exc))

    @app.exception_handler(MemoryStoreError)
    async def _memory_store_error(_request: Request, exc: MemoryStoreError):
        return _error_response(400, "invalid_request", str(exc))

    @app.exception_handler(ApprovalNotFound)
    async def _approval_not_found(_request: Request, exc: ApprovalNotFound):
        return _error_response(404, "not_found", str(exc))

    @app.exception_handler(ApprovalConflict)
    async def _approval_conflict(_request: Request, exc: ApprovalConflict):
        return _error_response(409, exc.code, str(exc))

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError):
        return _error_response(400, "invalid_request", str(exc))

    @app.exception_handler(KeyError)
    async def _key_error(_request: Request, exc: KeyError):
        return _error_response(404, "not_found", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        return _error_response(
            400,
            "invalid_request",
            "request validation failed",
            details={"errors": _validation_errors(exc.errors())},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return _error_response(exc.status_code, code, str(exc.detail))

    @app.get("/v1/health")
    async def get_health():
        payload = await _health_payload(
            conversation_store,
            memory_store,
            content_store=content_store,
            inference_health=inference_health,
        )
        return _json_response(200 if payload["status"] == "ready" else 503, payload)

    @app.post("/v1/conversations")
    async def post_conversation(body: ConversationCreateBody | None = Body(default=None)):
        payload = body or ConversationCreateBody()
        conversation = await conversation_store.create_conversation(
            CreateConversationCommand(
                user_id=settings.app.default_user_id,
                title=payload.title,
                active_project_namespace=payload.active_project_namespace,
                metadata=payload.metadata,
            ),
        )
        return _json_response(201, _conversation_payload(conversation))

    @app.get("/v1/conversations")
    async def get_conversations(limit: int = Query(default=20, ge=1, le=100)):
        conversations = await conversation_store.list_conversations(
            ListConversationsQuery(user_id=settings.app.default_user_id, limit=limit),
        )
        return {"conversations": [_conversation_payload(item) for item in conversations]}

    @app.get("/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        conversation = await conversation_store.get_conversation(str(_uuid(conversation_id)))
        if conversation is None:
            raise KeyError("conversation not found")
        return _conversation_payload(conversation)

    @app.post("/v1/conversations/{conversation_id}/messages")
    async def post_message(conversation_id: str, body: MessageCreateBody):
        submission = await conversation_store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=str(_uuid(conversation_id)),
                client_message_id=body.client_message_id,
                content=body.content,
                sensitivity=body.sensitivity,
                metadata=body.metadata,
                request_metadata=_runtime_request_metadata(body, settings),
            ),
        )
        if execution_manager is not None:
            await execution_manager.start(submission.request)
        return _json_response(
            202,
            {
                "request_id": submission.request.request_id,
                "conversation_id": submission.request.conversation_id,
                "user_message_id": submission.user_message.message_id,
                "status": submission.request.status.value,
                "stream_url": f"/v1/requests/{submission.request.request_id}/stream",
                "created_at": submission.request.created_at,
                "idempotent_replay": submission.idempotent_replay,
            },
        )

    @app.get("/v1/conversations/{conversation_id}/messages")
    async def get_conversation_messages(conversation_id: str):
        messages = await conversation_store.load_recent_messages(
            query=RecentMessagesQuery(conversation_id=str(_uuid(conversation_id)), limit=1000),
        )
        return {"messages": [_message_payload(message) for message in messages]}

    @app.get("/v1/requests/{request_id}")
    async def get_request_status(request_id: str):
        request_record = await conversation_store.get_assistant_request(str(_uuid(request_id)))
        if request_record is None:
            raise KeyError("request not found")
        return _request_payload(request_record)

    @app.get("/v1/requests/{request_id}/stream")
    async def stream_request(request_id: str):
        if execution_manager is None:
            return _error_response(404, "not_found", "runtime stream is not configured")
        request_record = await conversation_store.get_assistant_request(str(_uuid(request_id)))
        if request_record is None:
            raise KeyError("request not found")
        return StreamingResponse(
            sse_stream(execution_manager.stream(request_record.request_id)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel_request(request_id: str):
        if execution_manager is None:
            return _error_response(404, "not_found", "runtime stream is not configured")
        request_record = await conversation_store.get_assistant_request(str(_uuid(request_id)))
        if request_record is None:
            raise KeyError("request not found")
        cancelled = await execution_manager.cancel(request_record.request_id)
        return _json_response(202, _request_payload(cancelled))

    @app.get("/v1/approvals/{approval_id}")
    async def get_approval(approval_id: str):
        if approval_store is None:
            return _error_response(404, "not_found", "approval store is not configured")
        await approval_store.expire_stale(now=datetime.now(UTC))
        approval = await approval_store.get_approval(approval_id)
        if approval is None:
            raise ApprovalNotFound("approval not found")
        _ensure_approval_owner(approval, settings.app.default_user_id)
        return _approval_payload(approval)

    @app.post("/v1/approvals/{approval_id}/grant")
    async def grant_approval(
        approval_id: str,
        body: ApprovalDecisionBody | None = Body(default=None),
    ):
        if approval_store is None:
            return _error_response(404, "not_found", "approval store is not configured")
        payload = body or ApprovalDecisionBody()
        await approval_store.expire_stale(now=datetime.now(UTC))
        existing = await approval_store.get_approval(approval_id)
        if existing is None:
            raise ApprovalNotFound("approval not found")
        _ensure_approval_owner(existing, settings.app.default_user_id)
        approval = await approval_store.grant_approval(
            approval_id,
            actor_id=settings.app.default_user_id,
            reason=payload.reason,
        )
        return _approval_payload(approval)

    @app.post("/v1/approvals/{approval_id}/deny")
    async def deny_approval(
        approval_id: str,
        body: ApprovalDecisionBody | None = Body(default=None),
    ):
        if approval_store is None:
            return _error_response(404, "not_found", "approval store is not configured")
        payload = body or ApprovalDecisionBody()
        await approval_store.expire_stale(now=datetime.now(UTC))
        existing = await approval_store.get_approval(approval_id)
        if existing is None:
            raise ApprovalNotFound("approval not found")
        _ensure_approval_owner(existing, settings.app.default_user_id)
        approval = await approval_store.deny_approval(
            approval_id,
            actor_id=settings.app.default_user_id,
            reason=payload.reason,
        )
        return _approval_payload(approval)

    @app.post("/v1/memories")
    async def post_memory(body: MemoryCreateBody):
        sensitivity = body.sensitivity or _namespace_default_sensitivity(settings, body.namespace)
        memory = await memory_write_service.create_memory(
            CreateMemoryCommand(
                namespace=body.namespace,
                memory_type=body.memory_type,
                content=body.content,
                summary=body.summary,
                sensitivity=sensitivity,
                confidence=body.confidence,
                importance=body.importance,
                metadata=body.metadata,
            ),
        )
        return _json_response(201, _memory_payload(memory))

    @app.get("/v1/memories")
    async def get_memories(
        limit: int = Query(default=100, ge=1, le=500),
        query: str | None = None,
    ):
        memories = await memory_store.list_memories(limit=limit, query=query)
        return {"memories": [_memory_payload(memory) for memory in memories]}

    @app.delete("/v1/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        return await _archive_memory(
            memory_store,
            memory_id=memory_id,
            reason="deleted_by_user",
        )

    @app.post("/v1/memories/{memory_id}/archive")
    async def archive_memory(memory_id: str):
        return await _archive_memory(
            memory_store,
            memory_id=memory_id,
            reason="archived_by_user",
        )

    @app.get("/v1/runtime/status")
    async def get_runtime_status():
        return _runtime_status_payload(settings)

    @app.post("/v1/content/project-docs/ingest")
    async def post_content_project_docs_ingest():
        if content_ingestion is None:
            return _error_response(404, "not_found", "content ingestion is not configured")
        denied = await _authorize_content_operation(
            policy,
            settings=settings,
            capability=Capability.CONTENT_INGEST,
            operation="project_docs_ingest",
        )
        if denied is not None:
            return denied
        result = await content_ingestion.ingest()
        return _content_ingestion_payload(result)

    @app.post("/v1/content/project-docs/reindex")
    async def post_content_project_docs_reindex():
        if content_ingestion is None:
            return _error_response(404, "not_found", "content ingestion is not configured")
        denied = await _authorize_content_operation(
            policy,
            settings=settings,
            capability=Capability.CONTENT_INDEX,
            operation="project_docs_reindex",
        )
        if denied is not None:
            return denied
        result = await content_ingestion.ingest()
        return _content_ingestion_payload(result)

    @app.get("/v1/content/sources")
    async def get_content_sources():
        if content_retrieval_service is None:
            return _error_response(404, "not_found", "content store is not configured")
        denied = await _authorize_content_operation(
            policy,
            settings=settings,
            capability=Capability.CONTENT_RETRIEVE,
            operation="content_sources",
        )
        if denied is not None:
            return denied
        sources = await content_retrieval_service.list_sources()
        return {"sources": [_content_source_payload(source) for source in sources]}

    @app.get("/v1/content/status")
    async def get_content_status():
        if content_retrieval_service is None:
            return _error_response(404, "not_found", "content store is not configured")
        denied = await _authorize_content_operation(
            policy,
            settings=settings,
            capability=Capability.CONTENT_RETRIEVE,
            operation="content_status",
        )
        if denied is not None:
            return denied
        sources = await content_retrieval_service.list_sources()
        chunks = await content_retrieval_service.list_chunks()
        return {
            "sources": _content_status_summary(sources),
            "chunks": _content_status_summary(chunks),
        }

    return app


async def _archive_memory(memory_store, *, memory_id: str, reason: str):
    normalized_memory_id = str(_uuid(memory_id))
    await memory_store.archive_memory(
        ArchiveMemoryCommand(memory_id=normalized_memory_id, reason=reason),
    )
    memory = await memory_store.get_memory(normalized_memory_id)
    if memory is None:
        raise KeyError("memory not found")
    return _memory_lifecycle_payload(memory)


def _uuid(value: str) -> UUID:
    return UUID(value)


def _namespace_default_sensitivity(settings: Settings, namespace: str) -> Sensitivity:
    namespace_config = settings.memory.namespaces.get(namespace)
    if namespace_config is None:
        return Sensitivity.PERSONAL
    return Sensitivity(namespace_config.sensitivity)


def _ensure_approval_owner(approval, user_id: str) -> None:
    if approval.requested_by != user_id:
        raise ApprovalConflict("approval belongs to another user", code="approval_subject_mismatch")
