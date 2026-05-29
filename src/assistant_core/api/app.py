from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from typing import Any
from uuid import UUID, uuid4

from fastapi import Body, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from assistant_core.config.settings import Settings
from assistant_core.domain.conversations import (
    CreateConversationCommand,
    ListConversationsQuery,
    MessageSubmissionCommand,
    RecentMessagesQuery,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.memory import ArchiveMemoryCommand, CreateMemoryCommand, MemoryType
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.conversation_store import (
    ClientMessageIdConflict,
    InvalidRequestStatusTransition,
)
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.memory import MemoryStoreError


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


def create_app(
    *,
    conversation_store,
    memory_store,
    settings: Settings,
    runtime=None,
    event_log=None,
    lifespan=None,
) -> FastAPI:
    app = FastAPI(title="Jarvis Assistant Core", version="0.0.0", lifespan=lifespan)
    execution_manager = (
        _RequestExecutionManager(
            runtime=runtime,
            conversation_store=conversation_store,
            event_log=event_log,
            settings=settings,
        )
        if runtime is not None and event_log is not None
        else None
    )

    @app.exception_handler(ClientMessageIdConflict)
    async def _client_message_conflict(_request: Request, exc: ClientMessageIdConflict):
        return _error_response(409, "conflict", str(exc))

    @app.exception_handler(MemoryStoreError)
    async def _memory_store_error(_request: Request, exc: MemoryStoreError):
        return _error_response(400, "invalid_request", str(exc))

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
        payload = await _health_payload(conversation_store, memory_store)
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
    async def get_conversations(limit: int = 20):
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
            execution_manager.stream(request_record.request_id),
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

    @app.post("/v1/memories")
    async def post_memory(body: MemoryCreateBody):
        sensitivity = body.sensitivity or _namespace_default_sensitivity(settings, body.namespace)
        memory = await memory_store.create_memory(
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
    async def get_memories(limit: int = 100, query: str | None = None):
        memories = await memory_store.list_memories(limit=limit, query=query)
        return {"memories": [_memory_payload(memory) for memory in memories]}

    @app.delete("/v1/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        normalized_memory_id = str(_uuid(memory_id))
        await memory_store.archive_memory(
            ArchiveMemoryCommand(memory_id=normalized_memory_id, reason="deleted_by_user"),
        )
        memory = await memory_store.get_memory(normalized_memory_id)
        if memory is None:
            raise KeyError("memory not found")
        return _memory_payload(memory)

    @app.get("/v1/runtime/status")
    async def get_runtime_status():
        return _runtime_status_payload(settings)

    return app


class _RequestExecutionManager:
    def __init__(self, *, runtime, conversation_store, event_log, settings: Settings) -> None:
        self._runtime = runtime
        self._conversation_store = conversation_store
        self._event_log = event_log
        self._settings = settings
        self._tasks: dict[str, asyncio.Task] = {}
        self._events: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._conditions: dict[str, asyncio.Condition] = {}
        self._lock: asyncio.Lock | None = None

    async def start(self, request_record) -> None:
        lock = self._start_lock()
        async with lock:
            current = await self._conversation_store.get_assistant_request(request_record.request_id)
            if current is None or current.status != RequestStatus.ACCEPTED:
                return
            task = self._tasks.get(request_record.request_id)
            if task is not None and not task.done():
                return
            self._tasks[current.request_id] = asyncio.create_task(
                self._execute_request(current.request_id),
            )

    async def cancel(self, request_id: str):
        request_record = await self._conversation_store.get_assistant_request(request_id)
        if request_record is None:
            raise KeyError("request not found")
        if request_record.status in _TERMINAL_REQUEST_STATUSES:
            return request_record
        cancelled = await self._mark_cancelled(request_record)
        task = self._tasks.get(request_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=0.1)
        return cancelled

    async def stream(self, request_id: str):
        index = 0
        heartbeat_seconds = max(0.001, float(self._settings.api.sse_heartbeat_seconds))
        while True:
            buffered = self._events.get(request_id, [])
            while index < len(buffered):
                event_type, data = buffered[index]
                index += 1
                yield _sse(event_type, data)
                if event_type in _TERMINAL_EVENT_TYPES:
                    return

            request_record = await self._conversation_store.get_assistant_request(request_id)
            if request_record is None:
                raise KeyError("request not found")
            if request_record.status in _TERMINAL_REQUEST_STATUSES:
                buffered = self._events.get(request_id, [])
                while index < len(buffered):
                    event_type, data = buffered[index]
                    index += 1
                    yield _sse(event_type, data)
                    if event_type in _TERMINAL_EVENT_TYPES:
                        return
                if index == 0:
                    async for item in _event_log_stream(self._event_log, request_record):
                        yield item
                elif not _has_terminal_event(buffered[:index]):
                    yield _terminal_sse(request_record)
                return

            task = self._tasks.get(request_id)
            if (
                task is None or task.done()
            ) and request_record.status in {
                RequestStatus.ACCEPTED,
                RequestStatus.RUNNING,
            } and _request_execution_age_seconds(request_record) >= float(
                self._settings.api.request_timeout_seconds,
            ):
                error_code = (
                    "orphaned_running_request"
                    if request_record.status == RequestStatus.RUNNING
                    else "orphaned_accepted_request"
                )
                failed = await self._mark_failed(
                    request_record,
                    code=error_code,
                    message="request execution task is not active",
                )
                yield _terminal_sse(failed)
                return

            condition = self._condition(request_id)
            wait_seconds = heartbeat_seconds
            task = self._tasks.get(request_id)
            if task is None or task.done():
                wait_seconds = min(heartbeat_seconds, 0.05)
            try:
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=wait_seconds)
            except TimeoutError:
                yield _sse("heartbeat", {"request_id": request_id})

    async def _execute_request(self, request_id: str) -> None:
        request_record = await self._conversation_store.get_assistant_request(request_id)
        if request_record is None:
            return
        try:
            command = await self._runtime_command(request_record)
            await self._execute(command)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._mark_failed(
                request_record,
                code="background_task_failed",
                message="request failed in background execution",
            )

    async def _execute(self, command) -> None:
        try:
            async for event in self._runtime.stream_turn(command):
                await self._publish(command.request_id, event.event_type, event.data)
        except asyncio.CancelledError:
            request_record = await self._conversation_store.get_assistant_request(command.request_id)
            if request_record is not None and request_record.status not in _TERMINAL_REQUEST_STATUSES:
                await self._mark_cancelled(request_record)
            return
        except Exception:
            request_record = await self._conversation_store.get_assistant_request(command.request_id)
            if request_record is not None:
                await self._mark_failed(
                    request_record,
                    code="background_task_failed",
                    message="request failed in background execution",
                )

    async def _runtime_command(self, request_record):
        from assistant_core.runtime.agent_runtime import RuntimeTurnCommand

        messages = await self._conversation_store.load_recent_messages(
            RecentMessagesQuery(conversation_id=request_record.conversation_id, limit=1000),
        )
        user_message = next(
            message for message in messages if message.message_id == request_record.user_message_id
        )
        conversation = await self._conversation_store.get_conversation(request_record.conversation_id)
        if conversation is None:
            raise KeyError("conversation not found")
        return RuntimeTurnCommand(
            request_id=request_record.request_id,
            conversation_id=request_record.conversation_id,
            user_message_id=request_record.user_message_id,
            user_id=self._settings.app.default_user_id,
            user_input=user_message.content,
            active_project_namespace=conversation.active_project_namespace,
            current_message_sensitivity=user_message.sensitivity,
        )

    async def _mark_cancelled(self, request_record):
        current = await self._conversation_store.get_assistant_request(request_record.request_id)
        if current is None:
            raise KeyError("request not found")
        if current.status in _TERMINAL_REQUEST_STATUSES:
            return current
        try:
            cancelled = await self._conversation_store.update_assistant_request_status(
                UpdateAssistantRequestStatusCommand(
                    request_id=current.request_id,
                    status=RequestStatus.CANCELLED,
                    error_code="cancelled",
                    error_message="request cancelled",
                ),
            )
        except InvalidRequestStatusTransition:
            refreshed = await self._conversation_store.get_assistant_request(current.request_id)
            if refreshed is None:
                raise KeyError("request not found")
            return refreshed
        event = await self._event_log.append(_request_cancelled_event(cancelled))
        await self._publish(
            cancelled.request_id,
            EventType.REQUEST_PROCESSING_CANCELLED.value,
            {
                "request_id": cancelled.request_id,
                "event_id": event.event_id,
                "error": {
                    "code": "cancelled",
                    "message": "request cancelled",
                    "request_id": cancelled.request_id,
                    "details": {},
                },
            },
        )
        return cancelled

    async def _mark_failed(self, request_record, *, code: str, message: str):
        current = await self._conversation_store.get_assistant_request(request_record.request_id)
        if current is None:
            raise KeyError("request not found")
        if current.status in _TERMINAL_REQUEST_STATUSES:
            return current
        try:
            failed = await self._conversation_store.update_assistant_request_status(
                UpdateAssistantRequestStatusCommand(
                    request_id=current.request_id,
                    status=RequestStatus.FAILED,
                    error_code=code,
                    error_message=message,
                ),
            )
        except InvalidRequestStatusTransition:
            refreshed = await self._conversation_store.get_assistant_request(current.request_id)
            if refreshed is None:
                raise KeyError("request not found")
            return refreshed
        event = await self._event_log.append(_request_failed_event(failed, code=code))
        await self._publish(
            failed.request_id,
            EventType.REQUEST_PROCESSING_FAILED.value,
            {
                "request_id": failed.request_id,
                "event_id": event.event_id,
                "error": {
                    "code": failed.error_code,
                    "message": failed.error_message,
                    "request_id": failed.request_id,
                    "details": {},
                },
            },
        )
        return failed

    async def _publish(self, request_id: str, event_type: str, data: dict[str, Any]) -> None:
        payload = dict(data)
        payload.setdefault("request_id", request_id)
        self._events.setdefault(request_id, []).append(
            (event_type, _public_stream_data(event_type, payload)),
        )
        condition = self._condition(request_id)
        async with condition:
            condition.notify_all()

    def _condition(self, request_id: str) -> asyncio.Condition:
        condition = self._conditions.get(request_id)
        if condition is None:
            condition = asyncio.Condition()
            self._conditions[request_id] = condition
        return condition

    def _start_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


_TERMINAL_REQUEST_STATUSES = {
    RequestStatus.COMPLETED,
    RequestStatus.FAILED,
    RequestStatus.CANCELLED,
}
_TERMINAL_EVENT_TYPES = {
    EventType.REQUEST_PROCESSING_COMPLETED.value,
    EventType.REQUEST_PROCESSING_FAILED.value,
    EventType.REQUEST_PROCESSING_CANCELLED.value,
}
_STREAM_REPLAY_EVENT_TYPES = {
    EventType.REQUEST_PROCESSING_STARTED.value,
    EventType.CONTEXT_ASSEMBLY_STARTED.value,
    EventType.MEMORY_RETRIEVED.value,
    EventType.MEMORY_RETRIEVAL_FAILED.value,
    EventType.CONTEXT_ASSEMBLED.value,
    EventType.MODEL_REQUEST_CREATED.value,
    EventType.MODEL_RESPONSE_RECEIVED.value,
    EventType.ASSISTANT_MESSAGE_CREATED.value,
    EventType.REQUEST_PROCESSING_COMPLETED.value,
    EventType.REQUEST_PROCESSING_FAILED.value,
    EventType.REQUEST_PROCESSING_CANCELLED.value,
}
_PUBLIC_STREAM_FIELDS = {
    "token": ("request_id", "delta"),
    EventType.REQUEST_PROCESSING_STARTED.value: ("request_id", "event_id"),
    EventType.CONTEXT_ASSEMBLY_STARTED.value: ("request_id", "event_id"),
    EventType.MEMORY_RETRIEVED.value: ("request_id", "event_id"),
    EventType.MEMORY_RETRIEVAL_FAILED.value: ("request_id", "event_id", "error"),
    EventType.CONTEXT_ASSEMBLED.value: (
        "request_id",
        "event_id",
        "context_manifest_id",
        "degraded",
        "token_estimate",
    ),
    EventType.MODEL_REQUEST_CREATED.value: ("request_id", "event_id", "context_manifest_id"),
    EventType.MODEL_RESPONSE_RECEIVED.value: ("request_id", "event_id", "context_manifest_id"),
    EventType.ASSISTANT_MESSAGE_CREATED.value: (
        "request_id",
        "event_id",
        "message_id",
        "content_hash",
    ),
    EventType.REQUEST_PROCESSING_COMPLETED.value: (
        "request_id",
        "event_id",
        "assistant_message_id",
    ),
    EventType.REQUEST_PROCESSING_FAILED.value: ("request_id", "event_id", "error"),
    EventType.REQUEST_PROCESSING_CANCELLED.value: ("request_id", "event_id", "error"),
}


def _json_response(status_code: int, payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


def _error_response(
    status: int,
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return _json_response(
        status,
        {
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "details": details or {},
            },
        },
    )


def _validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "loc": list(error.get("loc", [])),
            "msg": error.get("msg", "validation error"),
            "type": error.get("type", "validation_error"),
        }
        for error in errors
    ]


async def _health_payload(conversation_store, memory_store) -> dict[str, Any]:
    checks = {
        "conversation_store": await _component_health(conversation_store),
        "memory_store": await _component_health(memory_store),
    }
    ready = all(value == "ok" for value in checks.values())
    return {
        "status": "ready" if ready else "not_ready",
        "liveness": {"status": "ok"},
        "readiness": {"status": "ok" if ready else "failed", "checks": checks},
    }


async def _component_health(component) -> str:
    health_check = getattr(component, "health_check", None)
    if health_check is None:
        return "ok"
    try:
        result = health_check()
        if hasattr(result, "__await__"):
            result = await result
    except Exception:
        return "failed"
    return "ok" if result else "failed"


async def _event_log_stream(event_log, request_record):
    yielded_terminal = False
    for event in await event_log.query(EventFilter(request_id=request_record.request_id)):
        if event.event_type.value not in _STREAM_REPLAY_EVENT_TYPES:
            continue
        if event.event_type.value in _TERMINAL_EVENT_TYPES:
            yielded_terminal = True
        yield _event_sse(event)
    if not yielded_terminal:
        yield _terminal_sse(request_record)


async def _terminal_event_stream(request_record):
    yield _terminal_sse(request_record)


def _terminal_sse(request_record) -> str:
    if request_record.status == RequestStatus.COMPLETED:
        return _sse(
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {
                "request_id": request_record.request_id,
                "assistant_message_id": request_record.assistant_message_id,
            },
        )
    if request_record.status == RequestStatus.CANCELLED:
        return _sse(
            EventType.REQUEST_PROCESSING_CANCELLED.value,
            {
                "request_id": request_record.request_id,
                "error": {
                    "code": "cancelled",
                    "message": "request cancelled",
                    "request_id": request_record.request_id,
                    "details": {},
                },
            },
        )
    return _sse(
        EventType.REQUEST_PROCESSING_FAILED.value,
        {
            "request_id": request_record.request_id,
            "error": {
                "code": request_record.error_code,
                "message": request_record.error_message,
                "request_id": request_record.request_id,
                "details": {},
            },
        },
    )


def _event_sse(event) -> str:
    return _sse(
        event.event_type.value,
        _public_stream_data(
            event.event_type.value,
            {"request_id": event.request_id, "event_id": event.event_id, **event.payload},
        ),
    )


def _public_stream_data(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    public_data = dict(data)
    if event_type == EventType.REQUEST_PROCESSING_FAILED.value and "error" not in public_data:
        public_data["error"] = {
            "code": public_data.get("error_code") or public_data.get("error_type"),
            "message": "request failed",
            "request_id": public_data.get("request_id"),
            "details": {},
        }
    if event_type == EventType.MEMORY_RETRIEVAL_FAILED.value and "error" not in public_data:
        public_data["error"] = {
            "code": public_data.get("error_code") or public_data.get("error_type"),
            "message": "memory retrieval failed",
            "request_id": public_data.get("request_id"),
            "details": {},
        }
    if event_type == EventType.REQUEST_PROCESSING_CANCELLED.value and "error" not in public_data:
        public_data["error"] = {
            "code": "cancelled",
            "message": "request cancelled",
            "request_id": public_data.get("request_id"),
            "details": {},
        }

    fields = _PUBLIC_STREAM_FIELDS.get(event_type, ("request_id", "event_id"))
    return {
        field: public_data[field]
        for field in fields
        if field in public_data and public_data[field] is not None
    }


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=_json_value)}\n\n"


def _has_terminal_event(events: list[tuple[str, dict[str, Any]]]) -> bool:
    return any(event_type in _TERMINAL_EVENT_TYPES for event_type, _ in events)


def _request_execution_age_seconds(request_record) -> float:
    anchor = request_record.started_at or request_record.created_at
    return (datetime.now(UTC) - anchor).total_seconds()


def _request_cancelled_event(request_record) -> EventEnvelope:
    now = datetime.now(UTC)
    return EventEnvelope(
        event_id=str(uuid4()),
        event_seq=0,
        event_type=EventType.REQUEST_PROCESSING_CANCELLED,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=request_record.conversation_id,
        request_id=request_record.request_id,
        correlation_id=request_record.request_id,
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="assistant_api",
        source_node=None,
        sensitivity=Sensitivity.PROJECT,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={"error_code": "cancelled"},
        metadata={},
    )


def _request_failed_event(request_record, *, code: str) -> EventEnvelope:
    now = datetime.now(UTC)
    return EventEnvelope(
        event_id=str(uuid4()),
        event_seq=0,
        event_type=EventType.REQUEST_PROCESSING_FAILED,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=request_record.conversation_id,
        request_id=request_record.request_id,
        correlation_id=request_record.request_id,
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="assistant_api",
        source_node=None,
        sensitivity=Sensitivity.PROJECT,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={"error_code": code},
        metadata={},
    )


def _uuid(value: str) -> UUID:
    return UUID(value)


def _namespace_default_sensitivity(settings: Settings, namespace: str) -> Sensitivity:
    namespace_config = settings.memory.namespaces.get(namespace)
    if namespace_config is None:
        return Sensitivity.PERSONAL
    return Sensitivity(namespace_config.sensitivity)


def _conversation_payload(conversation) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "title": conversation.title,
        "active_project_namespace": conversation.active_project_namespace,
        "status": conversation.status.value,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "metadata": conversation.metadata,
    }


def _message_payload(message) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "conversation_id": message.conversation_id,
        "request_id": message.request_id,
        "role": message.role.value,
        "content": message.content,
        "sensitivity": message.sensitivity.value,
        "created_at": message.created_at,
        "metadata": message.metadata,
    }


def _request_payload(request) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "conversation_id": request.conversation_id,
        "user_message_id": request.user_message_id,
        "assistant_message_id": request.assistant_message_id,
        "status": request.status.value,
        "created_at": request.created_at,
        "started_at": request.started_at,
        "completed_at": request.completed_at,
        "error": (
            None
            if request.error_code is None
            else {"code": request.error_code, "message": request.error_message}
        ),
    }


def _memory_payload(memory) -> dict[str, Any]:
    return {
        "memory_id": memory.id,
        "namespace": memory.namespace,
        "memory_type": memory.memory_type.value,
        "content": memory.content,
        "summary": memory.summary,
        "content_hash": memory.content_hash,
        "sensitivity": memory.sensitivity.value,
        "status": memory.status.value,
        "indexing_status": memory.indexing_status.value,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "archived_at": memory.archived_at,
        "archive_reason": memory.archive_reason,
        "metadata": memory.metadata,
    }


def _runtime_status_payload(settings: Settings) -> dict[str, Any]:
    return {
        "status": "ready",
        "default_model_profile": "local_main",
        "model_profiles": {
            name: {
                "purpose": profile.purpose,
                "provider": profile.provider,
                "enabled": profile.enabled,
                "cloud": profile.cloud,
                "model": profile.model,
                "endpoint": profile.endpoint,
                "max_input_tokens": profile.max_input_tokens,
                "max_output_tokens": profile.max_output_tokens,
                "temperature": profile.temperature,
                "supports_streaming": profile.supports_streaming,
            }
            for name, profile in settings.model_profiles.items()
        },
        "runtime_budgets": {
            name: {
                "max_model_calls": budget.max_model_calls,
                "max_tool_calls": budget.max_tool_calls,
                "max_wall_time_seconds": budget.max_wall_time_seconds,
                "max_output_tokens": budget.max_output_tokens,
                "allow_cloud": budget.allow_cloud,
                "allow_tools": budget.allow_tools,
                "allow_autonomous_memory_write": budget.allow_autonomous_memory_write,
            }
            for name, budget in settings.runtime_budgets.items()
        },
    }


def _json_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value
