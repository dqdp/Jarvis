from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from assistant_core.config.settings import Settings
from assistant_core.domain.context import ContextAssemblyRequest
from assistant_core.domain.conversations import (
    CompleteAssistantResponseCommand,
    ConversationMessage,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.models import ChatModelRequest
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.context_assembler import ContextAssemblerPort
from assistant_core.ports.conversation_store import (
    ConversationStorePort,
    InvalidRequestStatusTransition,
)
from assistant_core.ports.event_log import EventFilter, EventLogPort
from assistant_core.ports.model_router import ModelRouterPort


@dataclass(frozen=True)
class RuntimeTurnCommand:
    request_id: str
    conversation_id: str
    user_message_id: str
    user_id: str
    user_input: str
    active_project_namespace: str | None
    current_message_sensitivity: Sensitivity = Sensitivity.PROJECT
    model_profile: str = "local_main"
    loop_strategy: str = "memory_augmented_answer"


@dataclass(frozen=True)
class RuntimeTurnResult:
    request_id: str
    response_text: str
    assistant_message: ConversationMessage | None
    model_calls: int
    degraded: bool


@dataclass(frozen=True)
class RuntimeStreamEvent:
    event_type: str
    data: dict


class RuntimePolicyDenied(Exception):
    """Raised when runtime policy blocks request execution before model calls."""


class AgentRuntime:
    def __init__(
        self,
        *,
        conversation_store: ConversationStorePort,
        context_assembler: ContextAssemblerPort,
        model_router: ModelRouterPort,
        event_log: EventLogPort,
        settings: Settings,
    ) -> None:
        self._conversation_store = conversation_store
        self._context_assembler = context_assembler
        self._model_router = model_router
        self._event_log = event_log
        self._settings = settings

    async def run_turn(self, command: RuntimeTurnCommand) -> RuntimeTurnResult:
        budget = self._settings.runtime_budgets[command.loop_strategy]
        model_calls = 0
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=command.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        current_event = await self._append_event(
            EventType.REQUEST_PROCESSING_STARTED,
            command,
            payload={},
        )
        if command.current_message_sensitivity == Sensitivity.SECRET:
            exc = RuntimePolicyDenied("secret input cannot be sent to model context")
            await self._append_policy_denial(command, causation_id=current_event.event_id)
            await self._fail_request(command, exc, sensitivity=Sensitivity.SECRET)
            raise exc
        current_event = await self._append_event(
            EventType.CONTEXT_ASSEMBLY_STARTED,
            command,
            payload={},
            causation_id=current_event.event_id,
        )

        try:
            context = await asyncio.wait_for(
                self._context_assembler.assemble(
                    ContextAssemblyRequest(
                        request_id=command.request_id,
                        conversation_id=command.conversation_id,
                        user_id=command.user_id,
                        current_user_message=command.user_input,
                        active_project_namespace=command.active_project_namespace,
                        loop_strategy=command.loop_strategy,
                        model_profile=command.model_profile,
                        current_message_sensitivity=command.current_message_sensitivity,
                        current_user_message_id=command.user_message_id,
                        causation_event_id=current_event.event_id,
                    ),
                ),
                timeout=budget.max_context_assembly_seconds,
            )
            context_event = await self._context_event(
                command.request_id,
                context.manifest.context_manifest_id,
            )
            model_created = await self._append_event(
                EventType.MODEL_REQUEST_CREATED,
                command,
                payload={"context_manifest_id": context.manifest.context_manifest_id},
                causation_id=context_event.event_id if context_event is not None else None,
                sensitivity=context.manifest.max_sensitivity,
            )
            if model_calls >= budget.max_model_calls:
                raise RuntimeError("max_model_calls exceeded")
            model_calls += 1
            response = await asyncio.wait_for(
                self._model_router.chat(
                    ChatModelRequest(
                        profile=command.model_profile,
                        messages=context.messages,
                        sensitivity=context.manifest.max_sensitivity,
                        request_id=command.request_id,
                        conversation_id=command.conversation_id,
                        context_manifest_id=context.manifest.context_manifest_id,
                    ),
                ),
                timeout=budget.max_model_call_seconds,
            )
        except Exception as exc:
            await self._conversation_store.update_assistant_request_status(
                UpdateAssistantRequestStatusCommand(
                    request_id=command.request_id,
                    status=RequestStatus.FAILED,
                    error_code=_error_code(exc),
                    error_message=_safe_error_message(exc),
                ),
            )
            await self._append_event(
                EventType.MODEL_REQUEST_FAILED,
                command,
                payload={"error_type": type(exc).__name__},
            )
            await self._append_event(
                EventType.REQUEST_PROCESSING_FAILED,
                command,
                payload={"error_type": type(exc).__name__},
            )
            raise

        await self._append_event(
            EventType.MODEL_RESPONSE_RECEIVED,
            command,
            payload={"context_manifest_id": context.manifest.context_manifest_id},
            causation_id=model_created.event_id,
            sensitivity=context.manifest.max_sensitivity,
        )
        model_received = await self._context_event(
            command.request_id,
            context.manifest.context_manifest_id,
            event_type=EventType.MODEL_RESPONSE_RECEIVED,
        )
        completion = await self._conversation_store.complete_assistant_response(
            CompleteAssistantResponseCommand(
                conversation_id=command.conversation_id,
                request_id=command.request_id,
                content=response.text,
                sensitivity=context.manifest.max_sensitivity,
            ),
        )
        assistant_message = completion.message
        assistant_event = await self._append_event(
            EventType.ASSISTANT_MESSAGE_CREATED,
            command,
            payload={
                "message_id": assistant_message.message_id,
                "content_hash": assistant_message.content_hash,
            },
            causation_id=model_received.event_id if model_received is not None else None,
            sensitivity=context.manifest.max_sensitivity,
        )
        await self._append_event(
            EventType.REQUEST_PROCESSING_COMPLETED,
            command,
            payload={"assistant_message_id": assistant_message.message_id},
            causation_id=assistant_event.event_id,
            sensitivity=context.manifest.max_sensitivity,
        )
        return RuntimeTurnResult(
            request_id=command.request_id,
            response_text=response.text,
            assistant_message=assistant_message,
            model_calls=model_calls,
            degraded=context.manifest.degraded,
        )

    async def stream_turn(self, command: RuntimeTurnCommand):
        budget = self._settings.runtime_budgets[command.loop_strategy]
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=command.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        started = await self._append_event(
            EventType.REQUEST_PROCESSING_STARTED,
            command,
            payload={},
        )
        yield _stream_event(started)
        if command.current_message_sensitivity == Sensitivity.SECRET:
            exc = RuntimePolicyDenied("secret input cannot be sent to model context")
            await self._append_policy_denial(command, causation_id=started.event_id)
            failed = await self._fail_request(command, exc, sensitivity=Sensitivity.SECRET)
            yield _stream_event(failed)
            return
        context_started = await self._append_event(
            EventType.CONTEXT_ASSEMBLY_STARTED,
            command,
            payload={},
            causation_id=started.event_id,
        )
        yield _stream_event(context_started)

        response_parts: list[str] = []
        model_created: EventEnvelope | None = None
        try:
            context = await asyncio.wait_for(
                self._context_assembler.assemble(
                    ContextAssemblyRequest(
                        request_id=command.request_id,
                        conversation_id=command.conversation_id,
                        user_id=command.user_id,
                        current_user_message=command.user_input,
                        active_project_namespace=command.active_project_namespace,
                        loop_strategy=command.loop_strategy,
                        model_profile=command.model_profile,
                        current_message_sensitivity=command.current_message_sensitivity,
                        current_user_message_id=command.user_message_id,
                        causation_event_id=context_started.event_id,
                    ),
                ),
                timeout=budget.max_context_assembly_seconds,
            )
            memory_event = await self._latest_event(
                command.request_id,
                EventType.MEMORY_RETRIEVED,
                causation_id=context_started.event_id,
            )
            if memory_event is not None:
                yield _stream_event(memory_event)
            context_event = await self._context_event(
                command.request_id,
                context.manifest.context_manifest_id,
            )
            if context_event is not None:
                yield _stream_event(context_event)
            else:
                yield RuntimeStreamEvent(
                    "context.assembled",
                    {
                        "request_id": command.request_id,
                        "context_manifest_id": context.manifest.context_manifest_id,
                        "degraded": context.manifest.degraded,
                    },
                )
            model_created = await self._append_event(
                EventType.MODEL_REQUEST_CREATED,
                command,
                payload={"context_manifest_id": context.manifest.context_manifest_id},
                causation_id=context_event.event_id if context_event is not None else None,
                sensitivity=context.manifest.max_sensitivity,
            )
            yield _stream_event(model_created)
            async with asyncio.timeout(budget.max_model_call_seconds):
                async for event in self._model_router.stream_chat(
                    ChatModelRequest(
                        profile=command.model_profile,
                        messages=context.messages,
                        sensitivity=context.manifest.max_sensitivity,
                        request_id=command.request_id,
                        conversation_id=command.conversation_id,
                        context_manifest_id=context.manifest.context_manifest_id,
                    ),
                ):
                    if event.event_type == "token" and event.delta is not None:
                        if await self._request_is_terminal(command.request_id):
                            return
                        response_parts.append(event.delta)
                        yield RuntimeStreamEvent("token", {"delta": event.delta})
        except Exception as exc:
            await self._conversation_store.update_assistant_request_status(
                UpdateAssistantRequestStatusCommand(
                    request_id=command.request_id,
                    status=RequestStatus.FAILED,
                    error_code=_error_code(exc),
                    error_message=_safe_error_message(exc),
                ),
            )
            failed = await self._append_event(
                (
                    EventType.MODEL_REQUEST_FAILED
                    if model_created is not None
                    else EventType.CONTEXT_ASSEMBLY_FAILED
                ),
                command,
                payload={
                    "error_type": type(exc).__name__,
                    "error_code": _error_code(exc),
                },
                causation_id=(
                    model_created.event_id if model_created is not None else context_started.event_id
                ),
            )
            yield _stream_event(failed)
            request_failed = await self._append_event(
                EventType.REQUEST_PROCESSING_FAILED,
                command,
                payload={
                    "error_type": type(exc).__name__,
                    "error_code": _error_code(exc),
                    "error": {
                        "code": _error_code(exc),
                        "message": _safe_error_message(exc),
                        "request_id": command.request_id,
                        "details": {},
                    },
                },
                causation_id=failed.event_id,
            )
            yield _stream_event(request_failed)
            return

        if await self._request_is_terminal(command.request_id):
            return

        response_text = "".join(response_parts)
        model_received = await self._append_event(
            EventType.MODEL_RESPONSE_RECEIVED,
            command,
            payload={"context_manifest_id": context.manifest.context_manifest_id},
            causation_id=model_created.event_id,
            sensitivity=context.manifest.max_sensitivity,
        )
        yield _stream_event(model_received)
        if await self._request_is_terminal(command.request_id):
            return
        try:
            completion = await self._conversation_store.complete_assistant_response(
                CompleteAssistantResponseCommand(
                    conversation_id=command.conversation_id,
                    request_id=command.request_id,
                    content=response_text,
                    sensitivity=context.manifest.max_sensitivity,
                ),
            )
        except InvalidRequestStatusTransition:
            if await self._request_is_terminal(command.request_id):
                return
            raise
        assistant_message = completion.message
        assistant_event = await self._append_event(
            EventType.ASSISTANT_MESSAGE_CREATED,
            command,
            payload={
                "message_id": assistant_message.message_id,
                "content_hash": assistant_message.content_hash,
            },
            causation_id=model_received.event_id,
            sensitivity=context.manifest.max_sensitivity,
        )
        yield _stream_event(assistant_event)
        completed = await self._append_event(
            EventType.REQUEST_PROCESSING_COMPLETED,
            command,
            payload={"assistant_message_id": assistant_message.message_id},
            causation_id=assistant_event.event_id,
            sensitivity=context.manifest.max_sensitivity,
        )
        yield _stream_event(completed)

    async def _context_event(
        self,
        request_id: str,
        context_manifest_id: str,
        *,
        event_type: EventType = EventType.CONTEXT_ASSEMBLED,
    ) -> EventEnvelope | None:
        events = await self._event_log.query(EventFilter(request_id=request_id))
        for event in reversed(events):
            if (
                event.event_type == event_type
                and event.payload.get("context_manifest_id") == context_manifest_id
            ):
                return event
        return None

    async def _latest_event(
        self,
        request_id: str,
        event_type: EventType,
        *,
        causation_id: str | None = None,
    ) -> EventEnvelope | None:
        events = await self._event_log.query(EventFilter(request_id=request_id))
        for event in reversed(events):
            if event.event_type != event_type:
                continue
            if causation_id is not None and event.causation_id != causation_id:
                continue
            return event
        return None

    async def _request_is_terminal(self, request_id: str) -> bool:
        request = await self._conversation_store.get_assistant_request(request_id)
        return request is not None and request.status in {
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
        }

    async def _fail_request(
        self,
        command: RuntimeTurnCommand,
        exc: Exception,
        *,
        sensitivity: Sensitivity = Sensitivity.PROJECT,
    ) -> EventEnvelope:
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=command.request_id,
                status=RequestStatus.FAILED,
                error_code=_error_code(exc),
                error_message=_safe_error_message(exc),
            ),
        )
        return await self._append_event(
            EventType.REQUEST_PROCESSING_FAILED,
            command,
            payload={"error_type": type(exc).__name__, "error_code": _error_code(exc)},
            sensitivity=sensitivity,
        )

    async def _append_policy_denial(
        self,
        command: RuntimeTurnCommand,
        *,
        causation_id: str | None,
    ) -> EventEnvelope:
        return await self._append_event(
            EventType.POLICY_DECISION_RECORDED,
            command,
            payload={
                "source_ref": "current_user_message",
                "allowed": False,
                "code": "sensitivity_denied",
                "reason": "secret input cannot be sent to model context",
            },
            causation_id=causation_id,
            sensitivity=Sensitivity.SECRET,
        )

    async def _append_event(
        self,
        event_type: EventType,
        command: RuntimeTurnCommand,
        *,
        payload: dict,
        causation_id: str | None = None,
        sensitivity: Sensitivity = Sensitivity.PROJECT,
    ) -> EventEnvelope:
        now = datetime.now(UTC)
        return await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=event_type,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=command.conversation_id,
                request_id=command.request_id,
                correlation_id=command.request_id,
                causation_id=causation_id,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="agent_runtime",
                source_node=None,
                sensitivity=sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload=payload,
                metadata={},
            ),
        )


def _stream_event(event: EventEnvelope) -> RuntimeStreamEvent:
    return RuntimeStreamEvent(
        event_type=event.event_type.value,
        data={"request_id": event.request_id, "event_id": event.event_id, **event.payload},
    )


def _error_code(exc: Exception) -> str:
    if isinstance(exc, RuntimePolicyDenied):
        return "policy_denied"
    if isinstance(exc, TimeoutError):
        return "runtime_timeout"
    return type(exc).__name__


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, RuntimePolicyDenied):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "request timed out"
    return "request failed"
