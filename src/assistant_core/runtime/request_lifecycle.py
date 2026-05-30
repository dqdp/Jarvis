from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from assistant_core.domain.conversations import UpdateAssistantRequestStatusCommand
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.conversation_store import InvalidRequestStatusTransition
from assistant_core.runtime.request_streaming import TERMINAL_REQUEST_STATUSES


class RequestLifecycleService:
    def __init__(self, *, conversation_store, event_log, stream_buffer) -> None:
        self._conversation_store = conversation_store
        self._event_log = event_log
        self._stream_buffer = stream_buffer

    async def mark_cancelled(self, request_record):
        current = await self._conversation_store.get_assistant_request(request_record.request_id)
        if current is None:
            raise KeyError("request not found")
        if current.status in TERMINAL_REQUEST_STATUSES:
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
        event = await self._event_log.append(request_cancelled_event(cancelled))
        await self._stream_buffer.publish(
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

    async def mark_failed(self, request_record, *, code: str, message: str):
        current = await self._conversation_store.get_assistant_request(request_record.request_id)
        if current is None:
            raise KeyError("request not found")
        if current.status in TERMINAL_REQUEST_STATUSES:
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
        event = await self._event_log.append(request_failed_event(failed, code=code))
        await self._stream_buffer.publish(
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


def orphaned_request_error_code(status: RequestStatus) -> str:
    if status == RequestStatus.RUNNING:
        return "orphaned_running_request"
    if status == RequestStatus.WAITING_APPROVAL:
        return "orphaned_waiting_approval_request"
    return "orphaned_accepted_request"


def request_execution_age_seconds(request_record) -> float:
    anchor = request_record.started_at or request_record.created_at
    return (datetime.now(UTC) - anchor).total_seconds()


def request_cancelled_event(request_record) -> EventEnvelope:
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
        source_component="request_execution",
        source_node=None,
        sensitivity=Sensitivity.PROJECT,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={"error_code": "cancelled"},
        metadata={},
    )


def request_failed_event(request_record, *, code: str) -> EventEnvelope:
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
        source_component="request_execution",
        source_node=None,
        sensitivity=Sensitivity.PROJECT,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={"error_code": code},
        metadata={},
    )
