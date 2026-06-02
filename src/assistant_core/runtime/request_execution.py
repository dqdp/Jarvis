from __future__ import annotations

import asyncio

from assistant_core.config.settings import Settings
from assistant_core.domain.events import EventType
from assistant_core.domain.requests import RequestStatus
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.request_command import RuntimeTurnCommandBuilder
from assistant_core.runtime.request_lifecycle import (
    RequestLifecycleService,
    orphaned_request_error_code,
    request_execution_age_seconds,
)
from assistant_core.runtime.request_stream_buffer import RequestStreamBuffer
from assistant_core.runtime.request_streaming import (
    TERMINAL_EVENT_TYPES,
    TERMINAL_REQUEST_STATUSES,
    RequestStreamEvent,
    durable_replay_stream,
    event_stream_event,
    has_terminal_event,
    terminal_stream_event,
    terminal_stream_event_from_log,
)


PRE_START_STREAM_EVENT_TYPES = {
    EventType.LOOP_SELECTION_STARTED.value,
    EventType.LOOP_SELECTION_COMPLETED.value,
    EventType.LOOP_SELECTION_FAILED.value,
}


class RequestExecutionManager:
    def __init__(
        self,
        *,
        runtime,
        conversation_store,
        event_log,
        settings: Settings,
        approval_store=None,
    ) -> None:
        self._runtime = runtime
        self._conversation_store = conversation_store
        self._event_log = event_log
        self._settings = settings
        self._approval_store = approval_store
        self._tasks: dict[str, asyncio.Task] = {}
        self._stream_buffer = RequestStreamBuffer()
        self._command_builder = RuntimeTurnCommandBuilder(
            conversation_store=conversation_store,
            settings=settings,
        )
        self._lifecycle = RequestLifecycleService(
            conversation_store=conversation_store,
            event_log=event_log,
            stream_buffer=self._stream_buffer,
        )
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
        if request_record.status in TERMINAL_REQUEST_STATUSES:
            return request_record
        if self._approval_store is not None:
            await self._approval_store.cancel_pending_for_request(
                request_record.request_id,
                actor_id=self._settings.app.default_user_id,
                reason="request cancelled",
            )
        task = self._tasks.get(request_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=0.1)
        refreshed = await self._conversation_store.get_assistant_request(request_id)
        if refreshed is not None and refreshed.status in TERMINAL_REQUEST_STATUSES:
            return refreshed
        return await self._lifecycle.mark_cancelled(request_record)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._stream_buffer.clear()

    async def stream(self, request_id: str):
        self._stream_buffer.register_stream(request_id)
        index = 0
        heartbeat_seconds = max(0.001, float(self._settings.api.sse_heartbeat_seconds))
        try:
            while True:
                for event in self._stream_buffer.events_from(request_id, index):
                    index += 1
                    yield event
                    if event.event_type in TERMINAL_EVENT_TYPES:
                        return

                request_record = await self._conversation_store.get_assistant_request(request_id)
                if request_record is None:
                    raise KeyError("request not found")
                if request_record.status in TERMINAL_REQUEST_STATUSES:
                    for event in self._stream_buffer.events_from(request_id, index):
                        index += 1
                        yield event
                        if event.event_type in TERMINAL_EVENT_TYPES:
                            return
                    if index == 0:
                        async for item in durable_replay_stream(
                            self._event_log,
                            self._conversation_store,
                            request_record,
                        ):
                            yield item
                    elif not has_terminal_event(self._stream_buffer.raw_events_until(request_id, index)):
                        yield await terminal_stream_event_from_log(self._event_log, request_record)
                    return

                task = self._tasks.get(request_id)
                if (
                    task is None or task.done()
                ) and request_record.status in {
                    RequestStatus.ACCEPTED,
                    RequestStatus.RUNNING,
                    RequestStatus.WAITING_APPROVAL,
                } and request_execution_age_seconds(request_record) >= float(
                    self._settings.api.request_timeout_seconds,
                ):
                    error_code = orphaned_request_error_code(request_record.status)
                    if (
                        request_record.status == RequestStatus.WAITING_APPROVAL
                        and self._approval_store is not None
                    ):
                        await self._approval_store.cancel_pending_for_request(
                            request_record.request_id,
                            actor_id=self._settings.app.default_user_id,
                            reason="request execution task is not active",
                        )
                    failed = await self._lifecycle.mark_failed(
                        request_record,
                        code=error_code,
                        message="request execution task is not active",
                    )
                    yield terminal_stream_event(failed)
                    return

                wait_seconds = heartbeat_seconds
                task = self._tasks.get(request_id)
                if task is None or task.done():
                    wait_seconds = min(heartbeat_seconds, 0.05)
                if not await self._stream_buffer.wait(request_id, wait_seconds):
                    yield RequestStreamEvent("heartbeat", {"request_id": request_id})
        finally:
            self._stream_buffer.unregister_stream(request_id)
            await self._cleanup_terminal_state(request_id)

    async def _cleanup_terminal_state(self, request_id: str) -> None:
        if self._stream_buffer.has_active_stream(request_id):
            return
        request_record = await self._conversation_store.get_assistant_request(request_id)
        if request_record is None or request_record.status not in TERMINAL_REQUEST_STATUSES:
            return
        task = self._tasks.get(request_id)
        if task is not None and not task.done() and task is not asyncio.current_task():
            return
        self._tasks.pop(request_id, None)
        self._stream_buffer.discard(request_id)

    async def _execute_request(self, request_id: str) -> None:
        request_record = await self._conversation_store.get_assistant_request(request_id)
        if request_record is None:
            return
        try:
            command = await self._command_builder.build(request_record)
            await self._execute(command)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._lifecycle.mark_failed(
                request_record,
                code="background_task_failed",
                message="request failed in background execution",
            )

    async def _execute(self, command) -> None:
        seeded_event_log = False
        try:
            async for event in self._runtime.stream_turn(command):
                await self._publish(command.request_id, event.event_type, event.data)
                if (
                    not seeded_event_log
                    and event.event_type == EventType.REQUEST_PROCESSING_STARTED.value
                ):
                    await self._seed_stream_buffer_from_event_log(
                        command.request_id,
                        exclude_event_types={EventType.REQUEST_PROCESSING_STARTED.value},
                    )
                    seeded_event_log = True
        except asyncio.CancelledError:
            request_record = await self._conversation_store.get_assistant_request(command.request_id)
            if request_record is not None and request_record.status not in TERMINAL_REQUEST_STATUSES:
                await self._lifecycle.mark_cancelled(request_record)
            return
        except Exception:
            request_record = await self._conversation_store.get_assistant_request(command.request_id)
            if request_record is not None:
                await self._lifecycle.mark_failed(
                    request_record,
                    code="background_task_failed",
                    message="request failed in background execution",
                )
        finally:
            await self._cleanup_terminal_state(command.request_id)

    async def _publish(self, request_id: str, event_type: str, data: dict) -> None:
        await self._stream_buffer.publish(request_id, event_type, data)
        if event_type in TERMINAL_EVENT_TYPES:
            await self._cleanup_terminal_state(request_id)

    async def _seed_stream_buffer_from_event_log(
        self,
        request_id: str,
        *,
        exclude_event_types: set[str] | None = None,
    ) -> None:
        excluded = exclude_event_types or set()
        for event in await self._event_log.query(EventFilter(request_id=request_id)):
            if event.event_type.value not in PRE_START_STREAM_EVENT_TYPES:
                continue
            if event.event_type.value in excluded:
                continue
            stream_event = event_stream_event(event)
            await self._stream_buffer.publish(
                request_id,
                stream_event.event_type,
                stream_event.data,
            )

    def _start_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
