from __future__ import annotations

import asyncio
from typing import Any

from assistant_core.runtime.request_streaming import (
    STREAM_REPLAY_EVENT_TYPES,
    TERMINAL_EVENT_TYPES,
    RequestStreamEvent,
    public_stream_data,
)


class RequestStreamBuffer:
    def __init__(self) -> None:
        self._events: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._conditions: dict[str, asyncio.Condition] = {}
        self._active_streams: dict[str, int] = {}

    async def publish(self, request_id: str, event_type: str, data: dict[str, Any]) -> bool:
        if event_type != "token" and event_type not in STREAM_REPLAY_EVENT_TYPES:
            return False
        payload = dict(data)
        payload.setdefault("request_id", request_id)
        self._events.setdefault(request_id, []).append(
            (event_type, public_stream_data(event_type, payload)),
        )
        condition = self._condition(request_id)
        async with condition:
            condition.notify_all()
        return True

    def events_from(self, request_id: str, index: int) -> list[RequestStreamEvent]:
        return [
            RequestStreamEvent(event_type, data)
            for event_type, data in self._events.get(request_id, [])[index:]
        ]

    def raw_events_until(self, request_id: str, index: int) -> list[tuple[str, dict[str, Any]]]:
        return self._events.get(request_id, [])[:index]

    def register_stream(self, request_id: str) -> None:
        self._active_streams[request_id] = self._active_streams.get(request_id, 0) + 1

    def unregister_stream(self, request_id: str) -> None:
        active_count = self._active_streams.get(request_id, 0) - 1
        if active_count > 0:
            self._active_streams[request_id] = active_count
        else:
            self._active_streams.pop(request_id, None)

    def has_active_stream(self, request_id: str) -> bool:
        return self._active_streams.get(request_id, 0) > 0

    async def wait(self, request_id: str, timeout_seconds: float) -> bool:
        condition = self._condition(request_id)
        try:
            async with condition:
                await asyncio.wait_for(condition.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True

    def terminal_event_seen(self, request_id: str, index: int) -> bool:
        return any(
            event_type in TERMINAL_EVENT_TYPES
            for event_type, _ in self.raw_events_until(request_id, index)
        )

    def discard(self, request_id: str) -> None:
        self._events.pop(request_id, None)
        self._conditions.pop(request_id, None)
        self._active_streams.pop(request_id, None)

    def clear(self) -> None:
        self._events.clear()
        self._conditions.clear()
        self._active_streams.clear()

    def _condition(self, request_id: str) -> asyncio.Condition:
        condition = self._conditions.get(request_id)
        if condition is None:
            condition = asyncio.Condition()
            self._conditions[request_id] = condition
        return condition
