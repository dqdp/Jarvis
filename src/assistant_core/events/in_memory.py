from __future__ import annotations

from dataclasses import replace

from assistant_core.domain.events import EventEnvelope
from assistant_core.ports.event_log import (
    EventFilter,
    sanitize_event_envelope,
    validate_event_envelope,
)


class InMemoryEventLog:
    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self._next_event_seq = 1

    async def append(self, event: EventEnvelope) -> EventEnvelope:
        sanitized = sanitize_event_envelope(event)
        validate_event_envelope(sanitized)
        stored = replace(sanitized, event_seq=self._next_event_seq)
        self._next_event_seq += 1
        self._events.append(stored)
        return stored

    async def query(self, event_filter: EventFilter) -> list[EventEnvelope]:
        return [
            event
            for event in sorted(self._events, key=lambda item: item.event_seq)
            if _matches(event, event_filter)
        ]


def _matches(event: EventEnvelope, event_filter: EventFilter) -> bool:
    if event_filter.request_id is not None and event.request_id != event_filter.request_id:
        return False
    if (
        event_filter.conversation_id is not None
        and event.conversation_id != event_filter.conversation_id
    ):
        return False
    if (
        event_filter.correlation_id is not None
        and event.correlation_id != event_filter.correlation_id
    ):
        return False
    return True
