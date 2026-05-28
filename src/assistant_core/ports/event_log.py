from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assistant_core.domain.events import EventEnvelope


class EventEnvelopeValidationError(ValueError):
    """Raised when an event envelope violates the Phase 1 contract."""


@dataclass(frozen=True)
class EventFilter:
    request_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None


class EventLogPort(Protocol):
    async def append(self, event: EventEnvelope) -> EventEnvelope: ...

    async def query(self, event_filter: EventFilter) -> list[EventEnvelope]: ...


def validate_event_envelope(event: EventEnvelope) -> None:
    if not event.event_id:
        raise EventEnvelopeValidationError("event_id is required")
    if event.event_version < 1:
        raise EventEnvelopeValidationError("event_version must be positive")
    if event.event_seq < 0:
        raise EventEnvelopeValidationError("event_seq must not be negative")
    if not event.source_component:
        raise EventEnvelopeValidationError("source_component is required")
    if event.payload is None:
        raise EventEnvelopeValidationError("payload is required")
    if event.metadata is None:
        raise EventEnvelopeValidationError("metadata is required")
