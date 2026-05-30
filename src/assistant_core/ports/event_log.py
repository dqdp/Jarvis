from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import re
from typing import Any, Protocol

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


def sanitize_event_envelope(event: EventEnvelope) -> EventEnvelope:
    return replace(
        event,
        actor_id=redact_sensitive_text(event.actor_id),
        source_node=redact_sensitive_text(event.source_node),
        idempotency_key=redact_sensitive_text(event.idempotency_key),
        payload=_sanitize_mapping(event.payload),
        metadata=_sanitize_mapping(event.metadata),
    )


def _sanitize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_value(key, value) for key, value in mapping.items()}


def _sanitize_value(key: str, value: Any) -> Any:
    if _sensitive_key(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(key, item) for item in value]
    if isinstance(value, str) and sensitive_text(value):
        return "<redacted>"
    return value


def redact_sensitive_text(value: str | None) -> str | None:
    if value is None:
        return None
    return "<redacted>" if sensitive_text(value) else value


def sensitive_text(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in _SENSITIVE_VALUE_MARKERS)


def _sensitive_key(key: str) -> bool:
    normalized = key.lower()
    if normalized in _NON_SECRET_KEYS:
        return False
    if normalized in _SENSITIVE_EXACT_KEYS:
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", normalized) if part]
    if any(part in {"authorization", "credential", "password", "prompt", "secret"} for part in parts):
        return True
    if "token" in parts:
        return True
    if "api" in parts and "key" in parts:
        return True
    if "private" in parts and "key" in parts:
        return True
    return False


_NON_SECRET_KEYS = {
    "full_prompt_stored",
    "input_tokens",
    "max_input_tokens",
    "max_output_tokens",
    "output_tokens",
    "token_estimate",
    "total_tokens",
}

_SENSITIVE_EXACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "prompt",
    "raw_prompt",
    "secret",
    "token",
}

_SENSITIVE_VALUE_MARKERS = (
    "-----begin",
    "akia",
    "authorization:",
    "authorization=",
    "bearer ",
    "ghp_",
    "github_pat_",
    "id_ed25519",
    "id_rsa",
    "password=",
    "private key",
    "sk-",
    "sk_",
    "token=",
)
