from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import asyncio
import os
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.domain.events import (
    ActorType,
    EventEnvelope,
    EventType,
    EventVisibility,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventEnvelopeValidationError, EventFilter
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_events(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(text("truncate table events restart identity cascade"))
    finally:
        await engine.dispose()


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-event-log-contract:{label}"))


@pytest.fixture(params=["memory", "postgres"])
def event_log(request):
    if request.param == "memory":
        yield InMemoryEventLog()
        return

    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_events(database_url))
    engine = create_database_engine(database_url)
    try:
        yield PostgresEventLog(engine)
    finally:
        asyncio.run(engine.dispose())


def _event(
    event_id: str,
    event_type: EventType = EventType.USER_MESSAGE_CREATED,
    request_id: str = _id("req-1"),
    causation_id: str | None = None,
    idempotency_key: str | None = None,
) -> EventEnvelope:
    now = datetime.now(UTC)
    return EventEnvelope(
        event_id=event_id,
        event_seq=0,
        event_type=event_type,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=_id("conv-1"),
        request_id=request_id,
        correlation_id=request_id,
        causation_id=causation_id,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="test",
        source_node=None,
        sensitivity=Sensitivity.PERSONAL,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=idempotency_key,
        payload={"content_hash": "sha256:test"},
        metadata={},
    )


def test_event_log_contract_append_assigns_sequence(event_log) -> None:
    stored = asyncio.run(event_log.append(_event(_id("evt-1"))))

    assert stored.event_seq == 1


def test_event_log_contract_query_by_request_id_ordered(event_log) -> None:
    async def scenario() -> tuple[list[EventEnvelope], list[EventEnvelope]]:
        first = await event_log.append(_event(_id("evt-1"), request_id=_id("req-1")))
        await event_log.append(_event(_id("evt-2"), request_id=_id("req-2")))
        second = await event_log.append(
            _event(
                _id("evt-3"),
                event_type=EventType.REQUEST_PROCESSING_STARTED,
                request_id=_id("req-1"),
            ),
        )
        results = await event_log.query(EventFilter(request_id=_id("req-1")))
        return results, [first, second]

    results, expected = asyncio.run(scenario())

    assert results == expected


def test_event_log_contract_causation_chain(event_log) -> None:
    async def scenario() -> tuple[EventEnvelope, EventEnvelope]:
        cause = await event_log.append(_event(_id("evt-1")))
        effect = await event_log.append(
            _event(
                _id("evt-2"),
                event_type=EventType.CONTEXT_ASSEMBLY_STARTED,
                causation_id=cause.event_id,
            ),
        )
        return cause, effect

    cause, effect = asyncio.run(scenario())

    assert effect.causation_id == cause.event_id


def test_event_log_contract_preserves_idempotency_key(event_log) -> None:
    stored = asyncio.run(
        event_log.append(_event(_id("evt-1"), idempotency_key="client-message:abc")),
    )

    assert stored.idempotency_key == "client-message:abc"


def test_event_log_contract_redacts_secret_like_idempotency_key(event_log) -> None:
    stored = asyncio.run(
        event_log.append(
            _event(
                _id("evt-secret-idempotency"),
                idempotency_key="github_pat_1234567890abcdef",
            ),
        ),
    )

    assert stored.idempotency_key == "<redacted>"


def test_event_log_contract_sanitizes_sensitive_payload_and_metadata(event_log) -> None:
    raw = replace(
        _event(_id("evt-sensitive")),
        payload={
            "content_hash": "sha256:test",
            "api_key": "sk-test-secret",
            "nested": {"password": "swordfish"},
            "items": [{"token": "raw-token"}],
        },
        metadata={"prompt": "raw system prompt", "safe": "kept"},
    )

    stored = asyncio.run(event_log.append(raw))

    assert stored.payload["api_key"] == "<redacted>"
    assert stored.payload["nested"]["password"] == "<redacted>"
    assert stored.payload["items"][0]["token"] == "<redacted>"
    assert stored.metadata["prompt"] == "<redacted>"
    assert stored.metadata["safe"] == "kept"


def test_event_log_contract_preserves_non_secret_audit_fields(event_log) -> None:
    raw = replace(
        _event(_id("evt-audit-fields"), event_type=EventType.CONTEXT_ASSEMBLED),
        payload={
            "context_manifest_id": "manifest-1",
            "full_prompt_stored": False,
            "token_estimate": 42,
            "max_output_tokens": 1024,
            "raw_prompt": "secret prompt text",
        },
        metadata={},
    )

    stored = asyncio.run(event_log.append(raw))

    assert stored.payload["full_prompt_stored"] is False
    assert stored.payload["token_estimate"] == 42
    assert stored.payload["max_output_tokens"] == 1024
    assert stored.payload["raw_prompt"] == "<redacted>"


def test_event_log_contract_redacts_secret_like_values_under_neutral_keys(event_log) -> None:
    raw = replace(
        _event(_id("evt-neutral-secret-values")),
        payload={
            "output": "github_pat_1234567890abcdef",
            "note": "contains AKIAIOSFODNN7EXAMPLE credential",
            "nested": {"line": "private key begins here"},
        },
        metadata={"trace": "ghp_1234567890abcdefghijklmnop"},
    )

    stored = asyncio.run(event_log.append(raw))

    assert stored.payload["output"] == "<redacted>"
    assert stored.payload["note"] == "<redacted>"
    assert stored.payload["nested"]["line"] == "<redacted>"
    assert stored.metadata["trace"] == "<redacted>"


def test_event_envelope_validation(event_log) -> None:
    invalid = replace(_event(_id("evt-1")), event_version=0)

    with pytest.raises(EventEnvelopeValidationError):
        asyncio.run(event_log.append(invalid))
