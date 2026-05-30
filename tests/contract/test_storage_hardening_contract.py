from __future__ import annotations

from datetime import UTC, datetime
import asyncio
import os
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-storage-hardening:{label}"))


async def _truncate_hardening_tables(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table content_embeddings, content_chunks, content_sources, "
                    "memory_embeddings, memory_candidates, memories, "
                    "model_invocations, assistant_requests, messages, conversations, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


@pytest.fixture
def engine():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_hardening_tables(database_url))
    test_engine = create_database_engine(database_url)
    try:
        yield test_engine
    finally:
        asyncio.run(test_engine.dispose())


def _event(label: str) -> EventEnvelope:
    now = datetime.now(UTC)
    request_id = _id(f"{label}:request")
    return EventEnvelope(
        event_id=_id(label),
        event_seq=0,
        event_type=EventType.USER_MESSAGE_CREATED,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=_id(f"{label}:conversation"),
        request_id=request_id,
        correlation_id=request_id,
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="storage_hardening_contract",
        source_node=None,
        sensitivity=Sensitivity.PROJECT,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={"content_hash": "sha256:test"},
        metadata={},
    )


def test_postgres_events_are_append_only(engine) -> None:
    async def scenario() -> None:
        stored = await PostgresEventLog(engine).append(_event("append-only"))
        async with engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text("update events set source_component = 'tampered' where event_id = :event_id"),
                    {"event_id": stored.event_id},
                )
        async with engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text("delete from events where event_id = :event_id"),
                    {"event_id": stored.event_id},
                )
        async with engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(text("truncate table events cascade"))

    asyncio.run(scenario())


def test_postgres_schema_exposes_required_check_constraints(engine) -> None:
    async def scenario() -> set[str]:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "select conname from pg_constraint "
                        "where conrelid in ("
                        "'events'::regclass, 'conversations'::regclass, "
                        "'messages'::regclass, 'assistant_requests'::regclass, "
                        "'model_invocations'::regclass, 'memories'::regclass, "
                        "'memory_candidates'::regclass, 'content_sources'::regclass, "
                        "'content_chunks'::regclass, 'content_embeddings'::regclass"
                        ")",
                    ),
                )
            ).all()
        return {row[0] for row in rows}

    constraints = asyncio.run(scenario())

    assert {
        "events_event_type_check",
        "events_actor_type_check",
        "events_sensitivity_check",
        "events_visibility_check",
        "events_event_version_check",
        "conversations_status_check",
        "messages_role_check",
        "messages_sensitivity_check",
        "assistant_requests_status_check",
        "model_invocations_status_check",
        "model_invocations_sensitivity_no_secret_check",
        "memories_memory_type_check",
        "memories_status_check",
        "memories_indexing_status_check",
        "memories_sensitivity_no_secret_check",
        "memory_candidates_status_check",
        "memory_candidates_sensitivity_no_secret_check",
        "content_sources_status_check",
        "content_sources_sensitivity_no_secret_check",
        "content_chunks_status_check",
        "content_chunks_sensitivity_no_secret_check",
        "content_embeddings_status_check",
    }.issubset(constraints)


def test_postgres_schema_rejects_secret_memory_records(engine) -> None:
    async def scenario() -> None:
        async with engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text(
                        "insert into memories ("
                        "memory_id, namespace, memory_type, content, content_hash, sensitivity, "
                        "confidence, importance, status, indexing_status, source_event_ids, "
                        "supersedes_memory_ids, revision, created_at, updated_at, metadata"
                        ") values ("
                        "cast(:memory_id as uuid), 'project.personal_assistant', 'fact', "
                        "'raw secret', 'sha256:test', 'secret', 0.5, 0.5, 'active', "
                        "'indexed', '{}'::uuid[], '{}'::uuid[], 1, now(), now(), '{}'::jsonb"
                        ")",
                    ),
                    {"memory_id": _id("secret-memory")},
                )

    asyncio.run(scenario())
