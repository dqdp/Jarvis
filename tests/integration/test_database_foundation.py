from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader
from assistant_core.models.fake_provider import FakeEmbeddingProvider
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import DatabaseSafetyError, assert_test_database_url, create_database_engine
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _reset_database(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("drop table if exists memory_embeddings cascade"))
            await connection.execute(text("drop table if exists memory_candidates cascade"))
            await connection.execute(text("drop table if exists memories cascade"))
            await connection.execute(text("drop table if exists model_invocations cascade"))
            await connection.execute(text("drop table if exists assistant_requests cascade"))
            await connection.execute(text("drop table if exists messages cascade"))
            await connection.execute(text("drop table if exists conversations cascade"))
            await connection.execute(text("drop table if exists events cascade"))
            await connection.execute(text("drop table if exists alembic_version cascade"))
    finally:
        await engine.dispose()


def _run_migrations_to(database_url: str, revision: str) -> None:
    assert_test_database_url(database_url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(PROJECT_ROOT / "src" / "assistant_core" / "storage" / "migrations"),
    )
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, revision)


def _run_test_migrations(database_url: str) -> None:
    assert_test_database_url(database_url)
    run_migrations(database_url)


async def _scalar(database_url: str, statement: str):
    engine = create_database_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.scalar(text(statement))
    finally:
        await engine.dispose()


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-database-foundation:{label}"))


async def _insert_request_with_detached_assistant_message(database_url: str) -> None:
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    insert into conversations (
                      conversation_id, user_id, title, active_project_namespace,
                      status, created_at, updated_at, metadata
                    )
                    values (
                      cast(:conversation_id as uuid), 'local_user', 'dirty',
                      'project.personal_assistant', 'active', now(), now(), '{}'
                    )
                    """,
                ),
                {"conversation_id": _id("conversation")},
            )
            await connection.execute(
                text(
                    """
                    insert into messages (
                      message_id, conversation_id, request_id, event_id,
                      client_message_id, role, content, content_hash,
                      sensitivity, created_at, metadata
                    )
                    values (
                      cast(:user_message_id as uuid), cast(:conversation_id as uuid),
                      null, null, 'dirty-user', 'user', 'hello', 'sha256:user',
                      'project', now(), '{}'
                    )
                    """,
                ),
                {
                    "conversation_id": _id("conversation"),
                    "user_message_id": _id("user-message"),
                },
            )
            await connection.execute(
                text(
                    """
                    insert into messages (
                      message_id, conversation_id, request_id, event_id,
                      client_message_id, role, content, content_hash,
                      sensitivity, created_at, metadata
                    )
                    values (
                      cast(:assistant_message_id as uuid), cast(:conversation_id as uuid),
                      null, null, null, 'assistant', 'late answer', 'sha256:assistant',
                      'project', now(), '{}'
                    )
                    """,
                ),
                {
                    "conversation_id": _id("conversation"),
                    "assistant_message_id": _id("assistant-message"),
                },
            )
            await connection.execute(
                text(
                    """
                    insert into assistant_requests (
                      request_id, conversation_id, user_message_id, assistant_message_id,
                      status, client_message_id, created_at, started_at, completed_at,
                      error_code, error_message, metadata
                    )
                    values (
                      cast(:request_id as uuid), cast(:conversation_id as uuid),
                      cast(:user_message_id as uuid), cast(:assistant_message_id as uuid),
                      'completed', 'dirty-user', now(), now(), now(), null, null, '{}'
                    )
                    """,
                ),
                {
                    "request_id": _id("request"),
                    "conversation_id": _id("conversation"),
                    "user_message_id": _id("user-message"),
                    "assistant_message_id": _id("assistant-message"),
                },
            )
    finally:
        await engine.dispose()


def test_database_connects() -> None:
    assert asyncio.run(_scalar(_database_url(), "select 1")) == 1


def test_migrations_apply_cleanly() -> None:
    database_url = _database_url()
    asyncio.run(_reset_database(database_url))

    _run_test_migrations(database_url)

    assert asyncio.run(_scalar(database_url, "select to_regclass('public.events')")) == "events"
    assert asyncio.run(
        _scalar(
            database_url,
            "select exists ("
            "select 1 from pg_constraint where conname = 'messages_request_fk'"
            ")",
        ),
    ) is True
    assert asyncio.run(
        _scalar(
            database_url,
            "select exists ("
            "select 1 from pg_class c "
            "join pg_index i on i.indexrelid = c.oid "
            "where c.relname = 'assistant_requests_user_message_idx' "
            "and i.indisunique"
            ")",
        ),
    ) is True


def test_migrations_are_idempotent() -> None:
    database_url = _database_url()
    asyncio.run(_reset_database(database_url))

    _run_test_migrations(database_url)
    _run_test_migrations(database_url)

    assert asyncio.run(_scalar(database_url, "select count(*) from alembic_version")) == 1


def test_migration_0006_rejects_assistant_message_without_matching_request_id() -> None:
    database_url = _database_url()
    asyncio.run(_reset_database(database_url))
    try:
        _run_migrations_to(database_url, "0005_memory_embeddings")
        asyncio.run(_insert_request_with_detached_assistant_message(database_url))

        with pytest.raises(RuntimeError, match="assistant_message_id is not a same-request"):
            _run_test_migrations(database_url)
    finally:
        asyncio.run(_reset_database(database_url))
        _run_test_migrations(database_url)


def test_reset_database_rejects_non_test_database_url() -> None:
    with pytest.raises(DatabaseSafetyError):
        asyncio.run(
            _reset_database(
                "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_prod",
            ),
        )


def test_conversation_store_health_requires_migrated_schema() -> None:
    database_url = _database_url()
    asyncio.run(_reset_database(database_url))
    engine = create_database_engine(database_url)
    try:
        store = PostgresConversationStore(engine)
        assert asyncio.run(store.health_check()) is False
    finally:
        asyncio.run(engine.dispose())


def test_conversation_store_health_requires_integrity_constraints() -> None:
    database_url = _database_url()
    asyncio.run(_reset_database(database_url))
    _run_test_migrations(database_url)
    try:
        async def scenario() -> None:
            engine = create_database_engine(database_url)
            store = PostgresConversationStore(engine)
            assert await store.health_check() is True
            async with engine.begin() as connection:
                await connection.execute(text("alter table messages drop constraint messages_request_fk"))
            assert await store.health_check() is False
            await engine.dispose()

        asyncio.run(scenario())
    finally:
        asyncio.run(_reset_database(database_url))
        _run_test_migrations(database_url)


def test_memory_store_health_requires_embedding_integrity_constraints() -> None:
    database_url = _database_url()
    asyncio.run(_reset_database(database_url))
    _run_test_migrations(database_url)
    try:
        async def scenario() -> None:
            engine = create_database_engine(database_url)
            settings = ConfigLoader("config").load("test")
            store = PostgresMemoryStore(
                engine=engine,
                settings=settings,
                policy=ConfigPolicyEngine(settings),
                embedding_port=FakeEmbeddingProvider(),
            )
            assert await store.health_check() is True
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "alter table memory_embeddings "
                        "drop constraint memory_embeddings_memory_id_fkey",
                    ),
                )
            assert await store.health_check() is False
            await engine.dispose()

        asyncio.run(scenario())
    finally:
        asyncio.run(_reset_database(database_url))
        _run_test_migrations(database_url)
