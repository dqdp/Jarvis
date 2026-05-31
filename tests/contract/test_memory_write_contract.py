from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader, Settings
from assistant_core.domain.events import EventType
from assistant_core.domain.memory import (
    ArchiveMemoryCommand,
    CreateMemoryCommand,
    MemoryStatus,
    MemoryType,
    SupersedeMemoryCommand,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.memory import (
    InvalidMemoryType,
    MemoryPolicyDenied,
    MemoryTypeNotAllowed,
    UnknownMemoryNamespace,
)
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations


pytestmark = [pytest.mark.contract, pytest.mark.db]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _settings() -> Settings:
    return ConfigLoader(Path("config")).load("test")


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-memory-write-contract:{label}"))


async def _truncate_memory(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text("truncate table memory_candidates, memories, events restart identity cascade"),
            )
    finally:
        await engine.dispose()


@pytest.fixture
def store():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_memory(database_url))
    engine = create_database_engine(database_url)
    settings = _settings()
    try:
        yield PostgresMemoryStore(
            engine=engine,
            settings=settings,
            policy=ConfigPolicyEngine(settings),
        )
    finally:
        asyncio.run(engine.dispose())


def _create_command(
    *,
    memory_id: str = "memory-1",
    namespace: str = "project.personal_assistant",
    memory_type: MemoryType | str = MemoryType.FACT,
    content: str = "Phase 1 uses explicit memory writes.",
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> CreateMemoryCommand:
    return CreateMemoryCommand(
        memory_id=_id(memory_id),
        namespace=namespace,
        memory_type=memory_type,
        content=content,
        summary=None,
        sensitivity=sensitivity,
        confidence=0.9,
        importance=0.7,
        source_event_ids=[],
        metadata={"source": "contract"},
    )


def test_create_memory_allowed_namespace_type(store) -> None:
    async def scenario():
        memory = await store.create_memory(_create_command())
        events = await PostgresEventLog(store.engine).query(EventFilter())
        return memory, events

    memory, events = asyncio.run(scenario())

    assert memory.namespace == "project.personal_assistant"
    assert memory.memory_type == MemoryType.FACT
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.content_hash.startswith("sha256:")
    assert any(event.event_type == EventType.MEMORY_CREATED for event in events)


def test_reject_unknown_namespace(store) -> None:
    async def scenario() -> None:
        await store.create_memory(_create_command(namespace="unknown.namespace"))

    with pytest.raises(UnknownMemoryNamespace):
        asyncio.run(scenario())


def test_reject_invalid_memory_type(store) -> None:
    async def scenario() -> None:
        await store.create_memory(_create_command(memory_type="architecture_decision"))

    with pytest.raises(InvalidMemoryType):
        asyncio.run(scenario())


def test_reject_namespace_type_mismatch(store) -> None:
    async def scenario() -> None:
        await store.create_memory(
            _create_command(
                namespace="user.preferences",
                memory_type=MemoryType.FACT,
                sensitivity=Sensitivity.PERSONAL,
            ),
        )

    with pytest.raises(MemoryTypeNotAllowed):
        asyncio.run(scenario())


def test_reject_secret_memory_write(store) -> None:
    async def scenario() -> None:
        await store.create_memory(_create_command(sensitivity=Sensitivity.SECRET))

    with pytest.raises(MemoryPolicyDenied):
        asyncio.run(scenario())


def test_reject_secret_memory_write_records_policy_decision(store) -> None:
    async def scenario():
        with pytest.raises(MemoryPolicyDenied):
            await store.create_memory(
                _create_command(
                    memory_id="secret-denied",
                    sensitivity=Sensitivity.SECRET,
                ),
            )
        return await PostgresEventLog(store.engine).query(EventFilter())

    events = asyncio.run(scenario())

    policy_event = next(
        event for event in events if event.event_type == EventType.POLICY_DECISION_RECORDED
    )
    assert policy_event.payload["allowed"] is False
    assert policy_event.payload["source_ref"] == "memory_write:project.personal_assistant"


def test_allowed_memory_write_records_policy_decision(store) -> None:
    async def scenario():
        await store.create_memory(_create_command(memory_id="allowed-audit"))
        return await PostgresEventLog(store.engine).query(EventFilter())

    events = asyncio.run(scenario())

    policy_event = next(
        event for event in events if event.event_type == EventType.POLICY_DECISION_RECORDED
    )
    assert policy_event.payload["allowed"] is True
    assert policy_event.payload["source_ref"] == "memory_write:project.personal_assistant"


def test_archive_memory(store) -> None:
    async def scenario():
        memory = await store.create_memory(_create_command())
        await store.archive_memory(ArchiveMemoryCommand(memory_id=memory.id, reason="obsolete"))
        archived = await store.get_memory(memory.id)
        events = await PostgresEventLog(store.engine).query(EventFilter())
        return archived, events

    archived, events = asyncio.run(scenario())

    assert archived is not None
    assert archived.status == MemoryStatus.ARCHIVED
    assert archived.archive_reason == "obsolete"
    assert any(event.event_type == EventType.MEMORY_ARCHIVED for event in events)


def test_archive_memory_is_idempotent(store) -> None:
    async def scenario():
        memory = await store.create_memory(_create_command(memory_id="archive-idempotent"))
        await store.archive_memory(ArchiveMemoryCommand(memory_id=memory.id, reason="obsolete"))
        first = await store.get_memory(memory.id)
        await store.archive_memory(ArchiveMemoryCommand(memory_id=memory.id, reason="second"))
        second = await store.get_memory(memory.id)
        events = await PostgresEventLog(store.engine).query(EventFilter())
        return first, second, events

    first, second, events = asyncio.run(scenario())

    assert first is not None
    assert second is not None
    assert second.status == MemoryStatus.ARCHIVED
    assert second.archive_reason == first.archive_reason == "obsolete"
    assert second.archived_at == first.archived_at
    assert len([event for event in events if event.event_type == EventType.MEMORY_ARCHIVED]) == 1


def test_supersede_memory(store) -> None:
    async def scenario():
        old = await store.create_memory(_create_command(memory_id="old"))
        new = await store.supersede_memory(
            SupersedeMemoryCommand(
                superseded_memory_id=old.id,
                replacement=_create_command(
                    memory_id="new",
                    content="Phase 1 uses explicit memory writes and retrieval later.",
                ),
            ),
        )
        old_after = await store.get_memory(old.id)
        events = await PostgresEventLog(store.engine).query(EventFilter())
        return old_after, new, events

    old_after, new, events = asyncio.run(scenario())

    assert old_after is not None
    assert old_after.status == MemoryStatus.SUPERSEDED
    assert old_after.superseded_by_memory_id == new.id
    assert new.supersedes_memory_ids == [old_after.id]
    assert any(event.event_type == EventType.MEMORY_SUPERSEDED for event in events)


def test_memory_candidates_schema_exists_without_auto_extraction(store) -> None:
    async def scenario():
        async with store.engine.connect() as connection:
            table_name = await connection.scalar(text("select to_regclass('public.memory_candidates')"))
        await store.create_memory(_create_command())
        async with store.engine.connect() as connection:
            candidate_count = await connection.scalar(text("select count(*) from memory_candidates"))
        return table_name, candidate_count

    table_name, candidate_count = asyncio.run(scenario())

    assert table_name == "memory_candidates"
    assert candidate_count == 0
