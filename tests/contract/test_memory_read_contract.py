from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader, Settings
from assistant_core.domain.memory import (
    ArchiveMemoryCommand,
    CreateMemoryCommand,
    IndexingStatus,
    MemoryQuery,
    MemoryStatus,
    MemoryType,
    SupersedeMemoryCommand,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.fake_provider import FakeEmbeddingProvider
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.memory import MemoryRetrievalError
from assistant_core.storage.database import assert_test_database_url, create_database_engine
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
    return str(uuid5(NAMESPACE_URL, f"jarvis-memory-read-contract:{label}"))


async def _truncate_memory(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table memory_embeddings, memory_candidates, memories, events "
                    "restart identity cascade",
                ),
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
            embedding_port=FakeEmbeddingProvider(),
        )
    finally:
        asyncio.run(engine.dispose())


def _create_command(
    label: str,
    *,
    namespace: str = "project.personal_assistant",
    memory_type: MemoryType = MemoryType.FACT,
    content: str | None = None,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
    importance: float = 0.5,
) -> CreateMemoryCommand:
    return CreateMemoryCommand(
        memory_id=_id(label),
        namespace=namespace,
        memory_type=memory_type,
        content=content or f"{label} phase memory",
        summary=None,
        sensitivity=sensitivity,
        confidence=0.9,
        importance=importance,
        source_event_ids=[],
        metadata={},
    )


def _query(**kwargs) -> MemoryQuery:
    data = {"text": "phase memory", "namespaces": ["project.personal_assistant"]}
    data.update(kwargs)
    return MemoryQuery(**data)


def test_retrieve_active_memories_only(store) -> None:
    async def scenario():
        active = await store.create_memory(_create_command("active"))
        archived = await store.create_memory(_create_command("archived"))
        await store.archive_memory(ArchiveMemoryCommand(memory_id=archived.id, reason="old"))
        hits = await store.retrieve(_query())
        return active, archived, hits

    active, archived, hits = asyncio.run(scenario())

    assert [hit.memory.id for hit in hits] == [active.id]
    assert archived.id not in [hit.memory.id for hit in hits]


def test_get_memory_returns_record_by_id(store) -> None:
    async def scenario():
        memory = await store.create_memory(_create_command("get-by-id"))
        return memory, await store.get_memory(memory.id)

    memory, loaded = asyncio.run(scenario())

    assert loaded is not None
    assert loaded.id == memory.id
    assert loaded.content == memory.content


def test_list_memories_filters_by_query(store) -> None:
    async def scenario():
        target = await store.create_memory(
            _create_command("list-query-target", content="Alpha memory search target."),
        )
        await store.create_memory(
            _create_command("list-query-other", content="Unrelated note."),
        )
        return target, await store.list_memories(query="search target")

    target, memories = asyncio.run(scenario())

    assert [memory.id for memory in memories] == [target.id]


def test_list_memories_treats_query_wildcards_literally(store) -> None:
    async def scenario():
        literal = await store.create_memory(
            _create_command("literal-wildcard", content="Literal under_score memory."),
        )
        await store.create_memory(_create_command("wildcard-other", content="Other memory."))
        return literal, await store.list_memories(query="_")

    literal, memories = asyncio.run(scenario())

    assert [memory.id for memory in memories] == [literal.id]


def test_list_memories_treats_percent_wildcard_literally(store) -> None:
    async def scenario():
        literal = await store.create_memory(
            _create_command("literal-percent", content="Literal percent% memory."),
        )
        await store.create_memory(_create_command("percent-other", content="Other memory."))
        return literal, await store.list_memories(query="%")

    literal, memories = asyncio.run(scenario())

    assert [memory.id for memory in memories] == [literal.id]


def test_exclude_archived(store) -> None:
    async def scenario():
        memory = await store.create_memory(_create_command("archived"))
        await store.archive_memory(ArchiveMemoryCommand(memory_id=memory.id, reason="old"))
        return await store.retrieve(_query())

    assert asyncio.run(scenario()) == []


def test_exclude_superseded(store) -> None:
    async def scenario():
        old = await store.create_memory(_create_command("old"))
        new = await store.supersede_memory(
            SupersedeMemoryCommand(
                superseded_memory_id=old.id,
                replacement=_create_command("new", content="new phase memory"),
            ),
        )
        hits = await store.retrieve(_query())
        return old, new, hits

    old, new, hits = asyncio.run(scenario())

    assert [hit.memory.id for hit in hits] == [new.id]
    assert old.id not in [hit.memory.id for hit in hits]


def test_filter_by_namespace(store) -> None:
    async def scenario():
        project = await store.create_memory(_create_command("project"))
        working = await store.create_memory(
            _create_command(
                "working",
                namespace="user.working_style",
                memory_type=MemoryType.PROCEDURE,
                sensitivity=Sensitivity.PERSONAL,
                content="working phase memory",
            ),
        )
        hits = await store.retrieve(
            MemoryQuery(text="phase", namespaces=["user.working_style"]),
        )
        return project, working, hits

    project, working, hits = asyncio.run(scenario())

    assert [hit.memory.id for hit in hits] == [working.id]
    assert project.id not in [hit.memory.id for hit in hits]


def test_empty_namespace_query_returns_no_hits(store) -> None:
    async def scenario():
        await store.create_memory(_create_command("project"))
        return await store.retrieve(MemoryQuery(text="phase", namespaces=[]))

    assert asyncio.run(scenario()) == []


def test_secret_memory_records_are_rejected_by_database(store) -> None:
    async def scenario():
        visible = await store.create_memory(_create_command("visible"))
        async with store.engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text(
                        """
                        insert into memories (
                          memory_id, namespace, memory_type, content, summary,
                          content_hash, sensitivity, confidence, importance,
                          status, indexing_status, source_event_ids, supersedes_memory_ids,
                          revision, created_at, updated_at, metadata
                        )
                        values (
                          cast(:memory_id as uuid), 'project.personal_assistant', 'fact',
                          'secret phase memory', null, 'sha256:secret', 'secret', 1, 1,
                          'active', 'embedding_pending', '{}', '{}', 1, now(), now(), '{}'
                        )
                        """,
                    ),
                    {"memory_id": _id("secret")},
                )
        return visible, await store.retrieve(_query())

    visible, hits = asyncio.run(scenario())

    assert len(hits) == 1
    assert hits[0].memory.id == visible.id
    assert [hit.memory.sensitivity for hit in hits] == [Sensitivity.PROJECT]


def test_retrieve_respects_configured_excluded_sensitivity(store) -> None:
    async def scenario():
        base_settings = _settings()
        restricted_settings = replace(
            base_settings,
            memory=replace(
                base_settings.memory,
                retrieval=replace(
                    base_settings.memory.retrieval,
                    exclude_sensitivity=["secret", "infra"],
                ),
            ),
        )
        restricted_store = PostgresMemoryStore(
            engine=store.engine,
            settings=restricted_settings,
            policy=ConfigPolicyEngine(restricted_settings),
            embedding_port=FakeEmbeddingProvider(),
        )
        project = await restricted_store.create_memory(_create_command("visible-config-project"))
        infra = await restricted_store.create_memory(
            _create_command(
                "excluded-config-infra",
                namespace="environment.inference_node",
                sensitivity=Sensitivity.INFRA,
                content="infra phase memory",
            ),
        )
        hits = await restricted_store.retrieve(
            MemoryQuery(
                text="phase memory",
                namespaces=["project.personal_assistant", "environment.inference_node"],
            ),
        )
        return project, infra, hits

    project, infra, hits = asyncio.run(scenario())
    hit_ids = [hit.memory.id for hit in hits]

    assert project.id in hit_ids
    assert infra.id not in hit_ids


def test_retrieve_respects_configured_min_score(store) -> None:
    async def scenario():
        base_settings = _settings()
        threshold_settings = replace(
            base_settings,
            memory=replace(
                base_settings.memory,
                retrieval=replace(base_settings.memory.retrieval, min_score=999.0),
            ),
        )
        threshold_store = PostgresMemoryStore(
            engine=store.engine,
            settings=threshold_settings,
            policy=ConfigPolicyEngine(threshold_settings),
            embedding_port=FakeEmbeddingProvider(),
        )
        await threshold_store.create_memory(_create_command("below-min-score"))
        return await threshold_store.retrieve(_query())

    assert asyncio.run(scenario()) == []


def test_exclude_memories_without_current_indexed_embedding(store) -> None:
    async def scenario():
        failed = await store.create_memory(_create_command("failed-index", content="failed phase memory"))
        pending = await store.create_memory(_create_command("pending-index", content="pending phase memory"))
        stale = await store.create_memory(_create_command("stale-index", content="stale phase memory"))
        visible = await store.create_memory(_create_command("visible-index", content="visible phase memory"))
        async with store.engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    update memories
                    set indexing_status = :status
                    where memory_id = cast(:memory_id as uuid)
                    """,
                ),
                {"status": IndexingStatus.EMBEDDING_FAILED.value, "memory_id": failed.id},
            )
            await connection.execute(
                text(
                    """
                    update memories
                    set indexing_status = :status
                    where memory_id = cast(:memory_id as uuid)
                    """,
                ),
                {"status": IndexingStatus.EMBEDDING_PENDING.value, "memory_id": pending.id},
            )
            await connection.execute(
                text(
                    """
                    update memories
                    set content_hash = 'sha256:changed'
                    where memory_id = cast(:memory_id as uuid)
                    """,
                ),
                {"memory_id": stale.id},
            )
        hits = await store.retrieve(_query(text="phase memory"))
        return failed, pending, stale, visible, hits

    failed, pending, stale, visible, hits = asyncio.run(scenario())
    hit_ids = [hit.memory.id for hit in hits]

    assert visible.id in hit_ids
    assert failed.id not in hit_ids
    assert pending.id not in hit_ids
    assert stale.id not in hit_ids


def test_respects_max_hits_total(store) -> None:
    async def scenario():
        for index in range(5):
            await store.create_memory(_create_command(f"project-{index}"))
            await store.create_memory(
                _create_command(
                    f"working-total-{index}",
                    namespace="user.working_style",
                    memory_type=MemoryType.PROCEDURE,
                    sensitivity=Sensitivity.PERSONAL,
                ),
            )
        return await store.retrieve(
            MemoryQuery(
                text="phase memory",
                namespaces=["project.personal_assistant", "user.working_style"],
                limit=10,
            ),
        )

    hits = asyncio.run(scenario())

    assert len(hits) == 8


def test_respects_max_hits_per_namespace(store) -> None:
    async def scenario():
        for index in range(6):
            await store.create_memory(_create_command(f"project-{index}"))
            await store.create_memory(
                _create_command(
                    f"working-{index}",
                    namespace="user.working_style",
                    memory_type=MemoryType.PROCEDURE,
                    sensitivity=Sensitivity.PERSONAL,
                ),
            )
        return await store.retrieve(
            MemoryQuery(
                text="phase memory",
                namespaces=["project.personal_assistant", "user.working_style"],
                limit=10,
            ),
        )

    hits = asyncio.run(scenario())
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.memory.namespace] = counts.get(hit.memory.namespace, 0) + 1

    assert counts["project.personal_assistant"] <= 4
    assert counts["user.working_style"] <= 4


def test_ranking_score_importance_recency(store) -> None:
    async def scenario():
        low = await store.create_memory(
            _create_command("low", content="ranked phase memory", importance=0.1),
        )
        high = await store.create_memory(
            _create_command("high", content="ranked phase memory", importance=0.9),
        )
        hits = await store.retrieve(_query(text="ranked phase memory"))
        return low, high, hits

    low, high, hits = asyncio.run(scenario())

    assert hits[0].memory.id == high.id
    assert hits[0].score > hits[-1].score
    assert low.id in [hit.memory.id for hit in hits]


def test_retrieval_failure_can_be_reported(store) -> None:
    async def scenario() -> None:
        async with store.engine.begin() as connection:
            await connection.execute(text("alter table memories rename to memories_unavailable"))
        try:
            await store.retrieve(_query())
        finally:
            async with store.engine.begin() as connection:
                await connection.execute(text("alter table memories_unavailable rename to memories"))

    with pytest.raises(MemoryRetrievalError):
        asyncio.run(scenario())
