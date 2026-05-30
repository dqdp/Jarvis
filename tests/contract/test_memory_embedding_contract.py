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
    CreateMemoryCommand,
    IndexingStatus,
    MemoryType,
    UpdateMemoryCommand,
)
from assistant_core.domain.models import EmbeddingRequest, EmbeddingResponse
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.embedding_port import GenerateEmbeddingCommand, ModelRouterEmbeddingPort
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _settings() -> Settings:
    return ConfigLoader(Path("config")).load("test")


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-memory-embedding-contract:{label}"))


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


class RecordingModelRouter:
    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(vectors=[[0.1, 0.2]])


class RecordingEmbeddingPort:
    def __init__(
        self,
        *,
        vectors: list[list[float]] | None = None,
        fail: bool = False,
    ) -> None:
        self.vectors = vectors or [[0.1, 0.2]]
        self.fail = fail
        self.calls: list[GenerateEmbeddingCommand] = []

    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse:
        self.calls.append(command)
        if self.fail:
            raise RuntimeError("embedding failed")
        return EmbeddingResponse(vectors=self.vectors)


class SequencedEmbeddingPort:
    def __init__(self) -> None:
        self.calls = 0
        self.started: dict[int, asyncio.Event] = {}
        self.release: dict[int, asyncio.Event] = {}

    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse:
        self.calls += 1
        call = self.calls
        self.started.setdefault(call, asyncio.Event()).set()
        release = self.release.get(call)
        if release is not None:
            await release.wait()
        return EmbeddingResponse(vectors=[[float(call)]])


@pytest.fixture
def store_factory():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_memory(database_url))
    engine = create_database_engine(database_url)
    settings = _settings()

    def make_store(embedding_port: RecordingEmbeddingPort):
        return PostgresMemoryStore(
            engine=engine,
            settings=settings,
            policy=ConfigPolicyEngine(settings),
            embedding_port=embedding_port,
        )

    try:
        yield make_store
    finally:
        asyncio.run(engine.dispose())


def _create_command(content: str = "Remember this") -> CreateMemoryCommand:
    return CreateMemoryCommand(
        memory_id=_id(content),
        namespace="project.personal_assistant",
        memory_type=MemoryType.FACT,
        content=content,
        summary=None,
        sensitivity=Sensitivity.PROJECT,
        confidence=0.9,
        importance=0.7,
        source_event_ids=[],
        metadata={},
    )


def test_embedding_port_delegates_to_model_router() -> None:
    async def scenario():
        router = RecordingModelRouter()
        port = ModelRouterEmbeddingPort(router=router, profile="local_embedding")
        response = await port.embed(
            GenerateEmbeddingCommand(texts=["hello"], sensitivity=Sensitivity.PROJECT),
        )
        return response, router.requests

    response, requests = asyncio.run(scenario())

    assert response.vectors == [[0.1, 0.2]]
    assert requests[0].profile == "local_embedding"
    assert requests[0].texts == ["hello"]


def test_create_memory_generates_embedding(store_factory) -> None:
    async def scenario():
        embedding_port = RecordingEmbeddingPort(vectors=[[0.4, 0.5]])
        store = store_factory(embedding_port)
        memory = await store.create_memory(_create_command())
        embedding = await store.get_current_embedding(memory.id, "local_embedding")
        return memory, embedding, embedding_port

    memory, embedding, embedding_port = asyncio.run(scenario())

    assert memory.indexing_status == IndexingStatus.INDEXED
    assert embedding is not None
    assert embedding.embedding == [0.4, 0.5]
    assert embedding.content_hash == memory.content_hash
    assert len(embedding_port.calls) == 1


def test_embedding_failure_keeps_memory_with_embedding_failed(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort(fail=True))
        memory = await store.create_memory(_create_command())
        embedding = await store.get_current_embedding(memory.id, "local_embedding")
        events = await PostgresEventLog(store.engine).query(EventFilter())
        return memory, embedding, events

    memory, embedding, events = asyncio.run(scenario())

    assert memory.indexing_status == IndexingStatus.EMBEDDING_FAILED
    assert embedding is None
    assert any(event.event_type == EventType.MEMORY_EMBEDDING_FAILED for event in events)


def test_update_memory_content_recomputes_embedding(store_factory) -> None:
    async def scenario():
        embedding_port = RecordingEmbeddingPort(vectors=[[0.1]])
        store = store_factory(embedding_port)
        memory = await store.create_memory(_create_command("old content"))
        embedding_port.vectors = [[0.8, 0.9]]
        updated = await store.update_memory(
            UpdateMemoryCommand(memory_id=memory.id, content="new content"),
        )
        embedding = await store.get_current_embedding(memory.id, "local_embedding")
        return updated, embedding, embedding_port

    updated, embedding, embedding_port = asyncio.run(scenario())

    assert updated.indexing_status == IndexingStatus.INDEXED
    assert embedding is not None
    assert embedding.embedding == [0.8, 0.9]
    assert embedding.content_hash == updated.content_hash
    assert len(embedding_port.calls) == 2


def test_metadata_only_update_keeps_current_embedding_when_provider_fails(store_factory) -> None:
    async def scenario():
        embedding_port = RecordingEmbeddingPort(vectors=[[0.7]])
        store = store_factory(embedding_port)
        memory = await store.create_memory(_create_command("metadata stable content"))
        embedding_port.fail = True
        updated = await store.update_memory(
            UpdateMemoryCommand(memory_id=memory.id, summary="metadata only"),
        )
        embedding = await store.get_current_embedding(memory.id, "local_embedding")
        return updated, embedding, embedding_port

    updated, embedding, embedding_port = asyncio.run(scenario())

    assert updated.indexing_status == IndexingStatus.INDEXED
    assert updated.summary == "metadata only"
    assert embedding is not None
    assert embedding.embedding == [0.7]
    assert embedding.content_hash == updated.content_hash
    assert len(embedding_port.calls) == 1


def test_stale_embedding_excluded_by_content_hash(store_factory) -> None:
    async def scenario():
        embedding_port = RecordingEmbeddingPort(vectors=[[0.1]])
        store = store_factory(embedding_port)
        memory = await store.create_memory(_create_command("stable content"))
        embedding_port.fail = True
        updated = await store.update_memory(
            UpdateMemoryCommand(memory_id=memory.id, content="changed content"),
        )
        current_embedding = await store.get_current_embedding(memory.id, "local_embedding")
        async with store.engine.connect() as connection:
            stored_embedding_count = await connection.scalar(
                text("select count(*) from memory_embeddings where memory_id = :memory_id"),
                {"memory_id": memory.id},
            )
        return updated, current_embedding, stored_embedding_count

    updated, current_embedding, stored_embedding_count = asyncio.run(scenario())

    assert updated.indexing_status == IndexingStatus.EMBEDDING_FAILED
    assert stored_embedding_count == 1
    assert current_embedding is None


def test_stale_embedding_sync_cannot_overwrite_newer_memory_content(store_factory) -> None:
    async def scenario():
        embedding_port = SequencedEmbeddingPort()
        store = store_factory(embedding_port)
        memory = await store.create_memory(_create_command("initial content"))
        embedding_port.started[2] = asyncio.Event()
        embedding_port.release[2] = asyncio.Event()
        stale_task = asyncio.create_task(
            store.update_memory(UpdateMemoryCommand(memory_id=memory.id, content="first update")),
        )
        await asyncio.wait_for(embedding_port.started[2].wait(), timeout=1)
        fresh = await store.update_memory(
            UpdateMemoryCommand(memory_id=memory.id, content="second update"),
        )
        fresh_embedding = await store.get_current_embedding(memory.id, "local_embedding")
        embedding_port.release[2].set()
        stale_result = await stale_task
        final = await store.get_memory(memory.id)
        final_embedding = await store.get_current_embedding(memory.id, "local_embedding")
        return fresh, fresh_embedding, stale_result, final, final_embedding

    fresh, fresh_embedding, stale_result, final, final_embedding = asyncio.run(scenario())

    assert final is not None
    assert final.content == "second update"
    assert final.indexing_status == IndexingStatus.INDEXED
    assert fresh_embedding is not None
    assert final_embedding is not None
    assert final_embedding.content_hash == final.content_hash
    assert final_embedding.embedding == fresh_embedding.embedding == [3.0]
    assert stale_result.content == "second update"
    assert stale_result.indexing_status == IndexingStatus.INDEXED
