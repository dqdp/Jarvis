from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.domain.content_retrieval import (
    ContentChunk,
    ContentChunkStatus,
    ContentCitation,
    ContentHit,
    ContentRetrievalQuery,
    ContentSourceType,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.memory import MemoryHit
from assistant_core.domain.models import EmbeddingResponse
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.embedding import GenerateEmbeddingCommand
from assistant_core.ports.content_retrieval import ContentRetrievalPort
from assistant_core.ports.event_log import EventFilter
from assistant_core.storage.content_store import PostgresContentStore
from assistant_core.content_retrieval.project_docs import ProjectDocsSourceCandidate
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.migrations import run_migrations


pytestmark = [pytest.mark.contract, pytest.mark.db]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-content-retrieval-contract:{label}"))


async def _truncate_content(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table content_embeddings, content_chunks, content_sources, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


class RecordingEmbeddingPort:
    def __init__(self, *, fail_times: int = 0, dimension: int | None = None) -> None:
        self.fail_times = fail_times
        self.dimension = dimension
        self.calls: list[GenerateEmbeddingCommand] = []

    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse:
        self.calls.append(command)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("fake content embedding failure")
        if self.dimension is not None:
            return EmbeddingResponse(vectors=[[1.0] * self.dimension for _ in command.texts])
        return EmbeddingResponse(vectors=[_vector(text) for text in command.texts])


class _ModelProfileStub:
    def __init__(self, *, model: str, dimension: int) -> None:
        self.model = model
        self.dimension = dimension


class _SettingsStub:
    def __init__(self, *, model: str, dimension: int) -> None:
        self.model_profiles = {
            "local_embedding": _ModelProfileStub(model=model, dimension=dimension),
        }


@pytest.fixture
def store_factory():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_content(database_url))
    engine = create_database_engine(database_url)

    def make_store(
        embedding_port: RecordingEmbeddingPort,
        *,
        settings=None,
    ):
        return PostgresContentStore(engine=engine, embedding_port=embedding_port, settings=settings)

    try:
        yield make_store
    finally:
        asyncio.run(engine.dispose())


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float(lowered.count("project")),
        float(lowered.count("docs")),
        float(lowered.count("alpha")),
        float(lowered.count("beta")),
        1.0,
    ]


def _candidate(label: str, *, content_hash: str | None = None) -> ProjectDocsSourceCandidate:
    return ProjectDocsSourceCandidate(
        source_id=_id(f"source-{label}"),
        relative_path=Path(f"docs/{label}.md"),
        absolute_path=Path(f"/tmp/docs/{label}.md"),
        source_type=ContentSourceType.PROJECT_DOC,
        title=f"{label.title()} Guide",
        content=f"# {label.title()}\nalpha project docs\n",
        content_hash=content_hash or f"sha256:{label}",
        sensitivity=Sensitivity.PROJECT,
    )


def _chunk(
    label: str,
    *,
    source_id: str | None = None,
    content: str = "alpha project docs",
    content_hash: str | None = None,
    status: ContentChunkStatus = ContentChunkStatus.ACTIVE,
) -> ContentChunk:
    source_id = source_id or _id(f"source-{label}")
    path = Path(f"docs/{label}.md")
    return ContentChunk(
        chunk_id=_id(f"chunk-{label}-{content_hash or 'current'}"),
        source_id=source_id,
        ordinal=0,
        source_path=path,
        source_type=ContentSourceType.PROJECT_DOC,
        heading_path=[f"{label.title()} Guide"],
        content=content,
        content_hash=content_hash or f"sha256:{label}",
        line_start=1,
        line_end=3,
        citation=ContentCitation(path=path, line_start=1, line_end=3, heading_path=[f"{label.title()} Guide"]),
        sensitivity=Sensitivity.PROJECT,
        status=status,
        metadata={},
    )


async def _sync_one(
    store: PostgresContentStore,
    *,
    label: str = "guide",
    content: str = "alpha project docs",
    content_hash: str | None = None,
) -> ContentChunk:
    candidate = _candidate(label, content_hash=content_hash)
    chunk = _chunk(
        label,
        source_id=candidate.source_id,
        content=content,
        content_hash=content_hash,
    )
    await store.sync_source_chunks(candidate, [chunk])
    return chunk


def test_content_retrieval_port_returns_content_hits_not_memory_hits(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        assert isinstance(store, ContentRetrievalPort)
        await _sync_one(store)
        return await store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=3))

    hits = asyncio.run(scenario())

    assert hits
    assert isinstance(hits[0], ContentHit)
    assert not isinstance(hits[0], MemoryHit)


def test_content_hit_contains_source_chunk_score_citation_and_hash(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        chunk = await _sync_one(store, content_hash="sha256:current")
        hits = await store.retrieve(ContentRetrievalQuery(text="alpha docs", limit=1))
        return chunk, hits[0]

    chunk, hit = asyncio.run(scenario())

    assert hit.source_id == chunk.source_id
    assert hit.chunk_id == chunk.chunk_id
    assert hit.score > 0
    assert hit.citation.format() == "docs/guide.md:1-3"
    assert hit.content_hash == "sha256:current"
    assert hit.sensitivity is Sensitivity.PROJECT


def test_retrieval_excludes_stale_chunks(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        old = await _sync_one(store, content="alpha project docs", content_hash="sha256:old")
        new = await _sync_one(store, content="beta project docs", content_hash="sha256:new")
        chunks = await store.list_chunks()
        hits = await store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=10))
        return old, new, chunks, hits

    old, new, chunks, hits = asyncio.run(scenario())

    stale_ids = {chunk.chunk_id for chunk in chunks if chunk.status is ContentChunkStatus.STALE}
    assert old.chunk_id in stale_ids
    assert new.chunk_id not in stale_ids
    assert all(hit.chunk_id not in stale_ids for hit in hits)


def test_retrieval_excludes_deleted_chunks(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        await _sync_one(store)
        await store.mark_missing_sources_deleted(set())
        return await store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=10))

    assert asyncio.run(scenario()) == []


def test_retrieval_returns_citations(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        await _sync_one(store)
        return await store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=1))

    hits = asyncio.run(scenario())

    assert hits[0].citation.path == Path("docs/guide.md")
    assert hits[0].citation.line_start == 1
    assert hits[0].citation.line_end == 3


def test_fake_embedding_provider_is_used_for_content_embeddings(store_factory) -> None:
    async def scenario():
        embedding_port = RecordingEmbeddingPort()
        store = store_factory(embedding_port)
        await _sync_one(store)
        await store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=1))
        return embedding_port.calls

    calls = asyncio.run(scenario())

    assert [call.texts for call in calls] == [["alpha project docs"], ["alpha project docs"]]
    assert all(call.sensitivity is Sensitivity.PROJECT for call in calls)


def test_embedding_failure_excludes_failed_chunks_from_retrieval(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort(fail_times=1))
        await _sync_one(store)
        return await store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=10))

    assert asyncio.run(scenario()) == []


def test_retrieval_records_empty_result_event(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        request_id = _id("empty-result-request")
        hits = await store.retrieve(
            ContentRetrievalQuery(
                text="alpha project docs",
                request_id=request_id,
                correlation_id=request_id,
                sensitivity=Sensitivity.PERSONAL,
            ),
        )
        events = await PostgresEventLog(store.engine).query(EventFilter(request_id=request_id))
        return hits, events

    hits, events = asyncio.run(scenario())

    assert hits == []
    event = next(event for event in events if event.event_type is EventType.CONTENT_RETRIEVED)
    assert event.sensitivity is Sensitivity.PERSONAL
    assert event.payload["retrieved_content_refs"] == []
    assert event.payload["full_content_stored"] is False


def test_query_embedding_failure_records_content_retrieval_failed_event(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        await _sync_one(store)
        request_id = _id("query-embedding-failed")
        retrieval_store = PostgresContentStore(
            engine=store.engine,
            embedding_port=RecordingEmbeddingPort(fail_times=1),
            settings=_SettingsStub(model="local_embedding", dimension=5),
        )
        hits = await retrieval_store.retrieve(
            ContentRetrievalQuery(
                text="alpha project docs",
                request_id=request_id,
                correlation_id=request_id,
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        events = await PostgresEventLog(store.engine).query(EventFilter(request_id=request_id))
        return hits, events

    hits, events = asyncio.run(scenario())

    assert hits == []
    event = next(event for event in events if event.event_type is EventType.CONTENT_RETRIEVAL_FAILED)
    assert event.payload["reason"] == "query_embedding_failed"
    assert event.payload["error_type"] == "RuntimeError"
    assert event.payload["full_query_stored"] is False


def test_secret_content_rows_are_rejected_by_database(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        chunk = await _sync_one(store)
        async with store.engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text(
                        "update content_sources set sensitivity = 'secret' "
                        "where source_id = cast(:source_id as uuid)",
                    ),
                    {"source_id": chunk.source_id},
                )
        async with store.engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text(
                        "update content_chunks set sensitivity = 'secret' "
                        "where chunk_id = cast(:chunk_id as uuid)",
                    ),
                    {"chunk_id": chunk.chunk_id},
                )
        return await store.retrieve(
            ContentRetrievalQuery(
                text="alpha project docs",
                exclude_sensitivities=[],
                limit=10,
            ),
        )

    assert len(asyncio.run(scenario())) == 1


def test_secret_content_sources_are_rejected_by_database(store_factory) -> None:
    async def scenario():
        store = store_factory(RecordingEmbeddingPort())
        chunk = await _sync_one(store)
        async with store.engine.begin() as connection:
            with pytest.raises(Exception):
                await connection.execute(
                    text(
                        "update content_sources set sensitivity = 'secret' "
                        "where source_id = cast(:source_id as uuid)",
                    ),
                    {"source_id": chunk.source_id},
                )
        return await store.retrieve(
            ContentRetrievalQuery(
                text="alpha project docs",
                exclude_sensitivities=[],
                limit=10,
            ),
        )

    assert len(asyncio.run(scenario())) == 1


def test_embedding_model_change_reindexes_current_chunk(store_factory) -> None:
    async def scenario():
        initial_port = RecordingEmbeddingPort(dimension=5)
        initial_store = store_factory(
            initial_port,
            settings=_SettingsStub(model="embed-v1", dimension=5),
        )
        chunk = await _sync_one(initial_store)
        retry_port = RecordingEmbeddingPort(dimension=6)
        retry_store = store_factory(
            retry_port,
            settings=_SettingsStub(model="embed-v2", dimension=6),
        )
        await _sync_one(retry_store)
        current = await retry_store.get_current_embedding(chunk.chunk_id, "local_embedding")
        return initial_port.calls, retry_port.calls, current

    initial_calls, retry_calls, current = asyncio.run(scenario())

    assert [call.texts for call in initial_calls] == [["alpha project docs"]]
    assert [call.texts for call in retry_calls] == [["alpha project docs"]]
    assert current is not None
    assert current.embedding_model == "embed-v2"
    assert current.embedding_dimension == 6


def test_retrieval_excludes_stale_embedding_model_before_reingestion(store_factory) -> None:
    async def scenario():
        initial_store = store_factory(
            RecordingEmbeddingPort(dimension=5),
            settings=_SettingsStub(model="embed-v1", dimension=5),
        )
        await _sync_one(initial_store)
        retrieval_port = RecordingEmbeddingPort(dimension=6)
        retrieval_store = store_factory(
            retrieval_port,
            settings=_SettingsStub(model="embed-v2", dimension=6),
        )
        hits = await retrieval_store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=10))
        return hits, retrieval_port.calls

    hits, calls = asyncio.run(scenario())

    assert hits == []
    assert calls == []


def test_retrieval_excludes_stale_embedding_dimension_before_reingestion(store_factory) -> None:
    async def scenario():
        initial_store = store_factory(
            RecordingEmbeddingPort(dimension=5),
            settings=_SettingsStub(model="embed-v1", dimension=5),
        )
        await _sync_one(initial_store)
        retrieval_port = RecordingEmbeddingPort(dimension=6)
        retrieval_store = store_factory(
            retrieval_port,
            settings=_SettingsStub(model="embed-v1", dimension=6),
        )
        hits = await retrieval_store.retrieve(ContentRetrievalQuery(text="alpha project docs", limit=10))
        return hits, retrieval_port.calls

    hits, calls = asyncio.run(scenario())

    assert hits == []
    assert calls == []
