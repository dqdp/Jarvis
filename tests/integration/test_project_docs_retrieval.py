from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy import text

from assistant_core.content_retrieval.project_docs import (
    MarkdownChunker,
    ProjectDocsIngestionService,
    ProjectDocsSourceScanner,
)
from assistant_core.domain.content_retrieval import ContentEmbeddingStatus, ContentRetrievalQuery
from assistant_core.domain.models import EmbeddingResponse
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.embedding import GenerateEmbeddingCommand
from assistant_core.storage.content_store import PostgresContentStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.migrations import run_migrations


pytestmark = [pytest.mark.integration, pytest.mark.db]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def _truncate_content_and_memory(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table content_embeddings, content_chunks, content_sources, "
                    "memory_embeddings, memory_candidates, memories, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


class RecordingEmbeddingPort:
    def __init__(self, *, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls: list[GenerateEmbeddingCommand] = []

    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse:
        self.calls.append(command)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("fake content embedding failure")
        return EmbeddingResponse(vectors=[_vector(text) for text in command.texts])


@pytest.fixture
def content_store_factory():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_content_and_memory(database_url))
    engine = create_database_engine(database_url)

    def make_store(embedding_port: RecordingEmbeddingPort):
        return PostgresContentStore(engine=engine, embedding_port=embedding_port)

    try:
        yield make_store
    finally:
        asyncio.run(engine.dispose())


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float(lowered.count("project")),
        float(lowered.count("docs")),
        float(lowered.count("retrieval")),
        1.0,
    ]


def _service(root: Path, store: PostgresContentStore) -> ProjectDocsIngestionService:
    return ProjectDocsIngestionService(
        store=store,
        scanner=ProjectDocsSourceScanner(project_root=root),
        chunker=MarkdownChunker(max_chars=160),
    )


async def _scalar(store: PostgresContentStore, statement: str, params: dict | None = None):
    async with store.engine.connect() as connection:
        return await connection.scalar(text(statement), params or {})


def test_ingestion_creates_content_embeddings(
    tmp_path: Path,
    content_store_factory,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nproject docs retrieval\n")

    async def scenario():
        embedding_port = RecordingEmbeddingPort()
        store = content_store_factory(embedding_port)
        await _service(tmp_path, store).ingest()
        return (
            await _scalar(store, "select count(*) from content_embeddings"),
            embedding_port.calls,
        )

    embedding_count, calls = asyncio.run(scenario())

    assert embedding_count == 1
    assert [call.texts for call in calls] == [["# Guide\nproject docs retrieval"]]


def test_unchanged_reingestion_creates_missing_content_embeddings(
    tmp_path: Path,
    content_store_factory,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nproject docs retrieval\n")

    async def scenario():
        initial_store = content_store_factory(RecordingEmbeddingPort())
        await _service(tmp_path, initial_store).ingest()
        async with initial_store.engine.begin() as connection:
            await connection.execute(text("delete from content_embeddings"))
        embedding_port = RecordingEmbeddingPort()
        retry_store = content_store_factory(embedding_port)
        await _service(tmp_path, retry_store).ingest()
        return (
            await _scalar(retry_store, "select count(*) from content_embeddings"),
            embedding_port.calls,
        )

    embedding_count, calls = asyncio.run(scenario())

    assert embedding_count == 1
    assert [call.texts for call in calls] == [["# Guide\nproject docs retrieval"]]


def test_embedding_failure_marks_chunk_or_embedding_failed(
    tmp_path: Path,
    content_store_factory,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nproject docs retrieval\n")

    async def scenario():
        store = content_store_factory(RecordingEmbeddingPort(fail_times=1))
        await _service(tmp_path, store).ingest()
        return await _scalar(store, "select status from content_embeddings")

    assert asyncio.run(scenario()) == ContentEmbeddingStatus.FAILED.value


def test_retrieval_uses_content_embeddings_not_memory_embeddings(
    tmp_path: Path,
    content_store_factory,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nproject docs retrieval\n")

    async def scenario():
        store = content_store_factory(RecordingEmbeddingPort())
        await _service(tmp_path, store).ingest()
        hits_before = await store.retrieve(ContentRetrievalQuery(text="project docs retrieval", limit=5))
        async with store.engine.begin() as connection:
            await connection.execute(text("delete from content_embeddings"))
        hits_after = await store.retrieve(ContentRetrievalQuery(text="project docs retrieval", limit=5))
        memory_embedding_count = await _scalar(store, "select count(*) from memory_embeddings")
        return hits_before, hits_after, memory_embedding_count

    hits_before, hits_after, memory_embedding_count = asyncio.run(scenario())

    assert hits_before
    assert hits_after == []
    assert memory_embedding_count == 0


def test_content_tables_remain_separate_from_memory_tables() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    engine = create_database_engine(database_url)

    async def scenario():
        try:
            async with engine.connect() as connection:
                content_embeddings = await connection.scalar(
                    text("select to_regclass('public.content_embeddings')"),
                )
                memory_content_columns = await connection.scalar(
                    text(
                        """
                        select count(*)
                        from information_schema.columns
                        where table_name in ('memories', 'memory_embeddings')
                        and column_name in ('chunk_id', 'source_id', 'heading_path', 'citation')
                        """,
                    ),
                )
                content_memory_columns = await connection.scalar(
                    text(
                        """
                        select count(*)
                        from information_schema.columns
                        where table_name in ('content_sources', 'content_chunks', 'content_embeddings')
                        and column_name in ('memory_id', 'namespace', 'memory_type')
                        """,
                    ),
                )
                return content_embeddings, memory_content_columns, content_memory_columns
        finally:
            await engine.dispose()

    content_embeddings, memory_content_columns, content_memory_columns = asyncio.run(scenario())

    assert content_embeddings == "content_embeddings"
    assert memory_content_columns == 0
    assert content_memory_columns == 0
