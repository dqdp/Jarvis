from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from assistant_core.content_retrieval.project_docs import (
    MarkdownChunker,
    ProjectDocsIngestionService,
    ProjectDocsSourceCandidate,
    ProjectDocsSourceScanner,
)
from assistant_core.domain.content_retrieval import (
    ContentChunk,
    ContentChunkStatus,
    ContentCitation,
    ContentSourceStatus,
    ContentSourceType,
)
from assistant_core.domain.sensitivity import Sensitivity
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


async def _truncate_content(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table content_embeddings, content_chunks, content_sources "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


async def _scalar(database_url: str, statement: str):
    engine = create_database_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.scalar(text(statement))
    finally:
        await engine.dispose()


async def _chunk_created_at_rows(store: PostgresContentStore, source_id: str):
    async with store.engine.connect() as connection:
        return (
            await connection.execute(
                text(
                    """
                    select chunk_id::text, status, created_at
                    from content_chunks
                    where source_id = cast(:source_id as uuid)
                    order by chunk_id
                    """,
                ),
                {"source_id": source_id},
            )
        ).all()


async def _set_source_last_seen(store: PostgresContentStore, path: Path, value: datetime) -> None:
    async with store.engine.begin() as connection:
        await connection.execute(
            text(
                """
                update content_sources
                set last_seen_at = cast(:value as timestamptz)
                where path = :path
                """,
            ),
            {"path": path.as_posix(), "value": value},
        )


@pytest.fixture
def content_store():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_content(database_url))
    engine = create_database_engine(database_url)
    try:
        yield PostgresContentStore(engine=engine)
    finally:
        asyncio.run(engine.dispose())


def _service(root: Path, store: PostgresContentStore) -> ProjectDocsIngestionService:
    return ProjectDocsIngestionService(
        store=store,
        scanner=ProjectDocsSourceScanner(project_root=root),
        chunker=MarkdownChunker(max_chars=160),
    )


class _FailAtomicSyncStore:
    def __init__(self) -> None:
        self.calls = 0

    async def sync_source_chunks(self, candidate, chunks):
        self.calls += 1
        raise RuntimeError("simulated atomic sync failure")

    async def mark_missing_sources_deleted(self, seen_paths: set[Path]) -> tuple[int, int]:
        return 0, 0


def test_source_registry_creates_project_doc_sources(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "README.md", "# Readme\n")
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "docs/adr/ADR-001_decision.md", "# Decision\n")

    async def scenario():
        await _service(tmp_path, content_store).ingest()
        return await content_store.list_sources()

    sources = asyncio.run(scenario())
    by_path = {source.path.as_posix(): source for source in sources}
    assert set(by_path) == {
        "README.md",
        "docs/guide.md",
        "docs/adr/ADR-001_decision.md",
    }
    assert by_path["README.md"].source_type is ContentSourceType.README
    assert by_path["docs/guide.md"].source_type is ContentSourceType.PROJECT_DOC
    assert by_path["docs/adr/ADR-001_decision.md"].source_type is ContentSourceType.ADR
    assert {source.status for source in sources} == {ContentSourceStatus.ACTIVE}


def test_source_registry_updates_content_hash_on_change(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "README.md", "# Readme\nold\n")
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        before = await content_store.get_source_by_path(Path("README.md"))
        assert before is not None
        _write(tmp_path, "README.md", "# Readme\nnew\n")
        await service.ingest()
        after = await content_store.get_source_by_path(Path("README.md"))
        return before, after

    before, after = asyncio.run(scenario())
    assert after is not None
    assert after.source_id == before.source_id
    assert after.content_hash != before.content_hash
    assert after.status is ContentSourceStatus.ACTIVE


def test_ingestion_creates_content_sources_and_chunks(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nintro\n## Setup\ninstall\n")

    async def scenario():
        result = await _service(tmp_path, content_store).ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        chunks = await content_store.list_chunks(source_id=source.source_id)
        return result, source, chunks

    result, source, chunks = asyncio.run(scenario())
    assert result.created_chunks == len(chunks)
    assert [chunk.heading_path for chunk in chunks] == [["Guide"], ["Guide", "Setup"]]
    assert all(chunk.source_id == source.source_id for chunk in chunks)
    assert all(chunk.content_hash == source.content_hash for chunk in chunks)
    assert all(chunk.citation.format().startswith("docs/guide.md:") for chunk in chunks)


def test_ingestion_does_not_persist_secret_like_content(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nsafe project docs\n")
    _write(tmp_path, "docs/setup.md", "# Setup\nOPENAI_API_KEY=sk-live-secret-value-1234567890\n")

    async def scenario():
        result = await _service(tmp_path, content_store).ingest()
        sources = await content_store.list_sources()
        chunks = await content_store.list_chunks()
        return result, sources, chunks

    result, sources, chunks = asyncio.run(scenario())
    assert result.seen_sources == 1
    assert [source.path.as_posix() for source in sources] == ["docs/guide.md"]
    assert all("OPENAI_API_KEY" not in chunk.content for chunk in chunks)


def test_reingestion_marks_old_chunks_stale(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nold\n")
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        first_chunks = await content_store.list_chunks(source_id=source.source_id)
        _write(tmp_path, "docs/guide.md", "# Guide\nnew\n")
        await service.ingest()
        chunks = await content_store.list_chunks(source_id=source.source_id)
        return first_chunks, chunks

    first_chunks, chunks = asyncio.run(scenario())
    stale_ids = {chunk.chunk_id for chunk in chunks if chunk.status is ContentChunkStatus.STALE}
    active_ids = {chunk.chunk_id for chunk in chunks if chunk.status is ContentChunkStatus.ACTIVE}
    assert {chunk.chunk_id for chunk in first_chunks}.issubset(stale_ids)
    assert active_ids


def test_reingestion_reactivates_chunks_when_content_reverts(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        first_chunks = await content_store.list_chunks(source_id=source.source_id)
        _write(tmp_path, "docs/guide.md", "# Guide\nbeta\n")
        await service.ingest()
        _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
        revert_result = await service.ingest()
        chunks = await content_store.list_chunks(source_id=source.source_id)
        return revert_result, first_chunks, chunks

    result, first_chunks, chunks = asyncio.run(scenario())
    assert result.created_chunks == 0
    first_ids = {chunk.chunk_id for chunk in first_chunks}
    active_ids = {chunk.chunk_id for chunk in chunks if chunk.status is ContentChunkStatus.ACTIVE}
    assert first_ids <= active_ids


def test_reingestion_recovers_after_failed_atomic_sync(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
    normal_service = _service(tmp_path, content_store)
    failing_service = ProjectDocsIngestionService(
        store=_FailAtomicSyncStore(),  # type: ignore[arg-type]
        scanner=ProjectDocsSourceScanner(project_root=tmp_path),
        chunker=MarkdownChunker(max_chars=160),
    )

    async def scenario():
        await normal_service.ingest()
        _write(tmp_path, "docs/guide.md", "# Guide\nbeta\n")
        with pytest.raises(RuntimeError, match="simulated atomic sync failure"):
            await failing_service.ingest()
        await normal_service.ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        chunks = await content_store.list_chunks(source_id=source.source_id)
        return source, chunks

    source, chunks = asyncio.run(scenario())
    active_chunks = [chunk for chunk in chunks if chunk.status is ContentChunkStatus.ACTIVE]
    assert active_chunks
    assert all(chunk.content_hash == source.content_hash for chunk in active_chunks)
    assert all("beta" in chunk.content for chunk in active_chunks)


def test_failed_source_chunk_sync_does_not_publish_new_source_hash(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
    normal_service = _service(tmp_path, content_store)

    async def scenario():
        await normal_service.ingest()
        before = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert before is not None
        _write(tmp_path, "docs/guide.md", "# Guide\nbeta\n")
        failing_service = ProjectDocsIngestionService(
            store=_FailAtomicSyncStore(),  # type: ignore[arg-type]
            scanner=ProjectDocsSourceScanner(project_root=tmp_path),
            chunker=MarkdownChunker(max_chars=160),
        )
        with pytest.raises(RuntimeError, match="simulated atomic sync failure"):
            await failing_service.ingest()
        after = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert after is not None
        chunks = await content_store.list_chunks(source_id=before.source_id)
        return before, after, chunks

    before, after, chunks = asyncio.run(scenario())
    assert after.content_hash == before.content_hash
    assert all(chunk.content_hash == before.content_hash for chunk in chunks)


def test_real_source_chunk_sync_rolls_back_after_chunk_failure(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
    normal_service = _service(tmp_path, content_store)

    async def scenario():
        await normal_service.ingest()
        before = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert before is not None
        candidate = ProjectDocsSourceCandidate(
            source_id=before.source_id,
            relative_path=Path("docs/guide.md"),
            absolute_path=tmp_path / "docs" / "guide.md",
            source_type=ContentSourceType.PROJECT_DOC,
            title="Guide",
            content="# Guide\nbeta\n",
            content_hash="sha256:beta",
        )
        invalid_chunk = ContentChunk(
            chunk_id="106b9b4d-ce4a-55db-830d-1917a8755f45",
            source_id=before.source_id,
            ordinal=0,
            source_path=Path("docs/guide.md"),
            source_type=ContentSourceType.PROJECT_DOC,
            heading_path=["Guide"],
            content=None,  # type: ignore[arg-type]
            content_hash="sha256:beta",
            line_start=1,
            line_end=2,
            citation=ContentCitation(path=Path("docs/guide.md"), line_start=1, line_end=2, heading_path=["Guide"]),
            sensitivity=Sensitivity.PROJECT,
            status=ContentChunkStatus.ACTIVE,
            metadata={},
        )
        with pytest.raises(IntegrityError):
            await content_store.sync_source_chunks(candidate, [invalid_chunk])
        after = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert after is not None
        chunks = await content_store.list_chunks(source_id=before.source_id)
        return before, after, chunks

    before, after, chunks = asyncio.run(scenario())
    assert after.content_hash == before.content_hash
    assert all(chunk.content_hash == before.content_hash for chunk in chunks)
    assert {chunk.status for chunk in chunks} == {ContentChunkStatus.ACTIVE}


def test_source_chunk_sync_rejects_cross_source_chunks(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
    _write(tmp_path, "docs/other.md", "# Other\nstable\n")
    normal_service = _service(tmp_path, content_store)

    async def scenario():
        await normal_service.ingest()
        guide = await content_store.get_source_by_path(Path("docs/guide.md"))
        other = await content_store.get_source_by_path(Path("docs/other.md"))
        assert guide is not None
        assert other is not None
        candidate = ProjectDocsSourceCandidate(
            source_id=guide.source_id,
            relative_path=Path("docs/guide.md"),
            absolute_path=tmp_path / "docs" / "guide.md",
            source_type=ContentSourceType.PROJECT_DOC,
            title="Guide",
            content="# Guide\nbeta\n",
            content_hash="sha256:beta",
        )
        cross_source_chunk = ContentChunk(
            chunk_id="a693a9af-f571-551e-9eec-0f4fb26ed2e0",
            source_id=other.source_id,
            ordinal=0,
            source_path=Path("docs/guide.md"),
            source_type=ContentSourceType.PROJECT_DOC,
            heading_path=["Guide"],
            content="beta",
            content_hash="sha256:beta",
            line_start=1,
            line_end=2,
            citation=ContentCitation(path=Path("docs/guide.md"), line_start=1, line_end=2, heading_path=["Guide"]),
            sensitivity=Sensitivity.PROJECT,
            status=ContentChunkStatus.ACTIVE,
            metadata={},
        )
        with pytest.raises(ValueError, match="chunk source_id must match source"):
            await content_store.sync_source_chunks(candidate, [cross_source_chunk])
        guide_after = await content_store.get_source_by_path(Path("docs/guide.md"))
        other_chunks = await content_store.list_chunks(source_id=other.source_id)
        return guide, guide_after, other_chunks

    guide, guide_after, other_chunks = asyncio.run(scenario())
    assert guide_after is not None
    assert guide_after.content_hash == guide.content_hash
    assert all(chunk.source_path == Path("docs/other.md") for chunk in other_chunks)


def test_source_chunk_sync_rejects_chunks_with_mismatched_source_metadata(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")
    normal_service = _service(tmp_path, content_store)

    async def scenario():
        await normal_service.ingest()
        guide = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert guide is not None
        candidate = ProjectDocsSourceCandidate(
            source_id=guide.source_id,
            relative_path=Path("docs/guide.md"),
            absolute_path=tmp_path / "docs" / "guide.md",
            source_type=ContentSourceType.PROJECT_DOC,
            title="Guide",
            content="# Guide\nbeta\n",
            content_hash="sha256:beta",
        )
        mismatched_chunk = ContentChunk(
            chunk_id="74d710ca-46e1-5f23-9063-70d512650459",
            source_id=guide.source_id,
            ordinal=0,
            source_path=Path("docs/other.md"),
            source_type=ContentSourceType.ADR,
            heading_path=["Guide"],
            content="beta",
            content_hash="sha256:other",
            line_start=1,
            line_end=2,
            citation=ContentCitation(path=Path("docs/other.md"), line_start=1, line_end=2, heading_path=["Guide"]),
            sensitivity=Sensitivity.PROJECT,
            status=ContentChunkStatus.ACTIVE,
            metadata={},
        )
        with pytest.raises(ValueError, match="chunk source metadata must match source"):
            await content_store.sync_source_chunks(candidate, [mismatched_chunk])
        downgraded_chunk = ContentChunk(
            chunk_id="430ca6de-7ca1-5813-8f8b-dc341595ce31",
            source_id=guide.source_id,
            ordinal=0,
            source_path=Path("docs/guide.md"),
            source_type=ContentSourceType.PROJECT_DOC,
            heading_path=["Guide"],
            content="beta",
            content_hash="sha256:beta",
            line_start=1,
            line_end=2,
            citation=ContentCitation(path=Path("docs/guide.md"), line_start=1, line_end=2, heading_path=["Guide"]),
            sensitivity=Sensitivity.PUBLIC,
            status=ContentChunkStatus.ACTIVE,
            metadata={},
        )
        with pytest.raises(ValueError, match="chunk source metadata must match source"):
            await content_store.sync_source_chunks(candidate, [downgraded_chunk])
        guide_after = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert guide_after is not None
        chunks = await content_store.list_chunks(source_id=guide.source_id)
        return guide, guide_after, chunks

    guide, guide_after, chunks = asyncio.run(scenario())
    assert guide_after.content_hash == guide.content_hash
    assert all(chunk.source_path == Path("docs/guide.md") for chunk in chunks)
    assert all(chunk.content_hash == guide.content_hash for chunk in chunks)


def test_source_chunk_sync_rejects_secret_sensitivity_content(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nalpha\n")

    async def scenario():
        candidate = ProjectDocsSourceCandidate(
            source_id="b83c9833-14e3-5275-8741-4171e3fbb343",
            relative_path=Path("docs/guide.md"),
            absolute_path=tmp_path / "docs" / "guide.md",
            source_type=ContentSourceType.PROJECT_DOC,
            title="Guide",
            content="# Guide\nsecret\n",
            content_hash="sha256:secret",
            sensitivity=Sensitivity.SECRET,
        )
        secret_chunk = ContentChunk(
            chunk_id="fdc75c55-5f43-50b5-8ae5-d09e68417167",
            source_id=candidate.source_id,
            ordinal=0,
            source_path=Path("docs/guide.md"),
            source_type=ContentSourceType.PROJECT_DOC,
            heading_path=["Guide"],
            content="secret",
            content_hash="sha256:secret",
            line_start=1,
            line_end=2,
            citation=ContentCitation(path=Path("docs/guide.md"), line_start=1, line_end=2, heading_path=["Guide"]),
            sensitivity=Sensitivity.SECRET,
            status=ContentChunkStatus.ACTIVE,
            metadata={},
        )
        with pytest.raises(ValueError, match="secret content must not be indexed"):
            await content_store.sync_source_chunks(candidate, [secret_chunk])
        return await content_store.get_source_by_path(Path("docs/guide.md"))

    assert asyncio.run(scenario()) is None


def test_unchanged_reingestion_does_not_churn_chunks(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nstable\n")
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        before = await _chunk_created_at_rows(content_store, source.source_id)
        result = await service.ingest()
        after = await _chunk_created_at_rows(content_store, source.source_id)
        return result, before, after

    result, before, after = asyncio.run(scenario())
    assert result.created_chunks == 0
    assert result.stale_chunks == 0
    assert after == before


def test_unchanged_reingestion_refreshes_source_last_seen(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\nstable\n")
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        await _set_source_last_seen(
            content_store,
            Path("docs/guide.md"),
            datetime(2000, 1, 1, tzinfo=UTC),
        )
        before = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert before is not None
        await service.ingest()
        after = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert after is not None
        return before, after

    before, after = asyncio.run(scenario())
    assert after.last_seen_at > before.last_seen_at
    assert after.content_hash == before.content_hash


def test_deleted_source_resurrection_reactivates_existing_chunks(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    content = "# Guide\ncontent\n"
    _write(tmp_path, "docs/guide.md", content)
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        first_chunks = await content_store.list_chunks(source_id=source.source_id)
        (tmp_path / "docs" / "guide.md").unlink()
        await service.ingest()
        _write(tmp_path, "docs/guide.md", content)
        restore_result = await service.ingest()
        source_after = await content_store.get_source_by_path(Path("docs/guide.md"))
        chunks = await content_store.list_chunks(source_id=source.source_id)
        return restore_result, source_after, first_chunks, chunks

    result, source_after, first_chunks, chunks = asyncio.run(scenario())
    assert result.created_chunks == 0
    assert source_after is not None
    assert source_after.status is ContentSourceStatus.ACTIVE
    first_ids = {chunk.chunk_id for chunk in first_chunks}
    active_ids = {chunk.chunk_id for chunk in chunks if chunk.status is ContentChunkStatus.ACTIVE}
    assert first_ids <= active_ids


def test_deleted_source_marks_source_and_chunks_deleted_or_stale(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\ncontent\n")
    service = _service(tmp_path, content_store)

    async def scenario():
        await service.ingest()
        (tmp_path / "docs" / "guide.md").unlink()
        await service.ingest()
        source = await content_store.get_source_by_path(Path("docs/guide.md"))
        assert source is not None
        chunks = await content_store.list_chunks(source_id=source.source_id)
        return source, chunks

    source, chunks = asyncio.run(scenario())
    assert source.status is ContentSourceStatus.DELETED
    assert chunks
    assert {chunk.status for chunk in chunks} <= {
        ContentChunkStatus.DELETED,
        ContentChunkStatus.STALE,
    }


def test_project_docs_delete_pass_does_not_delete_other_content_corpora(
    tmp_path: Path,
    content_store: PostgresContentStore,
) -> None:
    _write(tmp_path, "docs/guide.md", "# Guide\ncontent\n")
    other_source_id = "f3b9bc35-344e-5b9e-ad1d-27f536d4fc8f"

    async def scenario():
        await _service(tmp_path, content_store).ingest()
        async with content_store.engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    insert into content_sources (
                      source_id, source_type, path, uri, title, content_hash,
                      last_seen_at, indexed_at, status, sensitivity, metadata
                    )
                    values (
                      cast(:source_id as uuid), 'external_doc', 'external/manual.md',
                      'external/manual.md', 'Manual', 'sha256:external',
                      now(), now(), 'active', 'project', '{}'
                    )
                    """,
                ),
                {"source_id": other_source_id},
            )
        await _service(tmp_path, content_store).ingest()
        async with content_store.engine.connect() as connection:
            return await connection.scalar(
                text(
                    """
                    select status
                    from content_sources
                    where source_id = cast(:source_id as uuid)
                    """,
                ),
                {"source_id": other_source_id},
            )

    assert asyncio.run(scenario()) == "active"


def test_content_tables_are_separate_from_memory_tables() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    assert asyncio.run(_scalar(database_url, "select to_regclass('public.content_sources')")) == (
        "content_sources"
    )
    assert asyncio.run(_scalar(database_url, "select to_regclass('public.content_chunks')")) == (
        "content_chunks"
    )
    memory_content_columns = asyncio.run(
        _scalar(
            database_url,
            """
            select count(*)
            from information_schema.columns
            where table_name in ('memories', 'memory_embeddings')
            and column_name in ('chunk_id', 'source_id', 'heading_path', 'citation')
            """,
        ),
    )
    assert memory_content_columns == 0
