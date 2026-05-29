from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncEngine

from assistant_core.content_retrieval.project_docs import ProjectDocsSourceCandidate
from assistant_core.domain.content_retrieval import (
    ContentChunk,
    ContentChunkStatus,
    ContentCitation,
    ContentSource,
    ContentSourceStatus,
    ContentSourceSyncResult,
    ContentSourceType,
)
from assistant_core.domain.sensitivity import Sensitivity


_metadata = sa.MetaData()

_content_sources = sa.Table(
    "content_sources",
    _metadata,
    sa.Column("source_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("source_type", sa.Text(), nullable=False),
    sa.Column("path", sa.Text(), nullable=False, unique=True),
    sa.Column("uri", sa.Text(), nullable=False),
    sa.Column("title", sa.Text(), nullable=False),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("sensitivity", sa.Text(), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)

_content_chunks = sa.Table(
    "content_chunks",
    _metadata,
    sa.Column("chunk_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column(
        "source_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("content_sources.source_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("ordinal", sa.Integer(), nullable=False),
    sa.Column("source_path", sa.Text(), nullable=False),
    sa.Column("source_type", sa.Text(), nullable=False),
    sa.Column("heading_path", postgresql.ARRAY(sa.Text()), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("line_start", sa.Integer(), nullable=False),
    sa.Column("line_end", sa.Integer(), nullable=False),
    sa.Column("citation", sa.Text(), nullable=False),
    sa.Column("sensitivity", sa.Text(), nullable=False),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)

_PROJECT_DOC_SOURCE_TYPES = {
    ContentSourceType.README.value,
    ContentSourceType.PROJECT_DOC.value,
    ContentSourceType.ADR.value,
}


class PostgresContentStore:
    def __init__(self, *, engine: AsyncEngine) -> None:
        self.engine = engine

    async def health_check(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                row = (
                    await connection.execute(
                        sa.text(
                            "select "
                            "to_regclass('public.content_sources'), "
                            "to_regclass('public.content_chunks'), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'content_chunks_source_id_fkey'"
                            "), "
                            "exists ("
                            "select 1 from pg_class c "
                            "join pg_index i on i.indexrelid = c.oid "
                            "where c.relname = 'content_sources_path_key' "
                            "and i.indisunique"
                            ")",
                        ),
                    )
                ).one()
        except Exception:
            return False
        return all(value is not None for value in row[:2]) and all(bool(value) for value in row[2:])

    async def get_source_by_path(self, path: Path) -> ContentSource | None:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(_content_sources).where(_content_sources.c.path == path.as_posix()),
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        return _row_to_source(row)

    async def list_sources(
        self,
        *,
        statuses: Iterable[ContentSourceStatus] | None = None,
    ) -> list[ContentSource]:
        query = sa.select(_content_sources).order_by(_content_sources.c.path)
        if statuses is not None:
            query = query.where(_content_sources.c.status.in_([status.value for status in statuses]))
        async with self.engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [_row_to_source(row) for row in rows]

    async def sync_source_chunks(
        self,
        candidate: ProjectDocsSourceCandidate,
        chunks: list[ContentChunk],
    ) -> ContentSourceSyncResult:
        if candidate.sensitivity is Sensitivity.SECRET:
            raise ValueError("secret content must not be indexed")
        now = _now()
        async with self.engine.begin() as connection:
            existing = (
                await connection.execute(
                    sa.select(_content_sources)
                    .where(_content_sources.c.path == candidate.relative_path.as_posix())
                    .with_for_update(),
                )
            ).mappings().one_or_none()
            source_id = _uuid(existing["source_id"]) if existing is not None else _uuid(candidate.source_id)
            _ensure_chunks_match_source(candidate, source_id, chunks)
            incoming_chunk_ids = {_uuid(chunk.chunk_id) for chunk in chunks}
            existing_chunk_ids = await _existing_chunk_ids(connection, source_id, incoming_chunk_ids)
            if existing is not None and _source_is_current(existing, candidate):
                active_chunk_ids = {
                    row[0]
                    for row in (
                        await connection.execute(
                            sa.select(_content_chunks.c.chunk_id)
                            .where(_content_chunks.c.source_id == source_id)
                            .where(_content_chunks.c.status == ContentChunkStatus.ACTIVE.value)
                            .where(_content_chunks.c.content_hash == candidate.content_hash),
                        )
                    ).all()
                }
                if active_chunk_ids == incoming_chunk_ids:
                    await connection.execute(
                        sa.update(_content_sources)
                        .where(_content_sources.c.source_id == source_id)
                        .values(last_seen_at=now),
                    )
                    return ContentSourceSyncResult()

            values = _source_values(candidate, now=now, source_id=source_id)
            insert_source = postgresql.insert(_content_sources).values(values)
            await connection.execute(
                insert_source.on_conflict_do_update(
                    index_elements=[_content_sources.c.path],
                    set_={
                        "source_type": insert_source.excluded.source_type,
                        "uri": insert_source.excluded.uri,
                        "title": insert_source.excluded.title,
                        "content_hash": insert_source.excluded.content_hash,
                        "last_seen_at": insert_source.excluded.last_seen_at,
                        "indexed_at": insert_source.excluded.indexed_at,
                        "status": insert_source.excluded.status,
                        "sensitivity": insert_source.excluded.sensitivity,
                        "metadata": insert_source.excluded.metadata,
                    },
                ),
            )
            stale_result = await connection.execute(
                sa.update(_content_chunks)
                .where(_content_chunks.c.source_id == source_id)
                .where(_content_chunks.c.status == ContentChunkStatus.ACTIVE.value)
                .values(status=ContentChunkStatus.STALE.value),
            )
            if chunks:
                insert_chunks = postgresql.insert(_content_chunks).values(
                    [_chunk_values(chunk, now=now) for chunk in chunks],
                )
                await connection.execute(
                    insert_chunks.on_conflict_do_update(
                        index_elements=[_content_chunks.c.chunk_id],
                        set_={
                            "source_id": insert_chunks.excluded.source_id,
                            "ordinal": insert_chunks.excluded.ordinal,
                            "source_path": insert_chunks.excluded.source_path,
                            "source_type": insert_chunks.excluded.source_type,
                            "heading_path": insert_chunks.excluded.heading_path,
                            "content": insert_chunks.excluded.content,
                            "content_hash": insert_chunks.excluded.content_hash,
                            "line_start": insert_chunks.excluded.line_start,
                            "line_end": insert_chunks.excluded.line_end,
                            "citation": insert_chunks.excluded.citation,
                            "sensitivity": insert_chunks.excluded.sensitivity,
                            "status": ContentChunkStatus.ACTIVE.value,
                            "created_at": insert_chunks.excluded.created_at,
                            "metadata": insert_chunks.excluded.metadata,
                        },
                    ),
                )
        return ContentSourceSyncResult(
            created_source=existing is None,
            updated_source=existing is not None,
            created_chunks=len(incoming_chunk_ids - existing_chunk_ids),
            stale_chunks=int(stale_result.rowcount or 0),
        )

    async def list_chunks(
        self,
        *,
        source_id: str | None = None,
        statuses: Iterable[ContentChunkStatus] | None = None,
    ) -> list[ContentChunk]:
        query = sa.select(_content_chunks).order_by(
            _content_chunks.c.source_path,
            _content_chunks.c.ordinal,
            _content_chunks.c.created_at,
        )
        if source_id is not None:
            query = query.where(_content_chunks.c.source_id == _uuid(source_id))
        if statuses is not None:
            query = query.where(_content_chunks.c.status.in_([status.value for status in statuses]))
        async with self.engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [_row_to_chunk(row) for row in rows]

    async def mark_missing_sources_deleted(self, seen_paths: set[Path]) -> tuple[int, int]:
        seen = {path.as_posix() for path in seen_paths}
        source_query = sa.select(_content_sources.c.source_id).where(
            _content_sources.c.status != ContentSourceStatus.DELETED.value,
        ).where(
            _content_sources.c.source_type.in_(_PROJECT_DOC_SOURCE_TYPES),
        )
        if seen:
            source_query = source_query.where(_content_sources.c.path.not_in(seen))

        async with self.engine.begin() as connection:
            source_ids = [row[0] for row in (await connection.execute(source_query)).all()]
            if not source_ids:
                return 0, 0
            deleted_sources = await connection.execute(
                sa.update(_content_sources)
                .where(_content_sources.c.source_id.in_(source_ids))
                .values(status=ContentSourceStatus.DELETED.value),
            )
            deleted_chunks = await connection.execute(
                sa.update(_content_chunks)
                .where(_content_chunks.c.source_id.in_(source_ids))
                .where(_content_chunks.c.status != ContentChunkStatus.DELETED.value)
                .values(status=ContentChunkStatus.DELETED.value),
            )
        return int(deleted_sources.rowcount or 0), int(deleted_chunks.rowcount or 0)


def _uuid(value: str | UUID) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(value)


def _now() -> datetime:
    return datetime.now(UTC)


def _ensure_chunks_match_source(
    candidate: ProjectDocsSourceCandidate,
    source_id: UUID,
    chunks: list[ContentChunk],
) -> None:
    if any(_uuid(chunk.source_id) != source_id for chunk in chunks):
        raise ValueError("chunk source_id must match source")
    if any(
        chunk.source_path != candidate.relative_path
        or chunk.source_type is not candidate.source_type
        or chunk.content_hash != candidate.content_hash
        or chunk.citation.path != candidate.relative_path
        or chunk.sensitivity is not candidate.sensitivity
        for chunk in chunks
    ):
        raise ValueError("chunk source metadata must match source")


def _source_is_current(row: Any, candidate: ProjectDocsSourceCandidate) -> bool:
    return (
        row["source_type"] == candidate.source_type.value
        and row["content_hash"] == candidate.content_hash
        and row["status"] == ContentSourceStatus.ACTIVE.value
        and row["sensitivity"] == candidate.sensitivity.value
    )


def _source_values(
    candidate: ProjectDocsSourceCandidate,
    *,
    now: datetime,
    source_id: UUID,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": candidate.source_type.value,
        "path": candidate.relative_path.as_posix(),
        "uri": candidate.relative_path.as_posix(),
        "title": candidate.title,
        "content_hash": candidate.content_hash,
        "last_seen_at": now,
        "indexed_at": now,
        "status": ContentSourceStatus.ACTIVE.value,
        "sensitivity": candidate.sensitivity.value,
        "metadata": {},
    }


def _chunk_values(chunk: ContentChunk, *, now: datetime) -> dict[str, Any]:
    return {
        "chunk_id": _uuid(chunk.chunk_id),
        "source_id": _uuid(chunk.source_id),
        "ordinal": chunk.ordinal,
        "source_path": chunk.source_path.as_posix(),
        "source_type": chunk.source_type.value,
        "heading_path": chunk.heading_path,
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "citation": chunk.citation.format(),
        "sensitivity": chunk.sensitivity.value,
        "status": chunk.status.value,
        "created_at": now,
        "metadata": chunk.metadata,
    }


async def _existing_chunk_ids(
    connection: Any,
    source_id: UUID,
    chunk_ids: set[UUID],
) -> set[UUID]:
    if not chunk_ids:
        return set()
    return {
        row[0]
        for row in (
            await connection.execute(
                sa.select(_content_chunks.c.chunk_id)
                .where(_content_chunks.c.source_id == source_id)
                .where(_content_chunks.c.chunk_id.in_(chunk_ids)),
            )
        ).all()
    }


def _row_to_source(row: Any) -> ContentSource:
    return ContentSource(
        source_id=str(row["source_id"]),
        source_type=ContentSourceType(row["source_type"]),
        path=Path(row["path"]),
        uri=row["uri"],
        title=row["title"],
        content_hash=row["content_hash"],
        last_seen_at=row["last_seen_at"],
        indexed_at=row["indexed_at"],
        status=ContentSourceStatus(row["status"]),
        sensitivity=Sensitivity(row["sensitivity"]),
        metadata=dict(row["metadata"]),
    )


def _row_to_chunk(row: Any) -> ContentChunk:
    source_path = Path(row["source_path"])
    return ContentChunk(
        chunk_id=str(row["chunk_id"]),
        source_id=str(row["source_id"]),
        ordinal=row["ordinal"],
        source_path=source_path,
        source_type=ContentSourceType(row["source_type"]),
        heading_path=list(row["heading_path"]),
        content=row["content"],
        content_hash=row["content_hash"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        citation=ContentCitation(
            path=source_path,
            line_start=row["line_start"],
            line_end=row["line_end"],
            heading_path=list(row["heading_path"]),
        ),
        sensitivity=Sensitivity(row["sensitivity"]),
        status=ContentChunkStatus(row["status"]),
        metadata=dict(row["metadata"]),
    )
