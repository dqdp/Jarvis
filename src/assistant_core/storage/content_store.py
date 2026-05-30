from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from assistant_core.config.settings import Settings
from assistant_core.content_retrieval.project_docs import ProjectDocsSourceCandidate
from assistant_core.domain.content_retrieval import (
    ContentChunk,
    ContentChunkStatus,
    ContentCitation,
    ContentEmbeddingRecord,
    ContentEmbeddingStatus,
    ContentHit,
    ContentRetrievalQuery,
    ContentSource,
    ContentSourceStatus,
    ContentSourceSyncResult,
    ContentSourceType,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.embedding import EmbeddingPort, GenerateEmbeddingCommand
from assistant_core.storage.event_log import insert_event


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

_content_embeddings = sa.Table(
    "content_embeddings",
    _metadata,
    sa.Column(
        "chunk_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("content_chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("embedding_profile", sa.Text(), primary_key=True),
    sa.Column("embedding_model", sa.Text(), nullable=False),
    sa.Column("embedding_dimension", sa.Integer(), nullable=False),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=True),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("error_type", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)

_PROJECT_DOC_SOURCE_TYPES = {
    ContentSourceType.README.value,
    ContentSourceType.PROJECT_DOC.value,
    ContentSourceType.ADR.value,
}


class PostgresContentStore:
    def __init__(
        self,
        *,
        engine: AsyncEngine,
        embedding_port: EmbeddingPort | None = None,
        settings: Settings | None = None,
        embedding_profile: str = "local_embedding",
    ) -> None:
        self.engine = engine
        self._embedding_port = embedding_port
        self._settings = settings
        self._embedding_profile = embedding_profile

    async def health_check(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                row = (
                    await connection.execute(
                        sa.text(
                            "select "
                            "to_regclass('public.content_sources'), "
                            "to_regclass('public.content_chunks'), "
                            "to_regclass('public.content_embeddings'), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'content_chunks_source_id_fkey'"
                            "), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'content_embeddings_pkey'"
                            "), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'content_embeddings_chunk_id_fkey'"
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
        return all(value is not None for value in row[:3]) and all(bool(value) for value in row[3:])

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
        sync_result: ContentSourceSyncResult | None = None
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
                    sync_result = ContentSourceSyncResult()

            if sync_result is None:
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
                sync_result = ContentSourceSyncResult(
                    created_source=existing is None,
                    updated_source=existing is not None,
                    created_chunks=len(incoming_chunk_ids - existing_chunk_ids),
                    stale_chunks=int(stale_result.rowcount or 0),
                )
        await self._sync_embeddings(chunks)
        return sync_result

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

    async def get_current_embedding(
        self,
        chunk_id: str,
        embedding_profile: str,
    ) -> ContentEmbeddingRecord | None:
        statement = (
            sa.select(_content_embeddings)
            .join(_content_chunks, _content_embeddings.c.chunk_id == _content_chunks.c.chunk_id)
            .where(
                _content_embeddings.c.chunk_id == _uuid(chunk_id),
                _content_embeddings.c.embedding_profile == embedding_profile,
                _content_embeddings.c.content_hash == _content_chunks.c.content_hash,
                _content_embeddings.c.status == ContentEmbeddingStatus.INDEXED.value,
            )
        )
        async with self.engine.connect() as connection:
            row = (await connection.execute(statement)).mappings().first()
        if row is None:
            return None
        return _row_to_embedding(row)

    async def retrieve(self, query: ContentRetrievalQuery) -> list[ContentHit]:
        if not query.text.strip():
            await self._record_content_retrieval_failed("empty_query", query=query)
            return []
        if self._embedding_port is None:
            await self._record_content_retrieval_failed("embedding_port_missing", query=query)
            return []
        source_types = (
            [_source_type(value).value for value in query.source_types]
            if query.source_types is not None
            else None
        )
        excluded_sensitivities = {_sensitivity(value) for value in query.exclude_sensitivities}
        excluded_sensitivities.add(Sensitivity.SECRET)
        excluded_sensitivity_values = [value.value for value in excluded_sensitivities]
        conditions = [
            _content_sources.c.status == ContentSourceStatus.ACTIVE.value,
            _content_chunks.c.status == ContentChunkStatus.ACTIVE.value,
            _content_embeddings.c.embedding_profile == self._embedding_profile,
            _content_embeddings.c.embedding_model == self._embedding_model(),
            _content_embeddings.c.status == ContentEmbeddingStatus.INDEXED.value,
            _content_embeddings.c.content_hash == _content_chunks.c.content_hash,
            _content_embeddings.c.embedding.is_not(None),
        ]
        expected_dimension = self._embedding_dimension()
        if expected_dimension is not None:
            conditions.append(_content_embeddings.c.embedding_dimension == expected_dimension)
        if source_types:
            conditions.append(_content_chunks.c.source_type.in_(source_types))
        if excluded_sensitivity_values:
            conditions.append(_content_chunks.c.sensitivity.notin_(excluded_sensitivity_values))
            conditions.append(_content_sources.c.sensitivity.notin_(excluded_sensitivity_values))
        statement = (
            sa.select(
                _content_chunks,
                _content_sources.c.title.label("title"),
                _content_embeddings.c.embedding.label("embedding"),
            )
            .join(_content_sources, _content_sources.c.source_id == _content_chunks.c.source_id)
            .join(_content_embeddings, _content_embeddings.c.chunk_id == _content_chunks.c.chunk_id)
            .where(*conditions)
        )
        async with self.engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        if not rows:
            await self._record_content_retrieved([], query=query)
            return []

        try:
            response = await self._embedding_port.embed(
                GenerateEmbeddingCommand(texts=[query.text], sensitivity=query.sensitivity),
            )
        except Exception as exc:
            await self._record_content_retrieval_failed(
                "query_embedding_failed",
                query=query,
                error_type=type(exc).__name__,
            )
            return []
        query_vector = response.vectors[0]
        hits = [
            _row_to_hit(row, score=_cosine_similarity(query_vector, row["embedding"]))
            for row in rows
        ]
        hits = [hit for hit in hits if hit.score > 0]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        selected = hits[: (query.limit or 8)]
        await self._record_content_retrieved(selected, query=query)
        return selected

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

    async def _sync_embeddings(self, chunks: list[ContentChunk]) -> None:
        if self._embedding_port is None:
            return
        for chunk in chunks:
            if chunk.status is not ContentChunkStatus.ACTIVE or chunk.sensitivity is Sensitivity.SECRET:
                continue
            current = await self.get_current_embedding(chunk.chunk_id, self._embedding_profile)
            if current is not None and self._embedding_is_current(current, chunk):
                continue
            try:
                response = await self._embedding_port.embed(
                    GenerateEmbeddingCommand(texts=[chunk.content], sensitivity=chunk.sensitivity),
                )
                vector = response.vectors[0]
            except Exception as exc:
                await self._mark_embedding_failed(chunk, exc)
                continue
            await self._upsert_embedding(chunk, vector)

    async def _upsert_embedding(self, chunk: ContentChunk, vector: list[float]) -> None:
        async with self.engine.begin() as connection:
            current = await _select_chunk(connection, chunk.chunk_id, for_update=True)
            if (
                current.content_hash != chunk.content_hash
                or current.status is not ContentChunkStatus.ACTIVE
            ):
                return
            embedding = await _upsert_embedding(
                connection,
                current,
                embedding_profile=self._embedding_profile,
                embedding_model=self._embedding_model(),
                vector=vector,
                status=ContentEmbeddingStatus.INDEXED,
                error_type=None,
            )
            await insert_event(
                connection,
                _content_embedding_event(
                    EventType.CONTENT_EMBEDDING_CREATED,
                    current,
                    payload_extra={
                        "embedding_profile": embedding.embedding_profile,
                        "embedding_dimension": embedding.embedding_dimension,
                    },
                ),
            )

    async def _mark_embedding_failed(self, chunk: ContentChunk, exc: Exception) -> None:
        async with self.engine.begin() as connection:
            current = await _select_chunk(connection, chunk.chunk_id, for_update=True)
            if (
                current.content_hash != chunk.content_hash
                or current.status is not ContentChunkStatus.ACTIVE
            ):
                return
            await _upsert_embedding(
                connection,
                current,
                embedding_profile=self._embedding_profile,
                embedding_model=self._embedding_model(),
                vector=[],
                status=ContentEmbeddingStatus.FAILED,
                error_type=type(exc).__name__,
            )
            await insert_event(
                connection,
                _content_embedding_event(
                    EventType.CONTENT_EMBEDDING_FAILED,
                    current,
                    payload_extra={
                        "embedding_profile": self._embedding_profile,
                        "error_type": type(exc).__name__,
                    },
                ),
            )

    async def _record_content_retrieved(
        self,
        hits: list[ContentHit],
        *,
        query: ContentRetrievalQuery,
    ) -> None:
        async with self.engine.begin() as connection:
            await insert_event(
                connection,
                _content_retrieved_event(
                    hits,
                    query=query,
                    embedding_profile=self._embedding_profile,
                ),
            )

    async def _record_content_retrieval_failed(
        self,
        reason: str,
        *,
        query: ContentRetrievalQuery,
        error_type: str | None = None,
    ) -> None:
        async with self.engine.begin() as connection:
            await insert_event(
                connection,
                _content_retrieval_failed_event(
                    reason=reason,
                    query=query,
                    error_type=error_type,
                    embedding_profile=self._embedding_profile,
                ),
            )

    def _embedding_model(self) -> str:
        if self._settings is None:
            return self._embedding_profile
        profile = self._settings.model_profiles.get(self._embedding_profile)
        if profile is None:
            return self._embedding_profile
        return profile.model

    def _embedding_dimension(self) -> int | None:
        if self._settings is None:
            return None
        profile = self._settings.model_profiles.get(self._embedding_profile)
        if profile is None:
            return None
        return profile.dimension

    def _embedding_is_current(
        self,
        embedding: ContentEmbeddingRecord,
        chunk: ContentChunk,
    ) -> bool:
        expected_dimension = self._embedding_dimension()
        return (
            embedding.content_hash == chunk.content_hash
            and embedding.embedding_model == self._embedding_model()
            and (expected_dimension is None or embedding.embedding_dimension == expected_dimension)
        )


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


async def _select_chunk(
    connection: AsyncConnection,
    chunk_id: str,
    *,
    for_update: bool = False,
) -> ContentChunk:
    statement = sa.select(_content_chunks).where(_content_chunks.c.chunk_id == _uuid(chunk_id))
    if for_update:
        statement = statement.with_for_update()
    row = (await connection.execute(statement)).mappings().first()
    if row is None:
        raise KeyError(f"content chunk not found: {chunk_id}")
    return _row_to_chunk(row)


async def _upsert_embedding(
    connection: AsyncConnection,
    chunk: ContentChunk,
    *,
    embedding_profile: str,
    embedding_model: str,
    vector: list[float],
    status: ContentEmbeddingStatus,
    error_type: str | None,
) -> ContentEmbeddingRecord:
    statement = (
        postgresql.insert(_content_embeddings)
        .values(
            {
                "chunk_id": _uuid(chunk.chunk_id),
                "embedding_profile": embedding_profile,
                "embedding_model": embedding_model,
                "embedding_dimension": len(vector),
                "content_hash": chunk.content_hash,
                "embedding": vector if status is ContentEmbeddingStatus.INDEXED else None,
                "status": status.value,
                "error_type": error_type,
                "created_at": _now(),
                "metadata": {},
            },
        )
        .on_conflict_do_update(
            index_elements=[
                _content_embeddings.c.chunk_id,
                _content_embeddings.c.embedding_profile,
            ],
            set_={
                "embedding_model": embedding_model,
                "embedding_dimension": len(vector),
                "content_hash": chunk.content_hash,
                "embedding": vector if status is ContentEmbeddingStatus.INDEXED else None,
                "status": status.value,
                "error_type": error_type,
                "created_at": _now(),
                "metadata": {},
            },
        )
        .returning(*_content_embeddings.c)
    )
    row = (await connection.execute(statement)).mappings().one()
    return _row_to_embedding(row)


def _row_to_embedding(row: Mapping[str, Any]) -> ContentEmbeddingRecord:
    embedding = row["embedding"] or []
    return ContentEmbeddingRecord(
        chunk_id=str(row["chunk_id"]),
        embedding_profile=row["embedding_profile"],
        embedding_model=row["embedding_model"],
        embedding_dimension=row["embedding_dimension"],
        content_hash=row["content_hash"],
        embedding=[float(value) for value in embedding],
        status=ContentEmbeddingStatus(row["status"]),
        created_at=_datetime(row["created_at"]),
        error_type=row["error_type"],
        metadata=dict(row["metadata"]),
    )


def _row_to_hit(row: Mapping[str, Any], *, score: float) -> ContentHit:
    chunk = _row_to_chunk(row)
    return ContentHit(
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        source_type=chunk.source_type,
        source_path=chunk.source_path,
        title=row["title"],
        content=chunk.content,
        score=score,
        citation=chunk.citation,
        sensitivity=chunk.sensitivity,
        content_hash=chunk.content_hash,
        metadata=dict(chunk.metadata),
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    left_values = left[:size]
    right_values = right[:size]
    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = sum(value * value for value in left_values) ** 0.5
    right_norm = sum(value * value for value in right_values) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _source_type(value: ContentSourceType | str) -> ContentSourceType:
    return value if isinstance(value, ContentSourceType) else ContentSourceType(value)


def _sensitivity(value: Sensitivity | str) -> Sensitivity:
    return value if isinstance(value, Sensitivity) else Sensitivity(value)


def _content_embedding_event(
    event_type: EventType,
    chunk: ContentChunk,
    *,
    payload_extra: dict[str, Any] | None = None,
) -> EventEnvelope:
    payload = {
        "source_id": chunk.source_id,
        "chunk_id": chunk.chunk_id,
        "content_hash": chunk.content_hash,
        "citation": chunk.citation.format(),
    }
    if payload_extra:
        payload.update(payload_extra)
    now = _now()
    return EventEnvelope(
        event_id=_new_id(),
        event_seq=0,
        event_type=event_type,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=None,
        request_id=None,
        correlation_id=None,
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="content_store",
        source_node=None,
        sensitivity=chunk.sensitivity,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload=payload,
        metadata={},
    )


def _content_retrieved_event(
    hits: list[ContentHit],
    *,
    query: ContentRetrievalQuery,
    embedding_profile: str,
) -> EventEnvelope:
    now = _now()
    return EventEnvelope(
        event_id=_new_id(),
        event_seq=0,
        event_type=EventType.CONTENT_RETRIEVED,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=query.conversation_id,
        request_id=query.request_id,
        correlation_id=query.correlation_id,
        causation_id=query.causation_id,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="content_store",
        source_node=None,
        sensitivity=_max_retrieval_sensitivity(query.sensitivity, hits),
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={
            "retrieved_content_refs": [
                {
                    "source_id": hit.source_id,
                    "chunk_id": hit.chunk_id,
                    "citation": hit.citation.format(),
                    "score": hit.score,
                    "content_hash": hit.content_hash,
                }
                for hit in hits
            ],
            "embedding_profile": embedding_profile,
            "full_content_stored": False,
        },
        metadata={},
    )


def _content_retrieval_failed_event(
    *,
    reason: str,
    query: ContentRetrievalQuery,
    error_type: str | None,
    embedding_profile: str,
) -> EventEnvelope:
    now = _now()
    payload = {
        "reason": reason,
        "query_hash": _query_hash(query.text),
        "embedding_profile": embedding_profile,
        "full_query_stored": False,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    return EventEnvelope(
        event_id=_new_id(),
        event_seq=0,
        event_type=EventType.CONTENT_RETRIEVAL_FAILED,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=query.conversation_id,
        request_id=query.request_id,
        correlation_id=query.correlation_id,
        causation_id=query.causation_id,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="content_store",
        source_node=None,
        sensitivity=query.sensitivity,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload=payload,
        metadata={},
    )


def _query_hash(text: str) -> str:
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


def _max_retrieval_sensitivity(
    query_sensitivity: Sensitivity,
    hits: list[ContentHit],
) -> Sensitivity:
    order = {
        Sensitivity.PUBLIC: 0,
        Sensitivity.PROJECT: 1,
        Sensitivity.PERSONAL: 2,
        Sensitivity.INFRA: 3,
        Sensitivity.SECRET: 4,
    }
    return max(
        [query_sensitivity, *(hit.sensitivity for hit in hits)],
        key=lambda value: order[value],
    )


def _new_id() -> str:
    return str(uuid4())


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime, got {type(value).__name__}")
