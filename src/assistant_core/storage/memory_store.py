from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from assistant_core.config.settings import Settings
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.memory import (
    ArchiveMemoryCommand,
    CreateMemoryCommand,
    IndexingStatus,
    MemoryEmbeddingRecord,
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    SupersedeMemoryCommand,
    UpdateMemoryCommand,
)
from assistant_core.domain.policy import MemoryWritePolicyRequest
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.embedding import EmbeddingPort, GenerateEmbeddingCommand
from assistant_core.ports.memory import (
    InvalidMemoryType,
    MemoryRetrievalError,
    MemoryPolicyDenied,
    MemoryTypeNotAllowed,
    UnknownMemoryNamespace,
)
from assistant_core.ports.policy import PolicyPort
from assistant_core.storage.event_log import insert_event


_metadata = sa.MetaData()

_memories = sa.Table(
    "memories",
    _metadata,
    sa.Column("memory_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("namespace", sa.Text(), nullable=False),
    sa.Column("memory_type", sa.Text(), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("summary", sa.Text(), nullable=True),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("sensitivity", sa.Text(), nullable=False),
    sa.Column("confidence", sa.Float(), nullable=False),
    sa.Column("importance", sa.Float(), nullable=False),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("indexing_status", sa.Text(), nullable=False),
    sa.Column("source_event_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
    sa.Column("supersedes_memory_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
    sa.Column("superseded_by_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("revision", sa.Integer(), nullable=False),
    sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
    sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
    sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("archive_reason", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)

_memory_embeddings = sa.Table(
    "memory_embeddings",
    _metadata,
    sa.Column("memory_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("embedding_profile", sa.Text(), primary_key=True),
    sa.Column("embedding_model", sa.Text(), nullable=False),
    sa.Column("embedding_dimension", sa.Integer(), nullable=False),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)


class PostgresMemoryStore:
    def __init__(
        self,
        *,
        engine: AsyncEngine,
        settings: Settings,
        policy: PolicyPort,
        embedding_port: EmbeddingPort | None = None,
        embedding_profile: str = "local_embedding",
    ) -> None:
        self.engine = engine
        self._settings = settings
        self._policy = policy
        self._embedding_port = embedding_port
        self._embedding_profile = embedding_profile

    async def health_check(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                row = (
                    await connection.execute(
                        sa.text(
                            "select "
                            "to_regclass('public.memories'), "
                            "to_regclass('public.memory_embeddings'), "
                            "to_regclass('public.memory_candidates'), "
                            "to_regclass('public.events'), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'memory_embeddings_pkey'"
                            "), "
                            "exists ("
                            "select 1 from pg_constraint "
                            "where conname = 'memory_embeddings_memory_id_fkey'"
                            "), "
                            "exists ("
                            "select 1 from pg_class c "
                            "join pg_index i on i.indexrelid = c.oid "
                            "where c.relname = 'memory_embeddings_content_hash_idx'"
                            "), "
                            "exists ("
                            "select 1 from pg_class c "
                            "join pg_index i on i.indexrelid = c.oid "
                            "where c.relname = 'memories_retrieval_filter_idx'"
                            ")",
                        ),
                    )
                ).one()
        except Exception:
            return False
        return all(value is not None for value in row[:4]) and all(bool(value) for value in row[4:])

    async def create_memory(self, command: CreateMemoryCommand) -> MemoryRecord:
        memory_type = self._validate_command(command)
        await self._authorize(command)
        async with self.engine.begin() as connection:
            memory = await _insert_memory(
                connection,
                command,
                memory_type=memory_type,
                supersedes_memory_ids=[],
            )
            await insert_event(connection, _memory_event(EventType.MEMORY_CREATED, memory))
        return await self._sync_embedding(memory)

    async def update_memory(self, command: UpdateMemoryCommand) -> MemoryRecord:
        async with self.engine.begin() as connection:
            current = await self._select_memory(connection, command.memory_id, for_update=True)
            content = command.content if command.content is not None else current.content
            content_hash = _content_hash(content)
            content_changed = content_hash != current.content_hash
            summary = command.summary if command.summary is not None else current.summary
            metadata = command.metadata if command.metadata is not None else current.metadata
            now = _now()
            values = {
                "content": content,
                "summary": summary,
                "content_hash": content_hash,
                "confidence": (
                    command.confidence
                    if command.confidence is not None
                    else current.confidence
                ),
                "importance": (
                    command.importance
                    if command.importance is not None
                    else current.importance
                ),
                "revision": current.revision + 1,
                "updated_at": now,
                "metadata": metadata,
            }
            if content_changed:
                values["indexing_status"] = IndexingStatus.EMBEDDING_PENDING.value
            row = (
                await connection.execute(
                    sa.update(_memories)
                    .where(_memories.c.memory_id == _uuid(command.memory_id))
                    .values(values)
                    .returning(*_memories.c),
                )
            ).mappings().one()
            updated = _row_to_memory(row)
        if not content_changed:
            return updated
        return await self._sync_embedding(updated)

    async def archive_memory(self, command: ArchiveMemoryCommand) -> None:
        async with self.engine.begin() as connection:
            memory = await self._select_memory(connection, command.memory_id, for_update=True)
            if memory.status == MemoryStatus.ARCHIVED:
                return
            now = _now()
            row = (
                await connection.execute(
                    sa.update(_memories)
                    .where(_memories.c.memory_id == _uuid(command.memory_id))
                    .values(
                        {
                            "status": MemoryStatus.ARCHIVED.value,
                            "archived_at": now,
                            "archive_reason": command.reason,
                            "updated_at": now,
                        },
                    )
                    .returning(*_memories.c),
                )
            ).mappings().one()
            archived = _row_to_memory(row)
            await insert_event(
                connection,
                _memory_event(
                    EventType.MEMORY_ARCHIVED,
                    archived,
                    payload_extra={"archive_reason": command.reason},
                ),
            )

    async def supersede_memory(self, command: SupersedeMemoryCommand) -> MemoryRecord:
        memory_type = self._validate_command(command.replacement)
        await self._authorize(command.replacement)
        async with self.engine.begin() as connection:
            old = await self._select_memory(
                connection,
                command.superseded_memory_id,
                for_update=True,
            )
            replacement = await _insert_memory(
                connection,
                command.replacement,
                memory_type=memory_type,
                supersedes_memory_ids=[old.id],
            )
            now = _now()
            await connection.execute(
                sa.update(_memories)
                .where(_memories.c.memory_id == _uuid(old.id))
                .values(
                    {
                        "status": MemoryStatus.SUPERSEDED.value,
                        "superseded_by_memory_id": _uuid(replacement.id),
                        "updated_at": now,
                    },
                ),
            )
            await insert_event(
                connection,
                _memory_event(
                    EventType.MEMORY_SUPERSEDED,
                    replacement,
                    payload_extra={
                        "superseded_memory_id": old.id,
                        "replacement_memory_id": replacement.id,
                    },
                ),
            )
        return await self._sync_embedding(replacement)

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(_memories).where(_memories.c.memory_id == _uuid(memory_id)),
                )
            ).mappings().first()
        if row is None:
            return None
        return _row_to_memory(row)

    async def list_memories(
        self,
        limit: int = 100,
        query: str | None = None,
    ) -> list[MemoryRecord]:
        conditions = [_memories.c.sensitivity != Sensitivity.SECRET.value]
        if query:
            pattern = _like_literal_pattern(query)
            conditions.append(
                sa.or_(
                    sa.func.lower(_memories.c.content).like(pattern, escape="\\"),
                    sa.func.lower(sa.func.coalesce(_memories.c.summary, "")).like(
                        pattern,
                        escape="\\",
                    ),
                ),
            )
        statement = (
            sa.select(_memories)
            .where(*conditions)
            .order_by(_memories.c.created_at, _memories.c.memory_id)
            .limit(limit)
        )
        async with self.engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return [_row_to_memory(row) for row in rows]

    async def retrieve(self, query: MemoryQuery) -> list[MemoryHit]:
        if not query.namespaces:
            return []
        try:
            statuses = [_memory_status(value) for value in query.include_statuses]
            memory_types = (
                [_memory_type(value) for value in query.memory_types]
                if query.memory_types is not None
                else None
            )
            excluded_sensitivities = [
                _sensitivity(value).value
                for value in self._settings.memory.retrieval.exclude_sensitivity
            ]
            conditions = [
                _memories.c.status.in_([status.value for status in statuses]),
                _memories.c.indexing_status == IndexingStatus.INDEXED.value,
                _memory_embeddings.c.embedding_profile == self._embedding_profile,
                _memory_embeddings.c.content_hash == _memories.c.content_hash,
            ]
            if excluded_sensitivities:
                conditions.append(_memories.c.sensitivity.notin_(excluded_sensitivities))
            if query.namespaces:
                conditions.append(_memories.c.namespace.in_(query.namespaces))
            if memory_types:
                conditions.append(
                    _memories.c.memory_type.in_([item.value for item in memory_types]),
                )

            statement = (
                sa.select(_memories)
                .join(_memory_embeddings, _memory_embeddings.c.memory_id == _memories.c.memory_id)
                .where(*conditions)
            )
            async with self.engine.connect() as connection:
                rows = (await connection.execute(statement)).mappings().all()
        except Exception as exc:
            raise MemoryRetrievalError("memory retrieval failed") from exc

        hits = [
            MemoryHit(memory=memory, score=_retrieval_score(query.text, memory))
            for memory in (_row_to_memory(row) for row in rows)
        ]
        min_score = self._settings.memory.retrieval.min_score
        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= min_score]
        hits.sort(
            key=lambda hit: (hit.score, hit.memory.importance, hit.memory.updated_at),
            reverse=True,
        )

        total_limit = min(
            query.limit or self._settings.memory.retrieval.max_hits_total,
            self._settings.memory.retrieval.max_hits_total,
        )
        per_namespace_limit = self._settings.memory.retrieval.max_hits_per_namespace
        selected: list[MemoryHit] = []
        namespace_counts: dict[str, int] = {}
        for hit in hits:
            count = namespace_counts.get(hit.memory.namespace, 0)
            if count >= per_namespace_limit:
                continue
            selected.append(hit)
            namespace_counts[hit.memory.namespace] = count + 1
            if len(selected) >= total_limit:
                break
        return selected

    async def get_current_embedding(
        self,
        memory_id: str,
        embedding_profile: str,
    ) -> MemoryEmbeddingRecord | None:
        statement = (
            sa.select(_memory_embeddings)
            .join(_memories, _memory_embeddings.c.memory_id == _memories.c.memory_id)
            .where(
                _memory_embeddings.c.memory_id == _uuid(memory_id),
                _memory_embeddings.c.embedding_profile == embedding_profile,
                _memory_embeddings.c.content_hash == _memories.c.content_hash,
            )
        )
        async with self.engine.connect() as connection:
            row = (await connection.execute(statement)).mappings().first()
        if row is None:
            return None
        return _row_to_embedding(row)

    async def _sync_embedding(self, memory: MemoryRecord) -> MemoryRecord:
        if self._embedding_port is None:
            return memory

        profile = self._settings.model_profiles[self._embedding_profile]
        try:
            response = await self._embedding_port.embed(
                GenerateEmbeddingCommand(
                    texts=[memory.content],
                    sensitivity=memory.sensitivity,
                ),
            )
            vector = response.vectors[0]
        except Exception as exc:
            return await self._mark_embedding_failed(memory, exc)

        async with self.engine.begin() as connection:
            current = await self._select_memory(connection, memory.id, for_update=True)
            if current.content_hash != memory.content_hash:
                return current
            await _upsert_embedding(
                connection,
                current,
                embedding_profile=self._embedding_profile,
                embedding_model=profile.model,
                vector=vector,
            )
            indexed = await _set_indexing_status(connection, current.id, IndexingStatus.INDEXED)
            await insert_event(
                connection,
                _memory_event(
                    EventType.MEMORY_EMBEDDING_CREATED,
                    indexed,
                    payload_extra={
                        "embedding_profile": self._embedding_profile,
                        "embedding_dimension": len(vector),
                    },
                ),
            )
            return indexed

    async def _mark_embedding_failed(
        self,
        memory: MemoryRecord,
        exc: Exception,
    ) -> MemoryRecord:
        async with self.engine.begin() as connection:
            current = await self._select_memory(connection, memory.id, for_update=True)
            if current.content_hash != memory.content_hash:
                return current
            failed = await _set_indexing_status(
                connection,
                current.id,
                IndexingStatus.EMBEDDING_FAILED,
            )
            await insert_event(
                connection,
                _memory_event(
                    EventType.MEMORY_EMBEDDING_FAILED,
                    failed,
                    payload_extra={
                        "embedding_profile": self._embedding_profile,
                        "error_type": type(exc).__name__,
                    },
                ),
            )
            return failed

    def _validate_command(self, command: CreateMemoryCommand) -> MemoryType:
        namespace = self._settings.memory.namespaces.get(command.namespace)
        if namespace is None:
            raise UnknownMemoryNamespace(f"unknown memory namespace: {command.namespace}")

        try:
            memory_type = (
                command.memory_type
                if isinstance(command.memory_type, MemoryType)
                else MemoryType(command.memory_type)
            )
        except ValueError as exc:
            raise InvalidMemoryType(f"invalid memory type: {command.memory_type}") from exc

        if memory_type.value not in namespace.allowed_types:
            raise MemoryTypeNotAllowed(
                f"{memory_type.value} is not allowed in namespace {command.namespace}",
            )
        return memory_type

    async def _authorize(self, command: CreateMemoryCommand) -> None:
        decision = await self._policy.evaluate_memory_write(
            MemoryWritePolicyRequest(
                namespace=command.namespace,
                sensitivity=command.sensitivity,
            ),
        )
        if decision.allowed:
            async with self.engine.begin() as connection:
                await insert_event(
                    connection,
                    _memory_write_policy_event(
                        command,
                        allowed=True,
                        code=decision.code,
                        reason=decision.reason,
                    ),
                )
            return
        if not decision.allowed:
            async with self.engine.begin() as connection:
                await insert_event(
                    connection,
                    _memory_write_policy_event(
                        command,
                        allowed=False,
                        code=decision.code,
                        reason=decision.reason,
                    ),
                )
            raise MemoryPolicyDenied(decision.reason)

    async def _select_memory(
        self,
        connection: AsyncConnection,
        memory_id: str,
        *,
        for_update: bool = False,
    ) -> MemoryRecord:
        statement = sa.select(_memories).where(_memories.c.memory_id == _uuid(memory_id))
        if for_update:
            statement = statement.with_for_update()
        row = (await connection.execute(statement)).mappings().first()
        if row is None:
            raise KeyError(f"memory not found: {memory_id}")
        return _row_to_memory(row)


async def _insert_memory(
    connection: AsyncConnection,
    command: CreateMemoryCommand,
    *,
    memory_type: MemoryType,
    supersedes_memory_ids: list[str],
) -> MemoryRecord:
    now = _now()
    statement = (
        sa.insert(_memories)
        .values(
            {
                "memory_id": _uuid(command.memory_id or _new_id()),
                "namespace": command.namespace,
                "memory_type": memory_type.value,
                "content": command.content,
                "summary": command.summary,
                "content_hash": _content_hash(command.content),
                "sensitivity": command.sensitivity.value,
                "confidence": command.confidence,
                "importance": command.importance,
                "status": MemoryStatus.ACTIVE.value,
                "indexing_status": IndexingStatus.EMBEDDING_PENDING.value,
                "source_event_ids": [_uuid(value) for value in command.source_event_ids],
                "supersedes_memory_ids": [_uuid(value) for value in supersedes_memory_ids],
                "superseded_by_memory_id": None,
                "revision": 1,
                "valid_from": command.valid_from,
                "valid_until": command.valid_until,
                "archived_at": None,
                "archive_reason": None,
                "created_at": now,
                "updated_at": now,
                "metadata": command.metadata,
            },
        )
        .returning(*_memories.c)
    )
    row = (await connection.execute(statement)).mappings().one()
    return _row_to_memory(row)


async def _set_indexing_status(
    connection: AsyncConnection,
    memory_id: str,
    indexing_status: IndexingStatus,
) -> MemoryRecord:
    row = (
        await connection.execute(
            sa.update(_memories)
            .where(_memories.c.memory_id == _uuid(memory_id))
            .values(
                {
                    "indexing_status": indexing_status.value,
                    "updated_at": _now(),
                },
            )
            .returning(*_memories.c),
        )
    ).mappings().one()
    return _row_to_memory(row)


async def _upsert_embedding(
    connection: AsyncConnection,
    memory: MemoryRecord,
    *,
    embedding_profile: str,
    embedding_model: str,
    vector: list[float],
) -> MemoryEmbeddingRecord:
    statement = (
        postgresql.insert(_memory_embeddings)
        .values(
            {
                "memory_id": _uuid(memory.id),
                "embedding_profile": embedding_profile,
                "embedding_model": embedding_model,
                "embedding_dimension": len(vector),
                "content_hash": memory.content_hash,
                "embedding": vector,
                "created_at": _now(),
                "metadata": {},
            },
        )
        .on_conflict_do_update(
            index_elements=[
                _memory_embeddings.c.memory_id,
                _memory_embeddings.c.embedding_profile,
            ],
            set_={
                "embedding_model": embedding_model,
                "embedding_dimension": len(vector),
                "content_hash": memory.content_hash,
                "embedding": vector,
                "created_at": _now(),
                "metadata": {},
            },
        )
        .returning(*_memory_embeddings.c)
    )
    row = (await connection.execute(statement)).mappings().one()
    return _row_to_embedding(row)


def _memory_event(
    event_type: EventType,
    memory: MemoryRecord,
    *,
    payload_extra: dict[str, Any] | None = None,
) -> EventEnvelope:
    payload = {
        "memory_id": memory.id,
        "namespace": memory.namespace,
        "memory_type": memory.memory_type.value,
        "content_hash": memory.content_hash,
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
        source_component="memory_store",
        source_node=None,
        sensitivity=memory.sensitivity,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload=payload,
        metadata={},
    )


def _memory_write_policy_event(
    command: CreateMemoryCommand,
    *,
    allowed: bool,
    code: str,
    reason: str,
) -> EventEnvelope:
    now = _now()
    return EventEnvelope(
        event_id=_new_id(),
        event_seq=0,
        event_type=EventType.POLICY_DECISION_RECORDED,
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
        source_component="memory_store",
        source_node=None,
        sensitivity=command.sensitivity,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload={
            "source_ref": f"memory_write:{command.namespace}",
            "allowed": allowed,
            "code": code,
            "reason": reason,
        },
        metadata={},
    )


def _row_to_embedding(row: Mapping[str, Any]) -> MemoryEmbeddingRecord:
    return MemoryEmbeddingRecord(
        memory_id=str(row["memory_id"]),
        embedding_profile=row["embedding_profile"],
        embedding_model=row["embedding_model"],
        embedding_dimension=row["embedding_dimension"],
        content_hash=row["content_hash"],
        embedding=[float(value) for value in row["embedding"]],
        created_at=_datetime(row["created_at"]),
        metadata=dict(row["metadata"]),
    )


def _row_to_memory(row: Mapping[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        id=str(row["memory_id"]),
        namespace=row["namespace"],
        memory_type=MemoryType(row["memory_type"]),
        content=row["content"],
        summary=row["summary"],
        content_hash=row["content_hash"],
        sensitivity=Sensitivity(row["sensitivity"]),
        confidence=row["confidence"],
        importance=row["importance"],
        status=MemoryStatus(row["status"]),
        indexing_status=IndexingStatus(row["indexing_status"]),
        source_event_ids=[str(value) for value in row["source_event_ids"]],
        supersedes_memory_ids=[str(value) for value in row["supersedes_memory_ids"]],
        superseded_by_memory_id=_optional_string(row["superseded_by_memory_id"]),
        revision=row["revision"],
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        archived_at=_optional_datetime(row["archived_at"]),
        archive_reason=row["archive_reason"],
        valid_from=_optional_datetime(row["valid_from"]),
        valid_until=_optional_datetime(row["valid_until"]),
        metadata=dict(row["metadata"]),
    )


def _content_hash(content: str) -> str:
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"


def _like_literal_pattern(value: str) -> str:
    escaped = (
        value.lower()
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{escaped}%"


def _memory_type(value: MemoryType | str) -> MemoryType:
    return value if isinstance(value, MemoryType) else MemoryType(value)


def _memory_status(value: MemoryStatus | str) -> MemoryStatus:
    return value if isinstance(value, MemoryStatus) else MemoryStatus(value)


def _sensitivity(value: Sensitivity | str) -> Sensitivity:
    return value if isinstance(value, Sensitivity) else Sensitivity(value)


def _retrieval_score(text: str, memory: MemoryRecord) -> float:
    query_terms = {term for term in text.lower().split() if term}
    memory_terms = {term for term in memory.content.lower().split() if term}
    lexical = 0.0
    if query_terms:
        lexical = len(query_terms & memory_terms) / len(query_terms)
    recency = memory.updated_at.timestamp() * 0.000000000001
    return lexical + (memory.importance * 0.1) + (memory.confidence * 0.01) + recency


def _new_id() -> str:
    return str(uuid4())


def _uuid(value: str) -> UUID:
    return UUID(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _now() -> datetime:
    return datetime.now(UTC)


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime, got {type(value).__name__}")


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)
