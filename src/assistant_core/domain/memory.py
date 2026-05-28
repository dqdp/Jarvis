from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from assistant_core.domain.sensitivity import Sensitivity


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    PROCEDURE = "procedure"
    SUMMARY = "summary"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


class IndexingStatus(StrEnum):
    INDEXED = "indexed"
    EMBEDDING_PENDING = "embedding_pending"
    EMBEDDING_FAILED = "embedding_failed"


class MemoryCandidateStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"
    EXPIRED = "expired"


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    namespace: str
    memory_type: MemoryType
    content: str
    summary: str | None
    content_hash: str
    sensitivity: Sensitivity
    confidence: float
    importance: float
    status: MemoryStatus
    indexing_status: IndexingStatus
    source_event_ids: list[str]
    supersedes_memory_ids: list[str]
    superseded_by_memory_id: str | None
    revision: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    archive_reason: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEmbeddingRecord:
    memory_id: str
    embedding_profile: str
    embedding_model: str
    embedding_dimension: int
    content_hash: str
    embedding: list[float]
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHit:
    memory: MemoryRecord
    score: float


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    namespaces: list[str]
    memory_types: list[MemoryType | str] | None = None
    include_statuses: list[MemoryStatus | str] = field(
        default_factory=lambda: [MemoryStatus.ACTIVE],
    )
    limit: int | None = None


@dataclass(frozen=True)
class MemoryCandidate:
    candidate_id: str
    proposed_namespace: str
    proposed_memory_type: MemoryType
    content: str
    sensitivity: Sensitivity
    status: MemoryCandidateStatus
    created_by: str
    created_at: datetime
    confidence: float | None = None
    source_event_ids: list[str] = field(default_factory=list)
    resolved_at: datetime | None = None
    resolution_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CreateMemoryCommand:
    namespace: str
    memory_type: MemoryType | str
    content: str
    summary: str | None
    sensitivity: Sensitivity
    confidence: float
    importance: float
    memory_id: str | None = None
    source_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_from: datetime | None = None
    valid_until: datetime | None = None


@dataclass(frozen=True)
class ArchiveMemoryCommand:
    memory_id: str
    reason: str


@dataclass(frozen=True)
class SupersedeMemoryCommand:
    superseded_memory_id: str
    replacement: CreateMemoryCommand


@dataclass(frozen=True)
class UpdateMemoryCommand:
    memory_id: str
    content: str | None = None
    summary: str | None = None
    confidence: float | None = None
    importance: float | None = None
    metadata: dict[str, Any] | None = None
