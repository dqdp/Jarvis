from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from assistant_core.domain.sensitivity import Sensitivity


class ContentSourceType(StrEnum):
    README = "readme"
    PROJECT_DOC = "project_doc"
    ADR = "adr"


class ContentSourceStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    DELETED = "deleted"
    FAILED = "failed"


class ContentChunkStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    DELETED = "deleted"


@dataclass(frozen=True)
class ContentCitation:
    path: Path
    line_start: int
    line_end: int
    heading_path: list[str] = field(default_factory=list)

    def format(self) -> str:
        return f"{self.path.as_posix()}:{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class ContentSource:
    source_id: str
    source_type: ContentSourceType
    path: Path
    uri: str
    title: str
    content_hash: str
    last_seen_at: datetime
    indexed_at: datetime
    status: ContentSourceStatus
    sensitivity: Sensitivity
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentChunk:
    chunk_id: str
    source_id: str
    ordinal: int
    source_path: Path
    source_type: ContentSourceType
    heading_path: list[str]
    content: str
    content_hash: str
    line_start: int
    line_end: int
    citation: ContentCitation
    sensitivity: Sensitivity
    status: ContentChunkStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentIngestionResult:
    seen_sources: int = 0
    created_sources: int = 0
    updated_sources: int = 0
    deleted_sources: int = 0
    created_chunks: int = 0
    stale_chunks: int = 0
    deleted_chunks: int = 0


@dataclass(frozen=True)
class ContentSourceSyncResult:
    created_source: bool = False
    updated_source: bool = False
    created_chunks: int = 0
    stale_chunks: int = 0


@dataclass(frozen=True)
class ReingestionPlan:
    reingest_required: bool
    source_status: ContentSourceStatus
    previous_chunk_status: ContentChunkStatus | None


@dataclass(frozen=True)
class DeletedSourcePlan:
    source_status: ContentSourceStatus
    chunk_status: ContentChunkStatus
