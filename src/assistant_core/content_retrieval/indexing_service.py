from __future__ import annotations

from pathlib import Path

from assistant_core.domain.content_retrieval import (
    ContentChunk,
    ContentSourceSyncResult,
    ProjectDocsSourceCandidate,
)


class ContentIndexingService:
    """Application boundary for content indexing workflows."""

    def __init__(self, store) -> None:
        self._store = store

    async def sync_source_chunks(
        self,
        candidate: ProjectDocsSourceCandidate,
        chunks: list[ContentChunk],
    ) -> ContentSourceSyncResult:
        return await self._store.sync_source_chunks(candidate, chunks)

    async def mark_missing_sources_deleted(self, seen_paths: set[Path]) -> tuple[int, int]:
        return await self._store.mark_missing_sources_deleted(seen_paths)
