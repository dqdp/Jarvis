from __future__ import annotations

from assistant_core.domain.content_retrieval import ContentHit, ContentRetrievalQuery


class ContentRetrievalService:
    """Application boundary for content retrieval workflows."""

    def __init__(self, store) -> None:
        self._store = store

    async def retrieve(
        self,
        query: ContentRetrievalQuery,
    ) -> list[ContentHit]:
        return await self._store.retrieve(query)

    async def list_sources(self):
        return await self._store.list_sources()

    async def list_chunks(self):
        return await self._store.list_chunks()
