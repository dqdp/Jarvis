from __future__ import annotations

from typing import Protocol, runtime_checkable

from assistant_core.domain.content_retrieval import ContentHit, ContentRetrievalQuery


@runtime_checkable
class ContentRetrievalPort(Protocol):
    async def retrieve(self, query: ContentRetrievalQuery) -> list[ContentHit]: ...
