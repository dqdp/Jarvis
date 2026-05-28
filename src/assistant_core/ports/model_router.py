from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from assistant_core.domain.models import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelStreamEvent,
    StructuredModelRequest,
    StructuredModelResponse,
)


class ModelRouterPort(Protocol):
    async def chat(self, request: ChatModelRequest) -> ChatModelResponse: ...

    def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[ModelStreamEvent]: ...

    async def structured(
        self,
        request: StructuredModelRequest,
    ) -> StructuredModelResponse: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
