from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from assistant_core.domain.models import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredModelRequest,
)


class ModelProviderError(Exception):
    """Raised when a concrete model provider cannot satisfy a request."""


class ModelProviderPort(Protocol):
    async def chat(self, request: ChatModelRequest) -> ChatModelResponse: ...

    def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[str]: ...

    async def structured(self, request: StructuredModelRequest) -> str: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
