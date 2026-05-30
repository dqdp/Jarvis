from __future__ import annotations

from collections.abc import AsyncIterator

from assistant_core.domain.models import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredModelRequest,
)
from assistant_core.ports.model_provider import ModelProviderError


class FakeModelProvider:
    def __init__(
        self,
        *,
        chat_response: str = "fake response",
        stream_tokens: list[str] | None = None,
        structured_text_responses: list[str] | None = None,
        fail_chat_times: int = 0,
        fail_stream_times: int = 0,
        call_log: list[str] | None = None,
    ) -> None:
        self.chat_response = chat_response
        self.stream_tokens = stream_tokens or ["fake response"]
        self.structured_text_responses = structured_text_responses or ['{"ok": true}']
        self.fail_chat_times = fail_chat_times
        self.fail_stream_times = fail_stream_times
        self.call_log = call_log
        self.chat_calls = 0
        self.stream_calls = 0
        self.structured_calls = 0

    async def chat(self, request: ChatModelRequest) -> ChatModelResponse:
        self.chat_calls += 1
        if self.call_log is not None:
            self.call_log.append("provider.chat")
        if self.fail_chat_times > 0:
            self.fail_chat_times -= 1
            raise ModelProviderError("fake chat failure")
        return ChatModelResponse(text=self.chat_response)

    async def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[str]:
        self.stream_calls += 1
        if self.call_log is not None:
            self.call_log.append("provider.stream_chat")
        if self.fail_stream_times > 0:
            self.fail_stream_times -= 1
            raise ModelProviderError("fake stream failure")
        for token in self.stream_tokens:
            yield token

    async def structured(self, request: StructuredModelRequest) -> str:
        self.structured_calls += 1
        index = min(self.structured_calls - 1, len(self.structured_text_responses) - 1)
        return self.structured_text_responses[index]


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        vectors: list[list[float]] | None = None,
        fail_embed_times: int = 0,
    ) -> None:
        self.vectors = vectors
        self.fail_embed_times = fail_embed_times
        self.embed_calls = 0

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embed_calls += 1
        if self.fail_embed_times > 0:
            self.fail_embed_times -= 1
            raise ModelProviderError("fake embedding failure")
        vectors = self.vectors
        if vectors is None:
            vectors = [[float(len(text))] for text in request.texts]
        return EmbeddingResponse(vectors=vectors)
