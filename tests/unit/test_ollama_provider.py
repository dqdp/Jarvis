from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import ChatModelRequest, EmbeddingRequest
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.ollama import OllamaProviderAdapter
from assistant_core.models.router import ModelProviderError


pytestmark = pytest.mark.unit


class FakeOllamaTransport:
    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        stream_chunks: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {"message": {"content": "default response"}}
        self.stream_chunks = stream_chunks or []
        self.error = error
        self.requests: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int | None,
    ) -> dict[str, Any]:
        self.requests.append(
            {"url": url, "payload": payload, "timeout_seconds": timeout_seconds},
        )
        if self.error is not None:
            raise self.error
        return self.response

    async def stream_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int | None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.requests.append(
            {"url": url, "payload": payload, "timeout_seconds": timeout_seconds},
        )
        if self.error is not None:
            raise self.error
        for chunk in self.stream_chunks:
            yield chunk


def _settings():
    return ConfigLoader(Path("config")).load("ollama")


def _profile(name: str):
    return _settings().model_profiles[name]


def _request() -> ChatModelRequest:
    return ChatModelRequest(
        profile="local_main",
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content=[TextPart(text="hello")],
                sensitivity=Sensitivity.PROJECT,
            ),
        ],
        sensitivity=Sensitivity.PROJECT,
    )


def test_ollama_provider_disables_thinking_for_chat() -> None:
    async def scenario():
        transport = FakeOllamaTransport(response={"message": {"content": "OK"}})
        adapter = OllamaProviderAdapter(profile=_profile("local_main"), transport=transport)
        response = await adapter.chat(_request())
        return response, transport.requests[0]

    response, recorded = asyncio.run(scenario())

    assert response.text == "OK"
    assert recorded["url"] == "http://127.0.0.1:11434/api/chat"
    assert recorded["payload"]["model"] == "qwen3.5:4b"
    assert recorded["payload"]["stream"] is False
    assert recorded["payload"]["think"] is False
    assert recorded["payload"]["options"]["num_predict"] == 1024


def test_ollama_provider_streams_content_only() -> None:
    async def scenario():
        transport = FakeOllamaTransport(
            stream_chunks=[
                {"message": {"reasoning": "hidden"}},
                {"message": {"content": "O"}},
                {"message": {"content": "K"}},
                {"done": True},
            ],
        )
        adapter = OllamaProviderAdapter(profile=_profile("local_main"), transport=transport)
        return [token async for token in adapter.stream_chat(_request())]

    assert asyncio.run(scenario()) == ["O", "K"]


def test_ollama_provider_stream_error_chunk_fails_request() -> None:
    async def scenario() -> None:
        transport = FakeOllamaTransport(
            stream_chunks=[
                {"message": {"content": "partial"}},
                {"error": "model runner failed"},
            ],
        )
        adapter = OllamaProviderAdapter(profile=_profile("local_main"), transport=transport)
        return [token async for token in adapter.stream_chat(_request())]

    with pytest.raises(ModelProviderError, match="model runner failed"):
        asyncio.run(scenario())


def test_ollama_provider_builds_embedding_request() -> None:
    async def scenario():
        transport = FakeOllamaTransport(response={"embeddings": [[0.1, 0.2]]})
        adapter = OllamaProviderAdapter(
            profile=_profile("local_embedding"),
            transport=transport,
        )
        response = await adapter.embed(
            EmbeddingRequest(
                profile="local_embedding",
                texts=["hello"],
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        return response, transport.requests[0]

    response, recorded = asyncio.run(scenario())

    assert response.vectors == [[0.1, 0.2]]
    assert recorded["url"] == "http://127.0.0.1:11434/api/embed"
    assert recorded["payload"] == {"model": "embeddinggemma:latest", "input": ["hello"]}


def test_ollama_provider_rejects_embedding_cardinality_mismatch() -> None:
    async def scenario() -> None:
        transport = FakeOllamaTransport(response={"embeddings": [[0.1, 0.2]]})
        adapter = OllamaProviderAdapter(
            profile=_profile("local_embedding"),
            transport=transport,
        )
        await adapter.embed(
            EmbeddingRequest(
                profile="local_embedding",
                texts=["hello", "world"],
                sensitivity=Sensitivity.PROJECT,
            ),
        )

    with pytest.raises(ModelProviderError, match="embedding count mismatch"):
        asyncio.run(scenario())


def test_ollama_timeout_maps_to_model_error() -> None:
    async def scenario() -> None:
        transport = FakeOllamaTransport(error=TimeoutError("timed out"))
        adapter = OllamaProviderAdapter(profile=_profile("local_main"), transport=transport)
        await adapter.chat(_request())

    with pytest.raises(ModelProviderError):
        asyncio.run(scenario())


def test_ollama_httpx_timeout_uses_timeout_error_message() -> None:
    async def scenario() -> None:
        transport = FakeOllamaTransport(error=httpx.ReadTimeout("read timed out"))
        adapter = OllamaProviderAdapter(profile=_profile("local_main"), transport=transport)
        await adapter.chat(_request())

    with pytest.raises(ModelProviderError, match="Ollama provider timed out"):
        asyncio.run(scenario())
