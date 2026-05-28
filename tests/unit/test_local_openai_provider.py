from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
from pathlib import Path
from typing import Any

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import ChatModelRequest
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.local_openai import LocalOpenAICompatibleProviderAdapter
from assistant_core.models.router import ModelProviderError


pytestmark = pytest.mark.unit


class FakeOpenAITransport:
    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        stream_chunks: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {
            "choices": [{"message": {"content": "default response"}}],
        }
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


def _profile():
    return ConfigLoader(Path("config")).load("test").model_profiles["local_main"]


def _request() -> ChatModelRequest:
    return ChatModelRequest(
        profile="local_main",
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content=[TextPart(text="hello"), TextPart(text="world")],
                sensitivity=Sensitivity.PROJECT,
            ),
        ],
        sensitivity=Sensitivity.PROJECT,
    )


def test_local_openai_provider_builds_chat_request() -> None:
    async def scenario():
        transport = FakeOpenAITransport()
        adapter = LocalOpenAICompatibleProviderAdapter(profile=_profile(), transport=transport)
        await adapter.chat(_request())
        return transport.requests[0]

    recorded = asyncio.run(scenario())

    assert recorded["url"] == "http://inference-node:8000/v1/chat/completions"
    assert recorded["timeout_seconds"] == 120
    assert recorded["payload"]["model"] == "qwen3-32b-instruct"
    assert recorded["payload"]["stream"] is False
    assert recorded["payload"]["messages"] == [{"role": "user", "content": "hello\nworld"}]


def test_local_openai_provider_parses_chat_response() -> None:
    async def scenario():
        transport = FakeOpenAITransport(
            response={"choices": [{"message": {"content": "parsed"}}]},
        )
        adapter = LocalOpenAICompatibleProviderAdapter(profile=_profile(), transport=transport)
        return await adapter.chat(_request())

    response = asyncio.run(scenario())

    assert response.text == "parsed"


def test_local_openai_provider_stream_normalization() -> None:
    async def scenario():
        transport = FakeOpenAITransport(
            stream_chunks=[
                {"choices": [{"delta": {"content": "A"}}]},
                {"choices": [{"delta": {"content": "B"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
        )
        adapter = LocalOpenAICompatibleProviderAdapter(profile=_profile(), transport=transport)
        return [token async for token in adapter.stream_chat(_request())]

    assert asyncio.run(scenario()) == ["A", "B"]


def test_local_openai_provider_timeout_maps_to_model_error() -> None:
    async def scenario() -> None:
        transport = FakeOpenAITransport(error=TimeoutError("timed out"))
        adapter = LocalOpenAICompatibleProviderAdapter(profile=_profile(), transport=transport)
        await adapter.chat(_request())

    with pytest.raises(ModelProviderError):
        asyncio.run(scenario())
