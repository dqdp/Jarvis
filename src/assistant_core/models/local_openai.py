from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any, Protocol

import httpx

from assistant_core.config.settings import ModelProfileConfig
from assistant_core.domain.messages import ChatMessage
from assistant_core.domain.models import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredModelRequest,
)
from assistant_core.models.router import ModelProviderError


class OpenAICompatibleTransport(Protocol):
    async def get_json(
        self,
        url: str,
        timeout_seconds: int | None,
    ) -> dict[str, Any]: ...

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int | None,
    ) -> dict[str, Any]: ...

    def stream_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int | None,
    ) -> AsyncIterator[dict[str, Any]]: ...


class HttpxOpenAICompatibleTransport:
    async def get_json(
        self,
        url: str,
        timeout_seconds: int | None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int | None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def stream_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int | None,
    ) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if not raw or raw == "[DONE]":
                        break
                    yield json.loads(raw)


class LocalOpenAICompatibleProviderAdapter:
    def __init__(
        self,
        *,
        profile: ModelProfileConfig,
        transport: OpenAICompatibleTransport,
    ) -> None:
        self._profile = profile
        self._transport = transport

    async def chat(self, request: ChatModelRequest) -> ChatModelResponse:
        payload = self._chat_payload(request.messages, stream=False)
        try:
            response = await self._transport.post_json(
                self._chat_url(),
                payload,
                self._profile.timeout_seconds,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(str(exc)) from exc

        return ChatModelResponse(text=_extract_chat_content(response))

    async def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[str]:
        payload = self._chat_payload(request.messages, stream=True)
        try:
            stream = self._transport.stream_json(
                self._chat_url(),
                payload,
                self._profile.timeout_seconds,
            )
            async for chunk in stream:
                token = _extract_stream_delta(chunk)
                if token:
                    yield token
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(str(exc)) from exc

    async def structured(self, request: StructuredModelRequest) -> str:
        payload = self._chat_payload(request.messages, stream=False)
        try:
            response = await self._transport.post_json(
                self._chat_url(),
                payload,
                self._profile.timeout_seconds,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(str(exc)) from exc

        return _extract_chat_content(response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload = {
            "model": self._profile.model,
            "input": request.texts,
        }
        try:
            response = await self._transport.post_json(
                self._embedding_url(),
                payload,
                self._profile.timeout_seconds,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(str(exc)) from exc

        vectors = _extract_embeddings(response)
        if len(vectors) != len(request.texts):
            raise ModelProviderError("embedding count mismatch")
        return EmbeddingResponse(vectors=vectors)

    async def health_check(self) -> bool:
        try:
            response = await self._transport.get_json(
                self._models_url(),
                self._profile.timeout_seconds,
            )
        except Exception:
            return False
        return _model_list_contains(response, self._profile.model)

    def _chat_payload(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._profile.model,
            "messages": [_message_payload(message) for message in messages],
            "stream": stream,
        }
        if self._profile.temperature is not None:
            payload["temperature"] = self._profile.temperature
        if self._profile.max_output_tokens is not None:
            payload["max_tokens"] = self._profile.max_output_tokens
        return payload

    def _chat_url(self) -> str:
        if not self._profile.endpoint:
            raise ModelProviderError("local OpenAI-compatible endpoint is not configured")
        return f"{self._profile.endpoint.rstrip('/')}/chat/completions"

    def _embedding_url(self) -> str:
        if not self._profile.endpoint:
            raise ModelProviderError("local OpenAI-compatible endpoint is not configured")
        return f"{self._profile.endpoint.rstrip('/')}/embeddings"

    def _models_url(self) -> str:
        if not self._profile.endpoint:
            raise ModelProviderError("local OpenAI-compatible endpoint is not configured")
        return f"{self._profile.endpoint.rstrip('/')}/models"


def _message_payload(message: ChatMessage) -> dict[str, str]:
    return {
        "role": message.role.value,
        "content": "\n".join(part.text for part in message.content),
    }


def _extract_chat_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelProviderError("invalid local OpenAI-compatible chat response") from exc
    if not isinstance(content, str):
        raise ModelProviderError("invalid local OpenAI-compatible chat content")
    return content


def _extract_stream_delta(chunk: dict[str, Any]) -> str | None:
    try:
        delta = chunk["choices"][0].get("delta", {})
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelProviderError("invalid local OpenAI-compatible stream chunk") from exc
    content = delta.get("content")
    if content is None:
        return None
    if not isinstance(content, str):
        raise ModelProviderError("invalid local OpenAI-compatible stream delta")
    return content


def _extract_embeddings(response: dict[str, Any]) -> list[list[float]]:
    try:
        items = response["data"]
    except KeyError as exc:
        raise ModelProviderError("invalid local OpenAI-compatible embedding response") from exc
    if not isinstance(items, list):
        raise ModelProviderError("invalid local OpenAI-compatible embedding response")

    vectors: list[list[float]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ModelProviderError("invalid local OpenAI-compatible embedding item")
        vector = item.get("embedding")
        if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
            raise ModelProviderError("invalid local OpenAI-compatible embedding vector")
        vectors.append([float(value) for value in vector])
    return vectors


def _model_list_contains(response: dict[str, Any], model: str) -> bool:
    items = response.get("data")
    if not isinstance(items, list):
        return False
    return any(isinstance(item, dict) and item.get("id") == model for item in items)
