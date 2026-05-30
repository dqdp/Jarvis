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
from assistant_core.ports.model_provider import ModelProviderError


REPEAT_LAST_N = 256
REPEAT_PENALTY = 1.15
MAX_REPEATED_LINE_OCCURRENCES = 3
MIN_REPEATED_LINE_LENGTH = 12


class OllamaTransport(Protocol):
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


class HttpxOllamaTransport:
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
                    if not line:
                        continue
                    yield json.loads(line)


class OllamaProviderAdapter:
    def __init__(
        self,
        *,
        profile: ModelProfileConfig,
        transport: OllamaTransport,
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
            raise ModelProviderError("Ollama provider timed out") from exc
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(str(exc)) from exc

        return ChatModelResponse(
            text=_trim_repeating_lines(_extract_chat_content(response)),
        )

    async def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[str]:
        payload = self._chat_payload(request.messages, stream=True)
        try:
            stream = self._transport.stream_json(
                self._chat_url(),
                payload,
                self._profile.timeout_seconds,
            )
            emitted = ""
            async for chunk in stream:
                token = _extract_stream_delta(chunk)
                if token:
                    safe_token, emitted, should_stop = _safe_stream_delta(emitted, token)
                    if safe_token:
                        yield safe_token
                    if should_stop:
                        return
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelProviderError("Ollama provider timed out") from exc
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
            raise ModelProviderError("Ollama provider timed out") from exc
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
                self._embed_url(),
                payload,
                self._profile.timeout_seconds,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelProviderError("Ollama provider timed out") from exc
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
                self._tags_url(),
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
            "think": False,
        }
        options: dict[str, Any] = {}
        if self._profile.temperature is not None:
            options["temperature"] = self._profile.temperature
        if self._profile.max_output_tokens is not None:
            options["num_predict"] = self._profile.max_output_tokens
        options["repeat_last_n"] = REPEAT_LAST_N
        options["repeat_penalty"] = REPEAT_PENALTY
        if options:
            payload["options"] = options
        return payload

    def _chat_url(self) -> str:
        if not self._profile.endpoint:
            raise ModelProviderError("Ollama endpoint is not configured")
        return f"{self._profile.endpoint.rstrip('/')}/api/chat"

    def _embed_url(self) -> str:
        if not self._profile.endpoint:
            raise ModelProviderError("Ollama endpoint is not configured")
        return f"{self._profile.endpoint.rstrip('/')}/api/embed"

    def _tags_url(self) -> str:
        if not self._profile.endpoint:
            raise ModelProviderError("Ollama endpoint is not configured")
        return f"{self._profile.endpoint.rstrip('/')}/api/tags"


def _message_payload(message: ChatMessage) -> dict[str, str]:
    return {
        "role": message.role.value,
        "content": "\n".join(part.text for part in message.content),
    }


def _extract_chat_content(response: dict[str, Any]) -> str:
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ModelProviderError("invalid Ollama chat response") from exc
    if not isinstance(content, str):
        raise ModelProviderError("invalid Ollama chat content")
    return content


def _extract_stream_delta(chunk: dict[str, Any]) -> str | None:
    if "error" in chunk:
        error = chunk["error"]
        if not isinstance(error, str):
            raise ModelProviderError("invalid Ollama stream error")
        raise ModelProviderError(error)

    message = chunk.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if content is None:
        return None
    if not isinstance(content, str):
        raise ModelProviderError("invalid Ollama stream delta")
    return content


def _safe_stream_delta(current_text: str, delta: str) -> tuple[str, str, bool]:
    candidate = current_text + delta
    safe_text = _trim_repeating_lines(candidate)
    if safe_text == candidate:
        return delta, candidate, False
    if safe_text.startswith(current_text):
        return safe_text.removeprefix(current_text), safe_text, True
    return "", current_text, True


def _trim_repeating_lines(text: str) -> str:
    occurrences: dict[str, int] = {}
    accepted: list[str] = []

    for line in text.splitlines(keepends=True):
        normalized = _normalize_repeated_line(line)
        if normalized is not None:
            occurrences[normalized] = occurrences.get(normalized, 0) + 1
            if occurrences[normalized] > MAX_REPEATED_LINE_OCCURRENCES:
                return "".join(accepted)
        accepted.append(line)

    return text


def _normalize_repeated_line(line: str) -> str | None:
    normalized = " ".join(line.strip().lower().split())
    if len(normalized) < MIN_REPEATED_LINE_LENGTH:
        return None
    return normalized


def _model_list_contains(response: dict[str, Any], model: str) -> bool:
    items = response.get("models")
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("name") == model or item.get("model") == model:
            return True
    return False


def _extract_embeddings(response: dict[str, Any]) -> list[list[float]]:
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list):
        raise ModelProviderError("invalid Ollama embedding response")

    vectors: list[list[float]] = []
    for vector in embeddings:
        if not isinstance(vector, list) or not all(
            isinstance(value, int | float)
            for value in vector
        ):
            raise ModelProviderError("invalid Ollama embedding vector")
        vectors.append([float(value) for value in vector])
    return vectors
