from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from assistant_core.config.settings import ModelProfileConfig
from assistant_core.domain.messages import ChatMessage
from assistant_core.domain.models import (
    ChatModelRequest,
    ChatModelResponse,
    StructuredModelRequest,
)
from assistant_core.models.router import ModelProviderError


class OpenAICompatibleTransport(Protocol):
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
        except TimeoutError as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
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
        except TimeoutError as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
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
        except TimeoutError as exc:
            raise ModelProviderError("local OpenAI-compatible provider timed out") from exc
        except Exception as exc:
            raise ModelProviderError(str(exc)) from exc

        return _extract_chat_content(response)

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
