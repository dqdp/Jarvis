from __future__ import annotations

from typing import Any, Protocol

import httpx

from assistant_core.cli_app.config import (
    REQUEST_TIMEOUT_SECONDS,
    STREAM_CONNECT_TIMEOUT_SECONDS,
    STREAM_READ_TIMEOUT_SECONDS,
)
from assistant_core.cli_app.sse import parse_sse_blocks


class JarvisClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def create_conversation(
        self,
        *,
        title: str | None,
        active_project_namespace: str | None,
    ) -> dict[str, Any]: ...

    async def list_conversations(self, *, limit: int = 20) -> dict[str, Any]: ...

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]: ...

    async def submit_message(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        content: str,
        sensitivity: str,
        loop_strategy: str | None = None,
        working_directory: str | None = None,
    ) -> dict[str, Any]: ...

    def stream_request(self, request_id: str): ...

    async def create_memory(
        self,
        *,
        namespace: str,
        memory_type: str,
        content: str,
        sensitivity: str,
    ) -> dict[str, Any]: ...

    async def list_memories(self) -> dict[str, Any]: ...

    async def search_memories(self, query: str) -> dict[str, Any]: ...

    async def delete_memory(self, memory_id: str) -> dict[str, Any]: ...

    async def cancel_request(self, request_id: str) -> dict[str, Any]: ...

    async def get_request_status(self, request_id: str) -> dict[str, Any]: ...

    async def runtime_status(self) -> dict[str, Any]: ...

    async def get_approval(self, approval_id: str) -> dict[str, Any]: ...

    async def grant_approval(self, approval_id: str) -> dict[str, Any]: ...

    async def deny_approval(self, approval_id: str) -> dict[str, Any]: ...

    async def ingest_project_docs(self) -> dict[str, Any]: ...

    async def reindex_project_docs(self) -> dict[str, Any]: ...

    async def list_content_sources(self) -> dict[str, Any]: ...

    async def content_status(self) -> dict[str, Any]: ...


class CliUserError(Exception):
    pass


class HttpJarvisClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def health(self) -> dict[str, Any]:
        return await self._get_json("/v1/health", accepted_status_codes={200, 503})

    async def create_conversation(
        self,
        *,
        title: str | None,
        active_project_namespace: str | None,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/v1/conversations",
            {
                "title": title,
                "active_project_namespace": active_project_namespace,
                "metadata": {},
            },
        )

    async def list_conversations(self, *, limit: int = 20) -> dict[str, Any]:
        return await self._get_json("/v1/conversations", params={"limit": limit})

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return await self._get_json(f"/v1/conversations/{conversation_id}")

    async def submit_message(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        content: str,
        sensitivity: str,
        loop_strategy: str | None = None,
        working_directory: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "client_message_id": client_message_id,
            "content": content,
            "sensitivity": sensitivity,
            "metadata": {},
        }
        if loop_strategy is not None:
            payload["loop_strategy"] = loop_strategy
        if working_directory is not None:
            payload["working_directory"] = working_directory
        return await self._post_json(
            f"/v1/conversations/{conversation_id}/messages",
            payload,
        )

    async def stream_request(self, request_id: str):
        try:
            timeout = httpx.Timeout(
                connect=STREAM_CONNECT_TIMEOUT_SECONDS,
                read=STREAM_READ_TIMEOUT_SECONDS,
                write=STREAM_CONNECT_TIMEOUT_SECONDS,
                pool=STREAM_CONNECT_TIMEOUT_SECONDS,
            )
            async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout) as client:
                async with client.stream("GET", f"/v1/requests/{request_id}/stream") as response:
                    response.raise_for_status()
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n\n" in buffer:
                            block, buffer = buffer.split("\n\n", 1)
                            for event in parse_sse_blocks(f"{block}\n\n"):
                                yield event
        except CliUserError:
            raise
        except httpx.HTTPStatusError as exc:
            raise CliUserError(_http_error_message(exc, "stream request")) from exc
        except httpx.HTTPError as exc:
            raise CliUserError(f"cannot reach daemon at {self._base_url}: {exc}") from exc
        except ValueError as exc:
            raise CliUserError("invalid streaming response from daemon") from exc

    async def create_memory(
        self,
        *,
        namespace: str,
        memory_type: str,
        content: str,
        sensitivity: str,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/v1/memories",
            {
                "namespace": namespace,
                "memory_type": memory_type,
                "content": content,
                "sensitivity": sensitivity,
                "metadata": {},
            },
        )

    async def list_memories(self) -> dict[str, Any]:
        return await self._get_json("/v1/memories")

    async def search_memories(self, query: str) -> dict[str, Any]:
        return await self._get_json("/v1/memories", params={"query": query})

    async def delete_memory(self, memory_id: str) -> dict[str, Any]:
        return await self._post_json(f"/v1/memories/{memory_id}/archive", {})

    async def cancel_request(self, request_id: str) -> dict[str, Any]:
        return await self._post_json(f"/v1/requests/{request_id}/cancel", {})

    async def get_request_status(self, request_id: str) -> dict[str, Any]:
        return await self._get_json(f"/v1/requests/{request_id}")

    async def runtime_status(self) -> dict[str, Any]:
        return await self._get_json("/v1/runtime/status")

    async def get_approval(self, approval_id: str) -> dict[str, Any]:
        return await self._get_json(f"/v1/approvals/{approval_id}")

    async def grant_approval(self, approval_id: str) -> dict[str, Any]:
        return await self._post_json(f"/v1/approvals/{approval_id}/grant", {})

    async def deny_approval(self, approval_id: str) -> dict[str, Any]:
        return await self._post_json(f"/v1/approvals/{approval_id}/deny", {})

    async def ingest_project_docs(self) -> dict[str, Any]:
        return await self._post_json("/v1/content/project-docs/ingest", {})

    async def reindex_project_docs(self) -> dict[str, Any]:
        return await self._post_json("/v1/content/project-docs/reindex", {})

    async def list_content_sources(self) -> dict[str, Any]:
        return await self._get_json("/v1/content/sources")

    async def content_status(self) -> dict[str, Any]:
        return await self._get_json("/v1/content/status")

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accepted_status_codes: set[int] | None = None,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as client:
                response = (
                    await client.get(path)
                    if params is None
                    else await client.get(path, params=params)
                )
                if getattr(response, "status_code", 200) in (accepted_status_codes or {200}):
                    return response.json()
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise CliUserError(_http_error_message(exc, path)) from exc
        except httpx.HTTPError as exc:
            raise CliUserError(f"cannot reach daemon at {self._base_url}: {exc}") from exc
        except ValueError as exc:
            raise CliUserError(f"invalid JSON response from daemon for {path}") from exc

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise CliUserError(_http_error_message(exc, path)) from exc
        except httpx.HTTPError as exc:
            raise CliUserError(f"cannot reach daemon at {self._base_url}: {exc}") from exc
        except ValueError as exc:
            raise CliUserError(f"invalid JSON response from daemon for {path}") from exc


def _http_error_message(exc: httpx.HTTPStatusError, action: str) -> str:
    detail = exc.response.text
    try:
        payload = exc.response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            code = error.get("code")
            detail = f"{code}: {error['message']}" if isinstance(code, str) else error["message"]
        elif isinstance(payload.get("detail"), str):
            detail = payload["detail"]
    return f"{action} failed: {exc.response.status_code} {detail}"

__all__ = ["CliUserError", "HttpJarvisClient", "JarvisClient", "_http_error_message"]
