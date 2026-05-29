from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_runtime_app(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "truncate table memory_embeddings, memory_candidates, memories, "
                    "model_invocations, assistant_requests, messages, conversations, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


async def _request(app, method: str, path: str, body: dict[str, Any] | None = None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        kwargs: dict[str, Any] = {"json": body} if body is not None else {}
        response = await client.request(method, path, **kwargs)
    return response.status_code, response.text


def _sse_events(raw: str) -> list[str]:
    return [
        block.splitlines()[0].removeprefix("event: ")
        for block in raw.strip().split("\n\n")
        if block
    ]


def test_runtime_app_factory_builds_dogfood_app_with_fake_providers() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        model_provider = FakeModelProvider(stream_tokens=["OK"])
        embedding_provider = FakeEmbeddingProvider()
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_main": model_provider,
                "local_embedding": embedding_provider,
            },
        )
        try:
            app = runtime_app.app
            health_status, health_raw = await _request(app, "GET", "/v1/health")
            _, conversation_raw = await _request(
                app,
                "POST",
                "/v1/conversations",
                {"title": "factory", "active_project_namespace": "project.personal_assistant"},
            )
            conversation = json.loads(conversation_raw)
            await _request(
                app,
                "POST",
                "/v1/memories",
                {
                    "namespace": "project.personal_assistant",
                    "memory_type": "fact",
                    "content": "factory memory",
                    "sensitivity": "project",
                },
            )
            _, message_raw = await _request(
                app,
                "POST",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
                {
                    "client_message_id": "client-factory",
                    "content": "factory memory",
                    "sensitivity": "project",
                },
            )
            submitted = json.loads(message_raw)
            _, stream_raw = await _request(
                app,
                "GET",
                f"/v1/requests/{submitted['request_id']}/stream",
            )
            _, request_raw = await _request(
                app,
                "GET",
                f"/v1/requests/{submitted['request_id']}",
            )
            return (
                health_status,
                json.loads(health_raw),
                _sse_events(stream_raw),
                json.loads(request_raw),
                model_provider.stream_calls,
                embedding_provider.embed_calls,
            )
        finally:
            await runtime_app.dispose()

    health_status, health, stream_events, request_status, stream_calls, embed_calls = asyncio.run(
        scenario(),
    )

    assert health_status == 200
    assert health["status"] == "ready"
    assert stream_events[-1] == "request.processing.completed"
    assert request_status["status"] == "completed"
    assert stream_calls == 1
    assert embed_calls == 1
