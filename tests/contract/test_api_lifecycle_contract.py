from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy import text

from assistant_core.api.app import create_app
from assistant_core.config.settings import ConfigLoader
from assistant_core.models.fake_provider import FakeEmbeddingProvider
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_api(database_url: str) -> None:
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


@pytest.fixture
def app_parts():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_api(database_url))
    engine = create_database_engine(database_url)
    settings = ConfigLoader(Path("config")).load("test")
    app = create_app(
        conversation_store=PostgresConversationStore(engine),
            memory_store=PostgresMemoryStore(
                engine=engine,
                settings=settings,
                policy=ConfigPolicyEngine(settings),
                embedding_port=FakeEmbeddingProvider(),
            ),
        settings=settings,
    )
    app.state.engine = engine
    assert isinstance(app, FastAPI)
    try:
        yield app
    finally:
        asyncio.run(engine.dispose())


async def _request(app, method: str, path: str, body: dict[str, Any] | None = None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        kwargs: dict[str, Any] = {"json": body} if body is not None else {}
        response = await client.request(method, path, **kwargs)
    payload = response.json() if response.content else None
    return response.status_code, payload


async def _create_conversation(app):
    status, payload = await _request(
        app,
        "POST",
        "/v1/conversations",
        {
            "title": "Runtime",
            "active_project_namespace": "project.personal_assistant",
            "metadata": {},
        },
    )
    assert status == 201
    return payload


def test_post_conversation(app_parts) -> None:
    status, payload = asyncio.run(
        _request(
            app_parts,
            "POST",
            "/v1/conversations",
            {
                "title": "Phase 1",
                "active_project_namespace": "project.personal_assistant",
                "metadata": {"source": "test"},
            },
        ),
    )

    assert status == 201
    assert payload["status"] == "active"
    assert payload["title"] == "Phase 1"


def test_get_conversations_lists_recent_sessions(app_parts) -> None:
    async def scenario():
        first = await _create_conversation(app_parts)
        second_status, second = await _request(
            app_parts,
            "POST",
            "/v1/conversations",
            {
                "title": "Second",
                "active_project_namespace": "project.personal_assistant",
                "metadata": {},
            },
        )
        assert second_status == 201
        return first, second, await _request(app_parts, "GET", "/v1/conversations")

    first, second, (status, payload) = asyncio.run(scenario())

    assert status == 200
    assert [item["conversation_id"] for item in payload["conversations"][:2]] == [
        second["conversation_id"],
        first["conversation_id"],
    ]


def test_get_conversations_rejects_invalid_limit(app_parts) -> None:
    status, payload = asyncio.run(_request(app_parts, "GET", "/v1/conversations?limit=0"))

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"


def test_get_conversation(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return conversation, await _request(
            app_parts,
            "GET",
            f"/v1/conversations/{conversation['conversation_id']}",
        )

    conversation, (status, payload) = asyncio.run(scenario())

    assert status == 200
    assert payload["conversation_id"] == conversation["conversation_id"]
    assert payload["title"] == "Runtime"


def test_post_message_returns_request_id(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-1",
                "content": "hello",
                "sensitivity": "project",
                "metadata": {},
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 202
    assert payload["request_id"]
    assert payload["status"] == "accepted"
    assert payload["stream_url"] == f"/v1/requests/{payload['request_id']}/stream"


def test_get_request_status(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        _, submitted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {"client_message_id": "client-1", "content": "hello", "sensitivity": "project"},
        )
        return await _request(app_parts, "GET", f"/v1/requests/{submitted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["status"] == "accepted"
    assert payload["user_message_id"]


def test_get_conversation_messages(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {"client_message_id": "client-1", "content": "hello", "sensitivity": "project"},
        )
        return await _request(
            app_parts,
            "GET",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
        )

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "hello"


def test_post_memory(app_parts) -> None:
    status, payload = asyncio.run(
        _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "API can create manual memory.",
                "sensitivity": "project",
                "confidence": 0.9,
                "importance": 0.7,
                "metadata": {},
            },
        ),
    )

    assert status == 201
    assert payload["namespace"] == "project.personal_assistant"
    assert payload["indexing_status"] == "indexed"


def test_post_memory_defaults_sensitivity_from_namespace_registry(app_parts) -> None:
    status, payload = asyncio.run(
        _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "environment.inference_node",
                "memory_type": "fact",
                "content": "Inference node runs locally.",
                "confidence": 0.9,
                "importance": 0.7,
                "metadata": {},
            },
        ),
    )

    assert status == 201
    assert payload["namespace"] == "environment.inference_node"
    assert payload["sensitivity"] == "infra"


def test_get_memories(app_parts) -> None:
    async def scenario():
        await _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "Memory list item.",
                "sensitivity": "project",
            },
        )
        return await _request(app_parts, "GET", "/v1/memories")

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["memories"][0]["content"] == "Memory list item."


def test_get_memories_rejects_invalid_limit(app_parts) -> None:
    status, payload = asyncio.run(_request(app_parts, "GET", "/v1/memories?limit=0"))

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"


def test_search_memories_filters_by_query(app_parts) -> None:
    async def scenario():
        await _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "Alpha memory search target.",
                "sensitivity": "project",
            },
        )
        await _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "Unrelated note.",
                "sensitivity": "project",
            },
        )
        return await _request(app_parts, "GET", "/v1/memories?query=search target")

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert [memory["content"] for memory in payload["memories"]] == [
        "Alpha memory search target.",
    ]


def test_delete_memory_archives_record(app_parts) -> None:
    async def scenario():
        _, memory = await _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "Archive me.",
                "sensitivity": "project",
            },
        )
        return await _request(app_parts, "DELETE", f"/v1/memories/{memory['memory_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["status"] == "archived"
    assert payload["archive_reason"] == "deleted_by_user"
    assert "content" not in payload
    assert "summary" not in payload


def test_archive_memory_endpoint_archives_record(app_parts) -> None:
    async def scenario():
        _, memory = await _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "Archive me explicitly.",
                "sensitivity": "project",
            },
        )
        return await _request(app_parts, "POST", f"/v1/memories/{memory['memory_id']}/archive")

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["status"] == "archived"
    assert payload["archive_reason"] == "archived_by_user"
    assert "content" not in payload
    assert "summary" not in payload


def test_archive_memory_endpoint_does_not_disclose_secret_memory_content(app_parts) -> None:
    async def scenario():
        async with app_parts.state.engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    insert into memories (
                      memory_id, namespace, memory_type, content, summary,
                      content_hash, sensitivity, confidence, importance,
                      status, indexing_status, source_event_ids, supersedes_memory_ids,
                      revision, created_at, updated_at, metadata
                    )
                    values (
                      '11111111-1111-1111-1111-111111111111',
                      'project.personal_assistant', 'fact',
                      'secret archive payload', 'secret summary', 'sha256:secret',
                      'secret', 1, 1, 'active', 'embedding_pending',
                      '{}', '{}', 1, now(), now(), '{}'
                    )
                    """,
                ),
            )
        return await _request(
            app_parts,
            "POST",
            "/v1/memories/11111111-1111-1111-1111-111111111111/archive",
        )

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["memory_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["status"] == "archived"
    assert "content" not in payload
    assert "summary" not in payload
    assert "secret archive payload" not in json.dumps(payload)


def test_get_runtime_status_exposes_active_local_profile(app_parts) -> None:
    status, payload = asyncio.run(_request(app_parts, "GET", "/v1/runtime/status"))

    assert status == 200
    assert payload["default_model_profile"] == "local_main"
    assert payload["model_profiles"]["local_main"]["provider"]
    assert payload["model_profiles"]["local_main"]["model"]
    assert "api_key_env" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload).lower()


def test_get_health(app_parts) -> None:
    status, payload = asyncio.run(_request(app_parts, "GET", "/v1/health"))

    assert status == 200
    assert payload["status"] == "ready"
    assert payload["liveness"]["status"] == "ok"
    assert payload["readiness"]["checks"]["conversation_store"] == "ok"
    assert payload["readiness"]["checks"]["memory_store"] == "ok"


def test_idempotent_message_submit(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        body = {"client_message_id": "client-repeat", "content": "hello", "sensitivity": "project"}
        _, first = await _request(app_parts, "POST", path, body)
        _, second = await _request(app_parts, "POST", path, body)
        return first, second

    first, second = asyncio.run(scenario())

    assert second["request_id"] == first["request_id"]
    assert second["idempotent_replay"] is True


def test_conflicting_client_message_id_returns_409(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        await _request(
            app_parts,
            "POST",
            path,
            {"client_message_id": "client-conflict", "content": "first", "sensitivity": "project"},
        )
        return await _request(
            app_parts,
            "POST",
            path,
            {"client_message_id": "client-conflict", "content": "second", "sensitivity": "project"},
        )

    status, payload = asyncio.run(scenario())

    assert status == 409
    assert payload["error"]["code"] == "conflict"


def test_standard_error_format(app_parts) -> None:
    status, payload = asyncio.run(_request(app_parts, "GET", "/v1/requests/not-a-uuid"))

    assert status == 400
    assert set(payload["error"]) == {"code", "message", "request_id", "details"}


def test_post_message_rejects_extra_fields(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-extra",
                "content": "hello",
                "sensitivity": "project",
                "unexpected": "field",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["details"]["errors"][0]["type"] == "extra_forbidden"


def test_validation_error_does_not_echo_raw_secret_input(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-secret-validation",
                "content": "secret password is swordfish",
                "sensitivity": "not-a-sensitivity",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert "swordfish" not in json.dumps(payload)
