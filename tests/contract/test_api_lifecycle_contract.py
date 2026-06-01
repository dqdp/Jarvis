from __future__ import annotations

import asyncio
from dataclasses import replace
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
from assistant_core.domain.events import EventType
from assistant_core.models.fake_provider import FakeEmbeddingProvider
from assistant_core.ports.event_log import EventFilter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.content_store import PostgresContentStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations


pytestmark = [pytest.mark.contract, pytest.mark.db]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


class _HealthyComponent:
    async def health_check(self) -> bool:
        return True


class _UnhealthyComponent:
    async def health_check(self) -> bool:
        return False


async def _truncate_api(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table content_embeddings, content_chunks, content_sources, "
                    "memory_embeddings, memory_candidates, memories, "
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
    event_log = PostgresEventLog(engine)
    policy = ConfigPolicyEngine(settings, event_log=event_log)
    app = create_app(
        conversation_store=PostgresConversationStore(engine),
        memory_store=PostgresMemoryStore(
            engine=engine,
            settings=settings,
            policy=policy,
            embedding_port=FakeEmbeddingProvider(),
        ),
        content_store=PostgresContentStore(engine=engine, embedding_port=FakeEmbeddingProvider()),
        settings=settings,
        event_log=event_log,
        policy=policy,
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


def _assert_agent_loop_request_metadata(
    metadata: dict,
    *,
    requested_mode: str = "auto",
    tool_policy: str = "available",
) -> None:
    assert metadata["requested_loop_mode"] == requested_mode
    assert metadata["selected_loop_strategy"] == "tool_react_loop"
    assert metadata["loop_strategy"] == "tool_react_loop"
    assert metadata["selected_model_profile"] == "local_main"
    assert metadata["model_profile"] == "local_main"
    assert metadata["agent_tool_policy"] == tool_policy
    assert metadata["request_plan_reason_code"].startswith("request_plan_")
    assert "loop_selection_status" not in metadata
    assert "loop_selection_reason_code" not in metadata
    assert "loop_selection_classification_source" not in metadata
    assert "loop_selection_confidence" not in metadata
    assert "loop_selection_intent_family" not in metadata
    assert "loop_selection_tool_names" not in metadata
    assert "loop_selection_direct_tool_plan" not in metadata
    assert "loop_selection_direct_tool_name" not in metadata
    assert "loop_selection_direct_tool_names" not in metadata
    assert "loop_selection_direct_scenario" not in metadata


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


def test_archive_memory_endpoint_never_discloses_memory_content(app_parts) -> None:
    async def scenario():
        _, memory = await _request(
            app_parts,
            "POST",
            "/v1/memories",
            {
                "namespace": "project.personal_assistant",
                "memory_type": "fact",
                "content": "private archive payload",
                "summary": "private summary",
                "sensitivity": "project",
            },
        )
        return await _request(
            app_parts,
            "POST",
            f"/v1/memories/{memory['memory_id']}/archive",
        )

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["status"] == "archived"
    assert "content" not in payload
    assert "summary" not in payload
    assert "private archive payload" not in json.dumps(payload)
    assert "private summary" not in json.dumps(payload)


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
    assert payload["readiness"]["checks"]["content_store"] == "ok"


def test_get_health_reports_content_store_readiness_failure() -> None:
    app = create_app(
        conversation_store=_HealthyComponent(),
        memory_store=_HealthyComponent(),
        content_store=_UnhealthyComponent(),
        settings=ConfigLoader(Path("config")).load("test"),
    )

    status, payload = asyncio.run(_request(app, "GET", "/v1/health"))

    assert status == 503
    assert payload["status"] == "not_ready"
    assert payload["readiness"]["checks"]["conversation_store"] == "ok"
    assert payload["readiness"]["checks"]["memory_store"] == "ok"
    assert payload["readiness"]["checks"]["content_store"] == "failed"


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


def test_client_message_id_reuse_with_different_sensitivity_returns_409(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-sensitivity-conflict",
                "content": "same content",
                "sensitivity": "project",
            },
        )
        return await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-sensitivity-conflict",
                "content": "same content",
                "sensitivity": "secret",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 409
    assert payload["error"]["code"] == "conflict"


def test_message_without_loop_strategy_uses_auto_mode(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-default",
                "content": "hello",
                "sensitivity": "project",
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_persists_requested_mode_and_selected_loop_metadata(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-explicit",
                "content": "what is listening on port 8080?",
                "sensitivity": "project",
                "loop_strategy": "auto",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_routes_current_time_question_to_safe_builtin_tool_loop(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-current-time",
                "content": "Сколько время?",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_routes_russian_cpu_temperature_to_system_sensors(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-russian-cpu-temperature",
                "content": "Текущая температура процессора.",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_routes_russian_free_memory_to_system_resources(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-russian-free-memory",
                "content": "Сколько памяти сейчас свободно в системе?",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_routes_russian_cpu_cores_and_load_to_cpu_overview_plan(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-russian-cpu-overview",
                "content": "Сколько ядер у центрального процессора и на сколько они загружены?",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_routes_russian_os_version_to_hardware_tool(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-russian-os-version",
                "content": "Какая версия операционной системы?",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_auto_mode_emits_loop_selection_event(app_parts) -> None:
    async def scenario():
        settings = ConfigLoader(Path("config")).load("test")
        engine = app_parts.state.engine
        event_log = PostgresEventLog(engine)
        policy = ConfigPolicyEngine(settings, event_log=event_log)
        app = create_app(
            conversation_store=PostgresConversationStore(engine),
            memory_store=PostgresMemoryStore(
                engine=engine,
                settings=settings,
                policy=policy,
                embedding_port=FakeEmbeddingProvider(),
            ),
            content_store=PostgresContentStore(engine=engine, embedding_port=FakeEmbeddingProvider()),
            settings=settings,
            event_log=event_log,
            policy=policy,
        )
        conversation = await _create_conversation(app)
        status, accepted = await _request(
            app,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-events",
                "content": "check cpu temperature",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        events = await event_log.query(EventFilter(request_id=accepted["request_id"]))
        return accepted, events

    accepted, events = asyncio.run(scenario())

    event_types = [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_STARTED in event_types
    assert EventType.LOOP_SELECTION_COMPLETED in event_types
    completed = next(
        event for event in events if event.event_type is EventType.LOOP_SELECTION_COMPLETED
    )
    assert completed.payload["request_id"] == accepted["request_id"]
    assert completed.payload["selected_loop_strategy"] == "tool_react_loop"
    assert completed.payload["request_plan_reason_code"] == "request_plan_auto_agent_loop"
    assert "intent_family" not in completed.payload
    assert "classification_source" not in completed.payload
    assert "confidence" not in completed.payload
    assert "check cpu temperature" not in json.dumps(completed.payload)
    assert "user_input" not in completed.payload
    assert "prompt" not in completed.payload


def test_idempotent_replay_does_not_emit_duplicate_loop_selection_events(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        body = {
            "client_message_id": "client-auto-event-replay",
            "content": "hello",
            "sensitivity": "project",
        }
        status, first = await _request(app_parts, "POST", path, body)
        assert status == 202
        status, second = await _request(app_parts, "POST", path, body)
        assert status == 202
        event_log = PostgresEventLog(app_parts.state.engine)
        events = await event_log.query(EventFilter(request_id=first["request_id"]))
        return first, second, events

    first, second, events = asyncio.run(scenario())

    assert second["request_id"] == first["request_id"]
    assert second["idempotent_replay"] is True
    assert [
        event.event_type for event in events if event.event_type is EventType.LOOP_SELECTION_STARTED
    ] == [EventType.LOOP_SELECTION_STARTED]
    assert [
        event.event_type for event in events if event.event_type is EventType.LOOP_SELECTION_COMPLETED
    ] == [EventType.LOOP_SELECTION_COMPLETED]


def test_conflicting_client_message_id_does_not_emit_loop_selection_events(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-conflict-no-selection-event",
                "content": "same",
                "sensitivity": "project",
            },
        )
        status, payload = await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-conflict-no-selection-event",
                "content": "different",
                "sensitivity": "project",
            },
        )
        event_log = PostgresEventLog(app_parts.state.engine)
        events = await event_log.query(
            EventFilter(
                request_id=payload["error"]["request_id"],
            ),
        )
        return status, payload, events

    status, payload, events = asyncio.run(scenario())

    assert status == 409
    assert payload["error"]["code"] == "conflict"
    assert [
        event.event_type for event in events if event.event_type is EventType.LOOP_SELECTION_STARTED
    ] == [EventType.LOOP_SELECTION_STARTED]
    assert [
        event.event_type for event in events if event.event_type is EventType.LOOP_SELECTION_COMPLETED
    ] == [EventType.LOOP_SELECTION_COMPLETED]


def test_model_profile_selection_failure_emits_loop_selection_failed_event(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, payload = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-selection-profile-failed-event",
                "content": "what is listening on port 8080?",
                "sensitivity": "project",
                "model_profile": "local_embedding",
                "working_directory": str(Path.cwd()),
            },
        )
        event_log = PostgresEventLog(app_parts.state.engine)
        events = await event_log.query(EventFilter(request_id=payload["error"]["request_id"]))
        return status, payload, events

    status, payload, events = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["message"] == "model profile purpose is not valid for selected loop"
    assert EventType.LOOP_SELECTION_STARTED in [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_FAILED in [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_COMPLETED not in [event.event_type for event in events]
    failed = next(event for event in events if event.event_type is EventType.LOOP_SELECTION_FAILED)
    assert failed.payload["request_plan_status"] == "invalid_override"
    assert (
        failed.payload["request_plan_reason_code"]
        == "request_plan_model_profile_invalid_for_selected_loop"
    )


def test_invalid_loop_strategy_emits_loop_selection_failed_event(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, payload = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-selection-invalid-mode",
                "content": "hello",
                "sensitivity": "project",
                "loop_strategy": "unsafe_loop",
            },
        )
        event_log = PostgresEventLog(app_parts.state.engine)
        events = await event_log.query(EventFilter(request_id=payload["error"]["request_id"]))
        return status, payload, events

    status, payload, events = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["message"] == "loop strategy is not configured"
    assert EventType.LOOP_SELECTION_STARTED in [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_FAILED in [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_COMPLETED not in [event.event_type for event in events]
    started = next(event for event in events if event.event_type is EventType.LOOP_SELECTION_STARTED)
    failed = next(event for event in events if event.event_type is EventType.LOOP_SELECTION_FAILED)
    assert started.payload["requested_mode"] == "invalid_override"
    assert failed.payload["requested_mode"] == "invalid_override"
    assert failed.payload["request_plan_status"] == "invalid_override"
    assert failed.payload["request_plan_reason_code"] == "request_plan_invalid_override"


def test_failed_pre_submit_selection_does_not_collide_with_later_accepted_request(
    app_parts,
) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        status, failed = await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-retry-after-selection-failure",
                "content": "show cpu usage",
                "sensitivity": "project",
                "model_profile": "local_embedding",
            },
        )
        assert status == 400
        status, accepted = await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-retry-after-selection-failure",
                "content": "hello",
                "sensitivity": "project",
            },
        )
        assert status == 202
        event_log = PostgresEventLog(app_parts.state.engine)
        failed_events = await event_log.query(
            EventFilter(request_id=failed["error"]["request_id"]),
        )
        accepted_events = await event_log.query(EventFilter(request_id=accepted["request_id"]))
        return failed, accepted, failed_events, accepted_events

    failed, accepted, failed_events, accepted_events = asyncio.run(scenario())

    assert failed["error"]["request_id"] != accepted["request_id"]
    assert EventType.LOOP_SELECTION_FAILED in [event.event_type for event in failed_events]
    assert EventType.LOOP_SELECTION_FAILED not in [
        event.event_type for event in accepted_events
    ]


def test_model_profile_matches_selected_loop(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-model-profile",
                "content": "show cpu usage",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


def test_tools_disabled_does_not_silently_fallback_to_chat(app_parts) -> None:
    async def scenario():
        settings = ConfigLoader(Path("config")).load("test")
        disabled_settings = replace(
            settings,
            policy=replace(settings.policy, tools_enabled=False),
        )
        engine = app_parts.state.engine
        app = create_app(
            conversation_store=PostgresConversationStore(engine),
            memory_store=PostgresMemoryStore(
                engine=engine,
                settings=disabled_settings,
                policy=ConfigPolicyEngine(disabled_settings),
                embedding_port=FakeEmbeddingProvider(),
            ),
            settings=disabled_settings,
        )
        conversation = await _create_conversation(app)
        status, accepted = await _request(
            app,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-tools-disabled-auto",
                "content": "show cpu usage",
                "sensitivity": "project",
            },
        )
        assert status == 202
        return await _request(app, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"], tool_policy="disabled")


def test_tool_loop_budget_without_tool_calls_rejects_before_request_persistence(app_parts) -> None:
    async def scenario():
        settings = ConfigLoader(Path("config")).load("test")
        budget = replace(settings.runtime_budgets["tool_react_loop"], max_tool_calls=0)
        budget_settings = replace(
            settings,
            runtime_budgets={**settings.runtime_budgets, "tool_react_loop": budget},
        )
        engine = app_parts.state.engine
        event_log = PostgresEventLog(engine)
        policy = ConfigPolicyEngine(budget_settings, event_log=event_log)
        conversation_store = PostgresConversationStore(engine)
        app = create_app(
            conversation_store=conversation_store,
            memory_store=PostgresMemoryStore(
                engine=engine,
                settings=budget_settings,
                policy=policy,
                embedding_port=FakeEmbeddingProvider(),
            ),
            content_store=PostgresContentStore(engine=engine, embedding_port=FakeEmbeddingProvider()),
            settings=budget_settings,
            event_log=event_log,
            policy=policy,
        )
        conversation = await _create_conversation(app)
        status, accepted = await _request(
            app,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-tool-loop-budget-zero",
                "content": "show cpu usage",
                "sensitivity": "project",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        request = await conversation_store.get_assistant_request(accepted["request_id"])
        events = await event_log.query(EventFilter(request_id=accepted["request_id"]))
        return status, accepted, request, events

    status, payload, request, events = asyncio.run(scenario())

    assert status == 202
    assert payload["request_id"]
    assert request is not None
    assert request.metadata["agent_tool_policy"] == "disabled"
    assert EventType.LOOP_SELECTION_STARTED in [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_COMPLETED in [event.event_type for event in events]
    assert EventType.LOOP_SELECTION_FAILED not in [event.event_type for event in events]


def test_tool_auto_without_working_directory_filters_scope_bound_tools(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-without-working-directory",
                "content": "show cpu usage",
                "sensitivity": "project",
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])
    assert "datetime.now" in payload["metadata"]["agent_allowed_tool_names"]
    assert "tool.system.read.resources" not in payload["metadata"]["agent_allowed_tool_names"]


def test_explicit_tools_mode_is_rejected_when_tools_disabled(app_parts) -> None:
    async def scenario():
        settings = ConfigLoader(Path("config")).load("test")
        disabled_settings = replace(
            settings,
            policy=replace(settings.policy, tools_enabled=False),
        )
        engine = app_parts.state.engine
        app = create_app(
            conversation_store=PostgresConversationStore(engine),
            memory_store=PostgresMemoryStore(
                engine=engine,
                settings=disabled_settings,
                policy=ConfigPolicyEngine(disabled_settings),
                embedding_port=FakeEmbeddingProvider(),
            ),
            settings=disabled_settings,
        )
        conversation = await _create_conversation(app)
        return await _request(
            app,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-tools-disabled-mode",
                "content": "hello",
                "sensitivity": "project",
                "loop_strategy": "tools",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "tool loop is disabled by policy"


def test_request_metadata_uses_main_model_profile_for_agent_loop(
    app_parts,
) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        status, accepted = await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-auto-profile-after-selection",
                "content": "what is listening on port 8080?",
                "sensitivity": "project",
                "model_profile": "local_main",
                "working_directory": str(Path.cwd()),
            },
        )
        assert status == 202
        return await _request(app_parts, "GET", f"/v1/requests/{accepted['request_id']}")

    status, payload = asyncio.run(scenario())

    assert status == 200
    _assert_agent_loop_request_metadata(payload["metadata"])


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


def test_conflicting_client_message_id_with_different_runtime_options_returns_409(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        path = f"/v1/conversations/{conversation['conversation_id']}/messages"
        await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-conflict-runtime-options",
                "content": "same",
                "sensitivity": "project",
                "loop_strategy": "memory_augmented_answer",
            },
        )
        return await _request(
            app_parts,
            "POST",
            path,
            {
                "client_message_id": "client-conflict-runtime-options",
                "content": "same",
                "sensitivity": "project",
                "loop_strategy": "tool_react_loop",
                "model_profile": "local_structured",
            },
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


def test_post_message_rejects_client_permission_mode(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-permission-mode",
                "content": "hello",
                "sensitivity": "project",
                "permission_mode": "developer_local",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["details"]["errors"][0]["type"] == "extra_forbidden"


def test_post_message_rejects_unknown_loop_strategy(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-unknown-loop",
                "content": "hello",
                "sensitivity": "project",
                "loop_strategy": "unsafe_loop",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "loop strategy is not configured"


def test_post_message_rejects_tool_loop_when_tools_disabled(app_parts) -> None:
    async def scenario():
        settings = ConfigLoader(Path("config")).load("test")
        disabled_settings = replace(
            settings,
            policy=replace(settings.policy, tools_enabled=False),
        )
        engine = app_parts.state.engine
        app = create_app(
            conversation_store=PostgresConversationStore(engine),
            memory_store=PostgresMemoryStore(
                engine=engine,
                settings=disabled_settings,
                policy=ConfigPolicyEngine(disabled_settings),
                embedding_port=FakeEmbeddingProvider(),
            ),
            settings=disabled_settings,
        )
        conversation = await _create_conversation(app)
        return await _request(
            app,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-tools-disabled-loop",
                "content": "hello",
                "sensitivity": "project",
                "loop_strategy": "tool_react_loop",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "tool loop is disabled by policy"


def test_post_message_rejects_unauthorized_model_profile(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-cloud-model",
                "content": "hello",
                "sensitivity": "project",
                "model_profile": "cloud_reasoning",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "model profile is not available for this request"


def test_post_message_rejects_model_profile_with_wrong_purpose(app_parts) -> None:
    async def scenario():
        conversation = await _create_conversation(app_parts)
        return await _request(
            app_parts,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-embedding-model",
                "content": "hello",
                "sensitivity": "project",
                "model_profile": "local_embedding",
            },
        )

    status, payload = asyncio.run(scenario())

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "model profile purpose is not valid for selected loop"


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
