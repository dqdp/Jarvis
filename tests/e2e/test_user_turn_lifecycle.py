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
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.domain.events import EventType
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.models.router import ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.agent_runtime import AgentRuntime
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository


pytestmark = pytest.mark.e2e


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_e2e(database_url: str) -> None:
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


def _run_user_turn_lifecycle():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        await _truncate_e2e(database_url)
        engine = create_database_engine(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        policy = ConfigPolicyEngine(settings)
        conversation_store = PostgresConversationStore(engine)
        event_log = PostgresEventLog(engine)
        memory_store = PostgresMemoryStore(
            engine=engine,
            settings=settings,
            policy=policy,
            embedding_port=FakeEmbeddingProvider(),
        )
        invocations = PostgresModelInvocationRepository(engine)
        router = ModelRouter(
            settings=settings,
            policy=policy,
            invocation_repository=invocations,
            event_log=event_log,
            providers={
                "local_openai_compatible": FakeModelProvider(stream_tokens=["OK"]),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        runtime = AgentRuntime(
            conversation_store=conversation_store,
            context_assembler=DeterministicContextAssembler(
                conversation_store=conversation_store,
                memory_read=memory_store,
                event_log=event_log,
                policy=policy,
            ),
            model_router=router,
            event_log=event_log,
            settings=settings,
        )
        app = create_app(
            conversation_store=conversation_store,
            memory_store=memory_store,
            settings=settings,
            runtime=runtime,
            event_log=event_log,
        )
        assert isinstance(app, FastAPI)
        try:
            _, conversation_raw = await _request(
                app,
                "POST",
                "/v1/conversations",
                {"title": "e2e", "active_project_namespace": "project.personal_assistant"},
            )
            conversation = json.loads(conversation_raw)
            await _request(
                app,
                "POST",
                "/v1/memories",
                {
                    "namespace": "project.personal_assistant",
                    "memory_type": "fact",
                    "content": "stream lifecycle memory",
                    "sensitivity": "project",
                },
            )
            _, message_raw = await _request(
                app,
                "POST",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
                {
                    "client_message_id": "client-e2e",
                    "content": "stream lifecycle memory",
                    "sensitivity": "project",
                },
            )
            submitted = json.loads(message_raw)
            for _ in range(100):
                _, request_raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}")
                if json.loads(request_raw)["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
            _, stream_raw = await _request(
                app,
                "GET",
                f"/v1/requests/{submitted['request_id']}/stream",
            )
            _, request_raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}")
            _, messages_raw = await _request(
                app,
                "GET",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
            )
            events = await event_log.query(EventFilter(request_id=submitted["request_id"]))
            invocation_rows = await invocations.list_recent(limit=10)
            return (
                submitted,
                _sse_events(stream_raw),
                json.loads(request_raw),
                json.loads(messages_raw),
                events,
                invocation_rows,
            )
        finally:
            await engine.dispose()

    return asyncio.run(scenario())


def test_e2e_user_turn_lifecycle_with_memory_and_fake_model() -> None:
    submitted, stream_events, request_status, messages, events, invocations = (
        _run_user_turn_lifecycle()
    )

    assert submitted["request_id"]
    assert stream_events[-1] == "request.processing.completed"
    assert request_status["status"] == "completed"
    assert messages["messages"][-1]["role"] == "assistant"
    assert messages["messages"][-1]["content"] == "OK"
    assert len(invocations) == 1
    assert all(event.request_id == submitted["request_id"] for event in events)
    context_event = next(event for event in events if event.event_type == EventType.CONTEXT_ASSEMBLED)
    memory_event = next(event for event in events if event.event_type == EventType.MEMORY_RETRIEVED)
    model_policy_event = next(
        event
        for event in events
        if event.event_type == EventType.POLICY_DECISION_RECORDED
        and event.payload.get("source_ref") == "model_request:local_main"
    )
    assert context_event.payload["used_memory_ids"]
    assert memory_event.payload["used_memory_ids"] == context_event.payload["used_memory_ids"]
    assert context_event.causation_id == memory_event.event_id
    assert model_policy_event.payload["allowed"] is True


def test_user_turn_lifecycle_still_uses_memory_augmented_answer() -> None:
    _, _, _, _, events, _ = _run_user_turn_lifecycle()

    loop_started = next(event for event in events if event.event_type == EventType.AGENT_LOOP_STARTED)
    loop_completed = next(
        event for event in events if event.event_type == EventType.AGENT_LOOP_COMPLETED
    )
    assert loop_started.payload["strategy_name"] == "memory_augmented_answer"
    assert loop_completed.payload["used_model_calls"] == 1
    assert loop_completed.payload["used_tool_calls"] == 0


def test_no_tool_events_are_emitted_for_memory_augmented_answer() -> None:
    _, _, _, _, events, _ = _run_user_turn_lifecycle()

    assert all(not event.event_type.value.startswith("tool.") for event in events)
