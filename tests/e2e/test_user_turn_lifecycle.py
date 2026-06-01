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
from assistant_core.runtime.loops import LoopStrategyRegistry, MemoryAugmentedAnswerLoop
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository
from assistant_core.tools.builtin import datetime_now_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolAdapter, ToolRegistry


pytestmark = [pytest.mark.e2e, pytest.mark.db]


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


def _run_user_turn_lifecycle(
    *,
    content: str = "stream lifecycle memory",
    client_message_id: str = "client-e2e",
    chat_response: str = "OK",
    structured_text_responses: list[str] | None = None,
    tool_adapters: list[ToolAdapter] | None = None,
):
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
                "local_openai_compatible": FakeModelProvider(
                    chat_response=chat_response,
                    stream_tokens=[chat_response],
                    structured_text_responses=structured_text_responses
                    or ['{"action":"final_answer"}'],
                ),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        context_assembler = DeterministicContextAssembler(
            conversation_store=conversation_store,
            memory_read=memory_store,
            event_log=event_log,
            policy=policy,
        )
        tool_gateway = ToolGateway(
            registry=ToolRegistry(tool_adapters or []),
            policy=policy,
            event_log=event_log,
        )
        runtime = AgentRuntime(
            conversation_store=conversation_store,
            context_assembler=context_assembler,
            model_router=router,
            event_log=event_log,
            settings=settings,
            loop_strategy_registry=LoopStrategyRegistry(
                [
                    MemoryAugmentedAnswerLoop(
                        conversation_store=conversation_store,
                        context_assembler=context_assembler,
                        model_router=router,
                        event_log=event_log,
                    ),
                    ToolReactLoop(
                        conversation_store=conversation_store,
                        context_assembler=context_assembler,
                        model_router=router,
                        event_log=event_log,
                        tool_gateway=tool_gateway,
                    ),
                ],
            ),
        )
        app = create_app(
            conversation_store=conversation_store,
            memory_store=memory_store,
            settings=settings,
            runtime=runtime,
            event_log=event_log,
            policy=policy,
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
                    "content": content,
                    "sensitivity": "project",
                },
            )
            _, message_raw = await _request(
                app,
                "POST",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
                {
                    "client_message_id": client_message_id,
                    "content": content,
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
    assert len(invocations) == 2
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


def test_user_turn_lifecycle_uses_agent_loop_request_metadata() -> None:
    _, _, _, _, events, _ = _run_user_turn_lifecycle()

    loop_started = next(event for event in events if event.event_type == EventType.AGENT_LOOP_STARTED)
    loop_completed = next(
        event for event in events if event.event_type == EventType.AGENT_LOOP_COMPLETED
    )
    assert loop_started.payload["strategy_name"] == "tool_react_loop"
    assert loop_completed.payload["used_model_calls"] == 2
    assert loop_completed.payload["used_tool_calls"] == 0


def test_no_tool_events_are_emitted_when_agent_loop_model_skips_tools() -> None:
    _, _, _, _, events, _ = _run_user_turn_lifecycle()

    assert all(not event.event_type.value.startswith("tool.") for event in events)


def test_transcript_like_api_turn_uses_agent_loop_lifecycle() -> None:
    submitted, stream_events, request_status, messages, events, invocations = (
        _run_user_turn_lifecycle(
            content="Джарвис, пожалуйста, кратко ответь по памяти",
            client_message_id="client-transcript-chat",
            chat_response="transcript OK",
        )
    )

    assert submitted["request_id"]
    assert stream_events[-1] == "request.processing.completed"
    assert request_status["status"] == "completed"
    assert messages["messages"][-1]["content"] == "transcript OK"
    loop_started = next(event for event in events if event.event_type == EventType.AGENT_LOOP_STARTED)
    assert loop_started.payload["strategy_name"] == "tool_react_loop"
    assert len(invocations) == 2


def test_transcript_like_tool_turn_uses_toolgateway() -> None:
    _, _, request_status, messages, events, invocations = _run_user_turn_lifecycle(
        content="Джарвис, сколько времени сейчас?",
        client_message_id="client-transcript-tool",
        chat_response="tool transcript OK",
        structured_text_responses=[
            '{"action":"tool_call","tool_name":"datetime.now","arguments":{}}',
            '{"action":"final_answer"}',
        ],
        tool_adapters=[datetime_now_tool()],
    )

    assert request_status["status"] == "completed"
    assert messages["messages"][-1]["content"] == "tool transcript OK"
    assert EventType.TOOL_CALL_COMPLETED in [event.event_type for event in events]
    loop_completed = next(
        event for event in events if event.event_type == EventType.AGENT_LOOP_COMPLETED
    )
    assert loop_completed.payload["used_tool_calls"] == 1
    assert len(invocations) == 3
