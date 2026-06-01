from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy import text

from assistant_core.api.app import create_app
from assistant_core.config.settings import ConfigLoader
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.domain.conversations import (
    CreateConversationCommand,
    MessageSubmissionCommand,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.messages import MessageRole
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.models.router import ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.memory import MemoryRetrievalError
from assistant_core.runtime.agent_runtime import AgentRuntime
from assistant_core.runtime.agent_runtime import RuntimeStreamEvent
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository
from assistant_core.tools.builtin import datetime_now_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry


pytestmark = [pytest.mark.contract, pytest.mark.db]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_stream(database_url: str) -> None:
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


class EmptyMemoryRead:
    async def retrieve(self, query):
        return []


class BlockingStreamProvider(FakeModelProvider):
    def __init__(self) -> None:
        super().__init__(stream_tokens=["slow"])
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def stream_chat(self, request):
        self.stream_calls += 1
        assert self.started is not None
        assert self.release is not None
        self.started.set()
        await self.release.wait()
        yield "slow"


class CancellationResistantProvider(FakeModelProvider):
    def __init__(self) -> None:
        super().__init__(stream_tokens=["late"])
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def stream_chat(self, request):
        self.stream_calls += 1
        assert self.started is not None
        assert self.release is not None
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        yield "late"


class BrokenRuntime:
    async def stream_turn(self, command):
        raise RuntimeError("unexpected background failure")
        yield


class ToolPhaseRuntime:
    async def stream_turn(self, command):
        yield RuntimeStreamEvent(
            EventType.REQUEST_PROCESSING_STARTED.value,
            {"request_id": command.request_id, "event_id": "runtime-started"},
        )
        yield RuntimeStreamEvent(
            EventType.CONTEXT_ASSEMBLY_STARTED.value,
            {"request_id": command.request_id, "event_id": "context-started"},
        )
        yield RuntimeStreamEvent(
            EventType.MEMORY_RETRIEVED.value,
            {"request_id": command.request_id, "event_id": "memory-retrieved"},
        )
        yield RuntimeStreamEvent(
            EventType.CONTENT_RETRIEVED.value,
            {
                "request_id": command.request_id,
                "event_id": "content-retrieved",
                "retrieved_content_refs": [{"chunk_id": "chunk-1", "content_hash": "hash"}],
                "full_content_stored": False,
            },
        )
        yield RuntimeStreamEvent("token", {"delta": "tool answer"})
        yield RuntimeStreamEvent(
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {
                "request_id": command.request_id,
                "event_id": "runtime-completed",
                "assistant_message_id": "message-tool",
            },
        )


class MemoryLoopRuntimeAdapter:
    def __init__(self, delegate: AgentRuntime) -> None:
        self._delegate = delegate

    async def stream_turn(self, command):
        memory_command = replace(
            command,
            loop_strategy=LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value,
        )
        async for event in self._delegate.stream_turn(memory_command):
            yield event


@pytest.fixture
def stream_parts():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_stream(database_url))
    engine = create_database_engine(database_url)
    settings = ConfigLoader(Path("config")).load("test")
    conversation_store = PostgresConversationStore(engine)
    event_log = PostgresEventLog(engine)
    memory_store = PostgresMemoryStore(
        engine=engine,
        settings=settings,
        policy=ConfigPolicyEngine(settings),
    )
    apps: list[FastAPI] = []

    def make_app(
        provider: FakeModelProvider,
        *,
        settings_override=None,
        runtime_override=None,
        tool_gateway_override=None,
        use_memory_loop_adapter: bool = True,
    ):
        app_settings = settings_override or settings
        policy = ConfigPolicyEngine(app_settings)
        router = ModelRouter(
            settings=app_settings,
            policy=policy,
            invocation_repository=PostgresModelInvocationRepository(engine),
            event_log=event_log,
            providers={
                "local_openai_compatible": provider,
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        runtime = AgentRuntime(
            conversation_store=conversation_store,
            context_assembler=DeterministicContextAssembler(
                conversation_store=conversation_store,
                memory_read=EmptyMemoryRead(),
                event_log=event_log,
                policy=policy,
            ),
            model_router=router,
            event_log=event_log,
            settings=app_settings,
            tool_gateway=tool_gateway_override,
        )
        selected_runtime = (
            runtime_override
            or (MemoryLoopRuntimeAdapter(runtime) if use_memory_loop_adapter else runtime)
        )
        app = create_app(
            conversation_store=conversation_store,
            memory_store=memory_store,
            settings=app_settings,
            runtime=selected_runtime,
            event_log=event_log,
            policy=policy,
        )
        app.state.test_engine = engine
        app.state.test_conversation_store = conversation_store
        app.state.test_model_invocations = PostgresModelInvocationRepository(engine)
        apps.append(app)
        assert isinstance(app, FastAPI)
        return app

    try:
        yield make_app, event_log
    finally:
        async def cleanup() -> None:
            for app in apps:
                manager = getattr(app.state, "request_execution_manager", None)
                if manager is not None:
                    await manager.shutdown()
            await engine.dispose()

        asyncio.run(cleanup())


async def _request(app, method: str, path: str, body: dict[str, Any] | None = None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        kwargs: dict[str, Any] = {"json": body} if body is not None else {}
        response = await client.request(method, path, headers={"accept": "text/event-stream"}, **kwargs)
    return response.status_code, response.text


async def _json_request(app, method: str, path: str, body: dict[str, Any] | None = None):
    status, raw = await _request(app, method, path, body)
    return status, json.loads(raw) if raw else None


async def _accepted_message(
    app,
    *,
    client_message_id: str = "client-sse",
    content: str = "hello stream",
    sensitivity: str = "project",
    loop_strategy: str | None = LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value,
):
    _, conversation_raw = await _request(
        app,
        "POST",
        "/v1/conversations",
        {"title": "sse", "active_project_namespace": "project.personal_assistant"},
    )
    conversation = json.loads(conversation_raw)
    body: dict[str, Any] = {
        "client_message_id": client_message_id,
        "content": content,
        "sensitivity": sensitivity,
    }
    if loop_strategy is not None:
        body["loop_strategy"] = loop_strategy
    _, message_raw = await _request(
        app,
        "POST",
        f"/v1/conversations/{conversation['conversation_id']}/messages",
        body,
    )
    return json.loads(message_raw)


def _sse_events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        lines = block.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


async def _wait_request_status(app, request_id: str, status: str, *, attempts: int = 100):
    for _ in range(attempts):
        _, payload = await _json_request(app, "GET", f"/v1/requests/{request_id}")
        if payload["status"] == status:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError(f"request {request_id} did not reach status {status}")


def test_sse_stream_emits_request_started(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(app)
        status, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return status, _sse_events(raw)

    status, events = asyncio.run(scenario())

    assert status == 200
    assert events[0][0] == "request.processing.started"


def test_live_sse_stream_does_not_expose_internal_loop_events(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-live-loop-events",
        )
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        persisted = await event_log.query(EventFilter(request_id=submitted["request_id"]))
        return _sse_events(raw), persisted

    live_events, persisted_events = asyncio.run(scenario())

    assert all(not event.startswith("agent.loop.") for event, _ in live_events)
    assert any(
        event.event_type == EventType.AGENT_LOOP_STARTED
        for event in persisted_events
    )
    assert any(
        event.event_type == EventType.AGENT_LOOP_COMPLETED
        for event in persisted_events
    )


def test_request_executes_without_opening_stream(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = FakeModelProvider(stream_tokens=["A"])
        app = make_app(provider)
        submitted = await _accepted_message(
            app,
            client_message_id="client-no-stream",
            content="run without subscriber",
        )
        request = await _wait_request_status(app, submitted["request_id"], "completed")
        return request, provider.stream_calls

    request, stream_calls = asyncio.run(scenario())

    assert request["status"] == "completed"
    assert stream_calls == 1


def test_sse_stream_emits_token_events(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A", "B"]))
        submitted = await _accepted_message(app)
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return _sse_events(raw)

    events = asyncio.run(scenario())

    assert [data["delta"] for event, data in events if event == "token"] == ["A", "B"]


def test_sse_stream_emits_memory_retrieved_event(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-memory-sse",
            content="memory event",
        )
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return _sse_events(raw)

    events = asyncio.run(scenario())

    assert "memory.retrieved" in [event for event, _ in events]


def test_live_sse_stream_exposes_tool_phase_events_without_losing_token(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(
            FakeModelProvider(stream_tokens=["unused"]),
            runtime_override=ToolPhaseRuntime(),
        )
        _, conversation_raw = await _request(
            app,
            "POST",
            "/v1/conversations",
            {"title": "tool sse", "active_project_namespace": "project.personal_assistant"},
        )
        conversation = json.loads(conversation_raw)
        status, message_raw = await _request(
            app,
            "POST",
            f"/v1/conversations/{conversation['conversation_id']}/messages",
            {
                "client_message_id": "client-tool-phase-sse",
                "content": "what is listening on port 8080?",
                "sensitivity": "project",
                "loop_strategy": "auto",
                "working_directory": str(Path.cwd()),
            },
        )
        submitted = json.loads(message_raw)
        assert status == 202, submitted
        stream_status, raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return status, stream_status, _sse_events(raw)

    submit_status, stream_status, events = asyncio.run(scenario())
    event_types = [event for event, _ in events]

    assert submit_status == 202
    assert stream_status == 200
    assert event_types[0] == EventType.REQUEST_PROCESSING_STARTED.value
    assert EventType.LOOP_SELECTION_STARTED.value in event_types
    assert EventType.LOOP_SELECTION_COMPLETED.value in event_types
    assert EventType.CONTEXT_ASSEMBLY_STARTED.value in event_types
    assert EventType.MEMORY_RETRIEVED.value in event_types
    assert EventType.CONTENT_RETRIEVED.value in event_types
    assert event_types.index("token") < event_types.index(EventType.REQUEST_PROCESSING_COMPLETED.value)
    content_event = next(data for event, data in events if event == EventType.CONTENT_RETRIEVED.value)
    assert content_event["hit_count"] == 1
    assert "retrieved_content_refs" not in content_event


def test_running_request_owned_by_another_manager_is_not_marked_orphaned(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = BlockingStreamProvider()
        app_one = make_app(provider)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app_one,
            client_message_id="client-orphan-running",
            content="orphan running",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        app_two = make_app(FakeModelProvider(stream_tokens=["SHOULD_NOT_RUN"]))
        stream_task = asyncio.create_task(
            _request(app_two, "GET", f"/v1/requests/{submitted['request_id']}/stream"),
        )
        await asyncio.sleep(0.02)
        provider.release.set()
        status, raw = await asyncio.wait_for(stream_task, timeout=1)
        return status, _sse_events(raw)

    status, events = asyncio.run(scenario())

    assert status == 200
    assert events[-1][0] == "request.processing.completed"


def test_background_runtime_exception_marks_request_failed(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(
            FakeModelProvider(stream_tokens=["SHOULD_NOT_RUN"]),
            runtime_override=BrokenRuntime(),
        )
        submitted = await _accepted_message(
            app,
            client_message_id="client-background-failure",
            content="background failure",
        )
        request = await _wait_request_status(app, submitted["request_id"], "failed")
        stream_status, stream_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return request, stream_status, _sse_events(stream_raw)

    request, stream_status, events = asyncio.run(scenario())

    assert request["error"]["code"] == "background_task_failed"
    assert stream_status == 200
    assert events[-1][0] == "request.processing.failed"


def test_running_request_stream_reconnect_subscribes_without_rerun(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = BlockingStreamProvider()
        app = make_app(provider)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app,
            client_message_id="client-running-reconnect",
            content="keep running",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        first_status, first_payload = await _json_request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}",
        )
        provider.release.set()
        stream_status, stream_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return first_status, first_payload, stream_status, _sse_events(stream_raw), provider.stream_calls

    first_status, first_payload, stream_status, events, stream_calls = asyncio.run(scenario())

    assert first_status == 200
    assert first_payload["status"] == "running"
    assert stream_status == 200
    assert events[-1][0] == "request.processing.completed"
    assert stream_calls == 1


def test_stream_emits_heartbeat_while_waiting_for_running_request(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = BlockingStreamProvider()
        base_settings = ConfigLoader(Path("config")).load("test")
        heartbeat_settings = replace(
            base_settings,
            api=replace(base_settings.api, sse_heartbeat_seconds=0),
        )
        app = make_app(provider, settings_override=heartbeat_settings)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app,
            client_message_id="client-heartbeat",
            content="wait with heartbeat",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        stream_task = asyncio.create_task(
            _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream"),
        )
        await asyncio.sleep(0.02)
        provider.release.set()
        status, raw = await stream_task
        return status, _sse_events(raw)

    status, events = asyncio.run(scenario())

    assert status == 200
    assert "heartbeat" in [event for event, _ in events]


def test_sse_stream_emits_assistant_message_created(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(app)
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return _sse_events(raw)

    assert "assistant.message.created" in [event for event, _ in asyncio.run(scenario())]


def test_tool_react_sse_stream_emits_model_tool_and_assistant_lifecycle_events(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        base_settings = ConfigLoader(Path("config")).load("test")
        tool_budget = base_settings.runtime_budgets[LoopStrategyName.TOOL_REACT_LOOP.value]
        app_settings = replace(
            base_settings,
            runtime_budgets={
                **base_settings.runtime_budgets,
                LoopStrategyName.TOOL_REACT_LOOP.value: replace(
                    tool_budget,
                    max_model_calls=3,
                    max_tool_calls=1,
                ),
            },
        )
        provider = FakeModelProvider(
            chat_response="agent loop response",
            structured_text_responses=[
                '{"action":"tool_call","tool_name":"datetime.now","arguments":{}}',
            ],
        )
        policy = ConfigPolicyEngine(app_settings)
        tool_gateway = ToolGateway(
            registry=ToolRegistry([datetime_now_tool()]),
            policy=policy,
            event_log=event_log,
        )
        app = make_app(
            provider,
            settings_override=app_settings,
            tool_gateway_override=tool_gateway,
            use_memory_loop_adapter=False,
        )
        submitted = await _accepted_message(
            app,
            client_message_id="client-tool-react-model-stream",
            content="echo hello through tool loop",
            loop_strategy=LoopStrategyName.TOOL_REACT_LOOP.value,
        )
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return _sse_events(raw), provider

    events, provider = asyncio.run(scenario())
    event_types = [event for event, _ in events]

    assert provider.structured_calls == 1
    assert provider.chat_calls == 1
    assert EventType.MODEL_REQUEST_CREATED.value in event_types
    assert EventType.MODEL_RESPONSE_RECEIVED.value in event_types
    assert EventType.TOOL_CALL_STARTED.value in event_types
    assert EventType.TOOL_CALL_COMPLETED.value in event_types
    assert EventType.ASSISTANT_MESSAGE_CREATED.value in event_types
    assert event_types.count(EventType.REQUEST_PROCESSING_COMPLETED.value) == 1
    assert event_types.index(EventType.MODEL_REQUEST_CREATED.value) < event_types.index(
        EventType.MODEL_RESPONSE_RECEIVED.value,
    )
    assert event_types.index(EventType.MODEL_RESPONSE_RECEIVED.value) < event_types.index(
        EventType.TOOL_CALL_STARTED.value,
    )
    assert event_types.index(EventType.TOOL_CALL_COMPLETED.value) < event_types.index(
        EventType.ASSISTANT_MESSAGE_CREATED.value,
    )
    assert event_types.index(EventType.ASSISTANT_MESSAGE_CREATED.value) < event_types.index(
        EventType.REQUEST_PROCESSING_COMPLETED.value,
    )


def test_sse_stream_emits_request_completed(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(app)
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return _sse_events(raw)

    events = asyncio.run(scenario())

    assert events[-1][0] == "request.processing.completed"


def test_sse_lifecycle_records_model_policy_decision(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-model-policy-audit",
            content="audit model policy",
        )
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return await event_log.query(EventFilter(request_id=submitted["request_id"]))

    events = asyncio.run(scenario())

    policy_event = next(
        event
        for event in events
        if event.event_type == EventType.POLICY_DECISION_RECORDED
        and event.payload.get("source_ref") == "model_request:local_main"
    )
    assert policy_event.payload["source_ref"] == "model_request:local_main"
    assert policy_event.payload["allowed"] is True


def test_token_events_not_persisted(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A", "B"]))
        submitted = await _accepted_message(app)
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        persisted = await event_log.query(EventFilter(request_id=submitted["request_id"]))
        return persisted

    persisted = asyncio.run(scenario())

    assert "token" not in [event.event_type.value for event in persisted]


def test_failed_request_emits_failure_event(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        app = make_app(FakeModelProvider(fail_stream_times=1))
        submitted = await _accepted_message(app)
        _, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        return _sse_events(raw)

    events = asyncio.run(scenario())

    assert events[-1][0] == "request.processing.failed"
    assert set(events[-1][1]["error"]) == {"code", "message", "request_id", "details"}


def test_cancel_running_request_marks_cancelled_and_emits_terminal_event(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = BlockingStreamProvider()
        app = make_app(provider)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app,
            client_message_id="client-cancel",
            content="cancel me",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        cancel_status, cancel_payload = await _json_request(
            app,
            "POST",
            f"/v1/requests/{submitted['request_id']}/cancel",
        )
        request = await _wait_request_status(app, submitted["request_id"], "cancelled")
        stream_status, stream_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return cancel_status, cancel_payload, request, stream_status, _sse_events(stream_raw)

    cancel_status, cancel_payload, request, stream_status, events = asyncio.run(scenario())

    assert cancel_status == 202
    assert cancel_payload["status"] == "cancelled"
    assert request["status"] == "cancelled"
    assert stream_status == 200
    assert events[-1][0] == "request.processing.cancelled"


def test_cancel_running_request_is_bounded_when_provider_resists_cancellation(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = CancellationResistantProvider()
        app = make_app(provider)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app,
            client_message_id="client-resistant-cancel",
            content="resist cancel",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        cancel_task = asyncio.create_task(
            _json_request(app, "POST", f"/v1/requests/{submitted['request_id']}/cancel"),
        )
        try:
            cancel_status, cancel_payload = await asyncio.wait_for(cancel_task, timeout=0.5)
        finally:
            provider.release.set()
            with suppress(asyncio.CancelledError, RuntimeError):
                await cancel_task
            await app.state.request_execution_manager.shutdown()
        request = await _wait_request_status(app, submitted["request_id"], "cancelled")
        return cancel_status, cancel_payload, request

    cancel_status, cancel_payload, request = asyncio.run(scenario())

    assert cancel_status == 202
    assert cancel_payload["status"] == "cancelled"
    assert request["status"] == "cancelled"


def test_cancelled_request_does_not_persist_late_assistant_side_effects(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        provider = CancellationResistantProvider()
        app = make_app(provider)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app,
            client_message_id="client-cancel-no-late-effects",
            content="cancel before late token",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        await _json_request(app, "POST", f"/v1/requests/{submitted['request_id']}/cancel")
        provider.release.set()
        await asyncio.sleep(0.05)
        _, messages_raw = await _request(
            app,
            "GET",
            f"/v1/conversations/{submitted['conversation_id']}/messages",
        )
        invocations = await app.state.test_model_invocations.list_recent(limit=10)
        events = await event_log.query(EventFilter(request_id=submitted["request_id"]))
        return json.loads(messages_raw), invocations, events

    messages_payload, invocations, events = asyncio.run(scenario())

    assert [
        message["role"] for message in messages_payload["messages"]
    ] == [MessageRole.USER.value]
    assert all(
        event.event_type not in {
            EventType.MODEL_RESPONSE_RECEIVED,
            EventType.ASSISTANT_MESSAGE_CREATED,
            EventType.REQUEST_PROCESSING_COMPLETED,
        }
        for event in events
    )
    assert all(invocation.status != "running" for invocation in invocations)


def test_cancelled_stream_invocation_is_finalized(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = BlockingStreamProvider()
        app = make_app(provider)
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        submitted = await _accepted_message(
            app,
            client_message_id="client-cancel-finalizes-invocation",
            content="cancel invocation",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        await _json_request(app, "POST", f"/v1/requests/{submitted['request_id']}/cancel")
        await asyncio.sleep(0.05)
        return await app.state.test_model_invocations.list_recent(limit=10)

    invocations = asyncio.run(scenario())

    assert len(invocations) == 1
    assert invocations[0].status == "cancelled"


def test_secret_message_fails_without_provider_call(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = FakeModelProvider(stream_tokens=["SHOULD_NOT_STREAM"])
        app = make_app(provider)
        submitted = await _accepted_message(
            app,
            client_message_id="client-secret",
            content="do not reveal this secret",
            sensitivity="secret",
        )
        stream_status, raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        request_status, request_raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}")
        return stream_status, _sse_events(raw), request_status, json.loads(request_raw), provider.stream_calls

    stream_status, events, request_status, request_payload, stream_calls = asyncio.run(scenario())

    assert stream_status == 200
    assert request_status == 200
    assert events[-1][0] == "request.processing.failed"
    assert request_payload["status"] == "failed"
    assert stream_calls == 0


def test_reconnect_stream_does_not_expose_internal_policy_audit_events(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["SHOULD_NOT_STREAM"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-secret-reconnect",
            content="secret reconnect",
            sensitivity="secret",
        )
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        now = datetime.now(UTC)
        await event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=EventType.POLICY_DECISION_RECORDED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=submitted["conversation_id"],
                request_id=submitted["request_id"],
                correlation_id=submitted["request_id"],
                causation_id=None,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="test",
                source_node=None,
                sensitivity=Sensitivity.SECRET,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "source_ref": "current_user_message",
                    "allowed": False,
                    "code": "sensitivity_denied",
                    "reason": "secret audit payload",
                },
                metadata={},
            ),
        )
        _, reconnect_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return _sse_events(reconnect_raw)

    events = asyncio.run(scenario())

    assert EventType.POLICY_DECISION_RECORDED.value not in [event for event, _ in events]
    assert "secret audit payload" not in json.dumps(events)


def test_reconnect_stream_does_not_expose_internal_loop_events(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-loop-reconnect",
        )
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        replay_app = make_app(FakeModelProvider(stream_tokens=["SHOULD_NOT_RERUN"]))
        _, reconnect_raw = await _request(
            replay_app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        persisted = await event_log.query(EventFilter(request_id=submitted["request_id"]))
        return _sse_events(reconnect_raw), persisted

    events, persisted_events = asyncio.run(scenario())

    assert all(not event.startswith("agent.loop.") for event, _ in events)
    assert any(
        event.event_type == EventType.AGENT_LOOP_STARTED
        for event in persisted_events
    )
    assert any(
        event.event_type == EventType.AGENT_LOOP_COMPLETED
        for event in persisted_events
    )


def test_reconnect_stream_projects_allowlisted_events_to_public_payloads(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-public-replay",
            content="public replay",
        )
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        now = datetime.now(UTC)
        await event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=EventType.CONTEXT_ASSEMBLED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=submitted["conversation_id"],
                request_id=submitted["request_id"],
                correlation_id=submitted["request_id"],
                causation_id=None,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="test",
                source_node=None,
                sensitivity=Sensitivity.PROJECT,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "context_manifest_id": "public-manifest",
                    "section_names": ["system_identity", "current_user_message"],
                    "used_message_ids": ["internal-message-id"],
                    "raw_prompt": "secret audit payload",
                },
                metadata={"trace": "internal metadata"},
            ),
        )
        replay_app = make_app(FakeModelProvider(stream_tokens=["SHOULD_NOT_RERUN"]))
        _, reconnect_raw = await _request(
            replay_app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return _sse_events(reconnect_raw)

    events = asyncio.run(scenario())
    context_payloads = [
        data
        for event, data in events
        if event == EventType.CONTEXT_ASSEMBLED.value
        and data.get("context_manifest_id") == "public-manifest"
    ]

    assert context_payloads
    assert set(context_payloads[-1]) == {"request_id", "event_id", "context_manifest_id"}
    assert "secret audit payload" not in json.dumps(events)
    assert "internal-message-id" not in json.dumps(events)


def test_reconnect_stream_redacts_model_and_message_payloads(stream_parts) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        app = make_app(FakeModelProvider(stream_tokens=["A"]))
        submitted = await _accepted_message(
            app,
            client_message_id="client-model-replay-redaction",
            content="model replay redaction",
        )
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        now = datetime.now(UTC)

        async def append(event_type: EventType, payload: dict[str, Any]) -> None:
            await event_log.append(
                EventEnvelope(
                    event_id=str(uuid4()),
                    event_seq=0,
                    event_type=event_type,
                    event_version=1,
                    occurred_at=now,
                    recorded_at=now,
                    conversation_id=submitted["conversation_id"],
                    request_id=submitted["request_id"],
                    correlation_id=submitted["request_id"],
                    causation_id=None,
                    parent_event_id=None,
                    actor_type=ActorType.SYSTEM,
                    actor_id=None,
                    source_component="test",
                    source_node=None,
                    sensitivity=Sensitivity.PROJECT,
                    visibility=EventVisibility.INTERNAL,
                    idempotency_key=None,
                    payload=payload,
                    metadata={"trace": "internal metadata"},
                ),
            )

        await append(
            EventType.MODEL_REQUEST_CREATED,
            {
                "context_manifest_id": "model-redaction-manifest",
                "messages": [{"role": "user", "content": "raw model request secret"}],
            },
        )
        await append(
            EventType.MODEL_RESPONSE_RECEIVED,
            {
                "context_manifest_id": "model-redaction-manifest",
                "raw_response": "raw model response secret",
            },
        )
        await append(
            EventType.ASSISTANT_MESSAGE_CREATED,
            {
                "message_id": "assistant-redaction-message",
                "content_hash": "assistant-redaction-hash",
                "content": "raw assistant message secret",
            },
        )
        replay_app = make_app(FakeModelProvider(stream_tokens=["SHOULD_NOT_RERUN"]))
        _, reconnect_raw = await _request(
            replay_app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return _sse_events(reconnect_raw)

    events = asyncio.run(scenario())
    encoded = json.dumps(events)

    assert "model-redaction-manifest" in encoded
    assert "assistant-redaction-message" in encoded
    assert "assistant-redaction-hash" in encoded
    assert "raw model request secret" not in encoded
    assert "raw model response secret" not in encoded
    assert "raw assistant message secret" not in encoded


def test_completed_request_stream_reconnect_does_not_rerun_provider(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        provider = FakeModelProvider(stream_tokens=["A"])
        app = make_app(provider)
        submitted = await _accepted_message(
            app,
            client_message_id="client-reconnect",
            content="hello reconnect",
        )
        first_status, first_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        second_status, second_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return first_status, _sse_events(first_raw), second_status, _sse_events(second_raw), provider.stream_calls

    first_status, first_events, second_status, second_events, stream_calls = asyncio.run(scenario())

    assert first_status == 200
    assert first_events[-1][0] == "request.processing.completed"
    assert second_status == 200
    assert second_events[-1][0] == "request.processing.completed"
    assert stream_calls == 1


def test_completed_request_reconnect_uses_durable_replay_after_live_buffer_cleanup(
    stream_parts,
) -> None:
    async def scenario():
        make_app, event_log = stream_parts
        provider = FakeModelProvider(stream_tokens=["A"])
        app = make_app(provider)
        submitted = await _accepted_message(
            app,
            client_message_id="client-same-process-replay",
            content="same process replay",
        )
        await _request(app, "GET", f"/v1/requests/{submitted['request_id']}/stream")
        now = datetime.now(UTC)
        await event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=EventType.CONTEXT_ASSEMBLED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=submitted["conversation_id"],
                request_id=submitted["request_id"],
                correlation_id=submitted["request_id"],
                causation_id=None,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="test",
                source_node=None,
                sensitivity=Sensitivity.PROJECT,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "context_manifest_id": "same-process-replay-manifest",
                    "raw_prompt": "must not leak",
                },
                metadata={},
            ),
        )
        _, reconnect_raw = await _request(
            app,
            "GET",
            f"/v1/requests/{submitted['request_id']}/stream",
        )
        return _sse_events(reconnect_raw), provider.stream_calls

    events, stream_calls = asyncio.run(scenario())

    assert "same-process-replay-manifest" in json.dumps(events)
    assert "must not leak" not in json.dumps(events)
    assert stream_calls == 1


def test_waiting_approval_request_without_active_task_fails_as_orphaned(stream_parts) -> None:
    async def scenario():
        make_app, _ = stream_parts
        base_settings = ConfigLoader(Path("config")).load("test")
        timeout_settings = replace(
            base_settings,
            api=replace(
                base_settings.api,
                request_timeout_seconds=0,
                sse_heartbeat_seconds=0,
            ),
        )
        app = make_app(FakeModelProvider(stream_tokens=["SHOULD_NOT_RUN"]), settings_override=timeout_settings)
        store = app.state.test_conversation_store
        conversation = await store.create_conversation(
            CreateConversationCommand(
                user_id="local_user",
                title="waiting approval orphan",
                active_project_namespace="project.personal_assistant",
            ),
        )
        submission = await store.submit_user_message(
            MessageSubmissionCommand(
                conversation_id=conversation.conversation_id,
                client_message_id="client-waiting-approval-orphan",
                content="needs approval",
                sensitivity=Sensitivity.PROJECT,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        await store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=submission.request.request_id,
                status=RequestStatus.WAITING_APPROVAL,
            ),
        )
        status, raw = await asyncio.wait_for(
            _request(app, "GET", f"/v1/requests/{submission.request.request_id}/stream"),
            timeout=1,
        )
        request_status, request_payload = await _json_request(
            app,
            "GET",
            f"/v1/requests/{submission.request.request_id}",
        )
        return status, _sse_events(raw), request_status, request_payload

    status, events, request_status, request_payload = asyncio.run(scenario())

    assert status == 200
    assert request_status == 200
    assert events[-1][0] == EventType.REQUEST_PROCESSING_FAILED.value
    assert request_payload["status"] == "failed"
    assert request_payload["error"]["code"] == "orphaned_waiting_approval_request"
