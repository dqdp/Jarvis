from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.conversations import CreateConversationCommand, MessageSubmissionCommand
from assistant_core.domain.events import EventType
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.agent_runtime import RuntimeTurnCommand
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.migrations import run_migrations


pytestmark = [pytest.mark.contract, pytest.mark.db]


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


def _sse_event_payloads(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in raw.strip().split("\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        event_type = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event_type, data))
    return events


class BlockingStreamModelProvider(FakeModelProvider):
    def __init__(self) -> None:
        super().__init__(stream_tokens=["unreachable"])
        self.started = asyncio.Event()

    async def stream_chat(self, request):
        self.stream_calls += 1
        self.started.set()
        await asyncio.Event().wait()
        yield "unreachable"


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
                "local_structured": FakeModelProvider(),
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
    assert health["readiness"]["checks"]["content_store"] == "ok"
    assert stream_events[-1] == "request.processing.completed"
    assert request_status["status"] == "completed"
    assert stream_calls == 1
    assert embed_calls == 1


def test_runtime_app_factory_dispose_shutdowns_active_request_tasks() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        model_provider = BlockingStreamModelProvider()
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_main": model_provider,
                "local_structured": FakeModelProvider(),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        disposed = False
        try:
            app = runtime_app.app
            _, conversation_raw = await _request(
                app,
                "POST",
                "/v1/conversations",
                {"title": "factory dispose", "active_project_namespace": "project.personal_assistant"},
            )
            conversation = json.loads(conversation_raw)
            status_code, message_raw = await _request(
                app,
                "POST",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
                {
                    "client_message_id": "client-factory-dispose",
                    "content": "hold the provider open",
                    "sensitivity": "project",
                },
            )
            submitted = json.loads(message_raw)
            manager = app.state.request_execution_manager
            await asyncio.wait_for(model_provider.started.wait(), timeout=1.0)
            assert status_code == 202
            assert submitted["request_id"] in manager._tasks

            await runtime_app.dispose()
            disposed = True
            return submitted["request_id"] in manager._tasks
        finally:
            if not disposed:
                await runtime_app.dispose()

    task_still_tracked = asyncio.run(scenario())

    assert task_still_tracked is False


def test_runtime_app_factory_health_reports_missing_inference_provider() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={"local_embedding": FakeEmbeddingProvider()},
        )
        try:
            status, health_raw = await _request(runtime_app.app, "GET", "/v1/health")
            return status, json.loads(health_raw)
        finally:
            await runtime_app.dispose()

    status, health = asyncio.run(scenario())

    assert status == 503
    assert health["status"] == "not_ready"
    assert health["readiness"]["checks"]["inference"] == "failed"
    assert "local_main" in health["readiness"]["reasons"]["inference"]


def test_runtime_app_factory_exposes_project_docs_content_ops(tmp_path: Path) -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    (tmp_path / "README.md").write_text("# Readme\nalpha project docs\n", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide\nbeta project docs\n", encoding="utf-8")

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            project_root=tmp_path,
            providers={
                "local_main": FakeModelProvider(stream_tokens=["OK"]),
                "local_structured": FakeModelProvider(),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            app = runtime_app.app
            ingest_status, ingest_raw = await _request(
                app,
                "POST",
                "/v1/content/project-docs/ingest",
            )
            sources_status, sources_raw = await _request(app, "GET", "/v1/content/sources")
            status_code, status_raw = await _request(app, "GET", "/v1/content/status")
            (docs_dir / "guide.md").write_text("# Guide\nupdated project docs\n", encoding="utf-8")
            reindex_status, reindex_raw = await _request(
                app,
                "POST",
                "/v1/content/project-docs/reindex",
            )
            return (
                ingest_status,
                json.loads(ingest_raw),
                sources_status,
                json.loads(sources_raw),
                status_code,
                json.loads(status_raw),
                reindex_status,
                json.loads(reindex_raw),
            )
        finally:
            await runtime_app.dispose()

    (
        ingest_status,
        ingest,
        sources_status,
        sources,
        status_code,
        content_status,
        reindex_status,
        reindex,
    ) = asyncio.run(scenario())

    assert ingest_status == 200
    assert ingest["seen_sources"] == 2
    assert sources_status == 200
    assert {source["path"] for source in sources["sources"]} == {"README.md", "docs/guide.md"}
    assert status_code == 200
    assert content_status["sources"]["total"] == 2
    assert content_status["chunks"]["total"] >= 2
    assert reindex_status == 200
    assert reindex["updated_sources"] >= 1


def test_runtime_app_factory_content_ops_require_policy_allow(tmp_path: Path) -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    (tmp_path / "README.md").write_text("# Readme\nalpha project docs\n", encoding="utf-8")

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        modes = {name: dict(actions) for name, actions in settings.permissions.modes.items()}
        modes["developer_local"]["content.retrieve"] = "deny"
        modes["developer_local"]["content.ingest"] = "deny"
        modes["developer_local"]["content.index"] = "deny"
        settings = replace(settings, permissions=replace(settings.permissions, modes=modes))
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            project_root=tmp_path,
            providers={
                "local_main": FakeModelProvider(stream_tokens=["OK"]),
                "local_structured": FakeModelProvider(),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            ingest_status, ingest_raw = await _request(
                runtime_app.app,
                "POST",
                "/v1/content/project-docs/ingest",
            )
            reindex_status, reindex_raw = await _request(
                runtime_app.app,
                "POST",
                "/v1/content/project-docs/reindex",
            )
            sources_status, sources_raw = await _request(
                runtime_app.app,
                "GET",
                "/v1/content/sources",
            )
            status_status, status_raw = await _request(
                runtime_app.app,
                "GET",
                "/v1/content/status",
            )
            return (
                ingest_status,
                json.loads(ingest_raw),
                reindex_status,
                json.loads(reindex_raw),
                sources_status,
                json.loads(sources_raw),
                status_status,
                json.loads(status_raw),
            )
        finally:
            await runtime_app.dispose()

    (
        ingest_status,
        ingest,
        reindex_status,
        reindex,
        sources_status,
        sources,
        status_status,
        content_status,
    ) = asyncio.run(scenario())

    assert ingest_status == 403
    assert ingest["error"]["code"] == "capability_denied"
    assert reindex_status == 403
    assert reindex["error"]["code"] == "capability_denied"
    assert sources_status == 403
    assert sources["error"]["code"] == "capability_denied"
    assert status_status == 403
    assert content_status["error"]["code"] == "capability_denied"


def test_runtime_app_factory_registers_tool_react_loop() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_structured": FakeModelProvider(
                    structured_text_responses=[
                        json.dumps(
                            {
                                "action": "tool_call",
                                "tool_name": "fake.echo",
                                "arguments": {"message": "factory"},
                            },
                        ),
                        json.dumps({"action": "final_answer", "final_answer": "factory"}),
                    ],
                ),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            conversation_store = PostgresConversationStore(runtime_app.engine)
            event_log = PostgresEventLog(runtime_app.engine)
            conversation = await conversation_store.create_conversation(
                CreateConversationCommand(
                    user_id=settings.app.default_user_id,
                    title="factory tool loop",
                    active_project_namespace="project.personal_assistant",
                ),
            )
            submission = await conversation_store.submit_user_message(
                MessageSubmissionCommand(
                    conversation_id=conversation.conversation_id,
                    client_message_id="client-factory-tool-loop",
                    content="use fake echo",
                    sensitivity=Sensitivity.PROJECT,
                ),
            )
            result = await runtime_app.runtime.run_turn(
                RuntimeTurnCommand(
                    request_id=submission.request.request_id,
                    conversation_id=submission.request.conversation_id,
                    user_message_id=submission.user_message.message_id,
                    user_id=settings.app.default_user_id,
                    user_input=submission.user_message.content,
                    active_project_namespace=conversation.active_project_namespace,
                    model_profile="local_structured",
                    loop_strategy=LoopStrategyName.TOOL_REACT_LOOP.value,
                ),
            )
            events = await event_log.query(EventFilter(request_id=submission.request.request_id))
            return result, [event.event_type for event in events]
        finally:
            await runtime_app.dispose()

    result, event_types = asyncio.run(scenario())

    assert result.response_text == "factory"
    assert EventType.TOOL_CALL_COMPLETED in event_types
    assert EventType.POLICY_CAPABILITY_DECISION_RECORDED in event_types


def test_runtime_app_factory_registers_project_shell_read_tool() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_structured": FakeModelProvider(
                    structured_text_responses=[
                        json.dumps(
                            {
                                "action": "tool_call",
                                "tool_name": "tool.shell.read.project",
                                "arguments": {
                                    "argv": ["pwd"],
                                    "cwd": str(Path.cwd()),
                                },
                            },
                        ),
                        json.dumps({"action": "final_answer", "final_answer": "shell ok"}),
                    ],
                ),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            conversation_store = PostgresConversationStore(runtime_app.engine)
            event_log = PostgresEventLog(runtime_app.engine)
            conversation = await conversation_store.create_conversation(
                CreateConversationCommand(
                    user_id=settings.app.default_user_id,
                    title="factory shell tool loop",
                    active_project_namespace="project.personal_assistant",
                ),
            )
            submission = await conversation_store.submit_user_message(
                MessageSubmissionCommand(
                    conversation_id=conversation.conversation_id,
                    client_message_id="client-factory-shell-tool-loop",
                    content="use shell read",
                    sensitivity=Sensitivity.PROJECT,
                ),
            )
            result = await runtime_app.runtime.run_turn(
                RuntimeTurnCommand(
                    request_id=submission.request.request_id,
                    conversation_id=submission.request.conversation_id,
                    user_message_id=submission.user_message.message_id,
                    user_id=settings.app.default_user_id,
                    user_input=submission.user_message.content,
                    active_project_namespace=conversation.active_project_namespace,
                    model_profile="local_structured",
                    loop_strategy=LoopStrategyName.TOOL_REACT_LOOP.value,
                    permission_mode="developer_local",
                    working_directory=str(Path.cwd()),
                ),
            )
            events = await event_log.query(EventFilter(request_id=submission.request.request_id))
            return result, [event.event_type for event in events]
        finally:
            await runtime_app.dispose()

    result, event_types = asyncio.run(scenario())

    assert result.response_text == "shell ok"
    assert EventType.TOOL_SHELL_COMPLETED in event_types


def test_runtime_app_factory_registers_system_diagnostics_tool() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_structured": FakeModelProvider(
                    structured_text_responses=[
                        json.dumps(
                            {
                                "action": "tool_call",
                                "tool_name": "tool.system.read.process",
                                "arguments": {
                                    "argv": ["kill", "123"],
                                    "cwd": str(Path.cwd()),
                                },
                            },
                        ),
                    ],
                ),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            conversation_store = PostgresConversationStore(runtime_app.engine)
            event_log = PostgresEventLog(runtime_app.engine)
            conversation = await conversation_store.create_conversation(
                CreateConversationCommand(
                    user_id=settings.app.default_user_id,
                    title="factory system diagnostics tool loop",
                    active_project_namespace="project.personal_assistant",
                ),
            )
            submission = await conversation_store.submit_user_message(
                MessageSubmissionCommand(
                    conversation_id=conversation.conversation_id,
                    client_message_id="client-factory-system-diagnostics-loop",
                    content="use diagnostics",
                    sensitivity=Sensitivity.INFRA,
                ),
            )
            try:
                await runtime_app.runtime.run_turn(
                    RuntimeTurnCommand(
                        request_id=submission.request.request_id,
                        conversation_id=submission.request.conversation_id,
                        user_message_id=submission.user_message.message_id,
                        user_id=settings.app.default_user_id,
                        user_input=submission.user_message.content,
                        active_project_namespace=conversation.active_project_namespace,
                        model_profile="local_structured",
                        loop_strategy=LoopStrategyName.TOOL_REACT_LOOP.value,
                        permission_mode="developer_local",
                        working_directory=str(Path.cwd()),
                    ),
                )
            except RuntimeError:
                pass
            events = await event_log.query(EventFilter(request_id=submission.request.request_id))
            return [event.event_type for event in events]
        finally:
            await runtime_app.dispose()

    event_types = asyncio.run(scenario())

    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED in event_types


def test_runtime_app_factory_registers_all_enabled_system_diagnostics_tools() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        settings = ConfigLoader(Path("config")).load("test")
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_structured": FakeModelProvider(),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            registry = runtime_app.runtime._loop_strategy_registry
            tool_loop = registry.get(LoopStrategyName.TOOL_REACT_LOOP)
            tool_gateway = tool_loop._tool_gateway
            tools = await tool_gateway.list_tools()
            return {tool.name for tool in tools}
        finally:
            await runtime_app.dispose()

    tool_names = asyncio.run(scenario())

    assert {
        "tool.system.read.process",
        "tool.system.read.resources",
        "tool.system.read.hardware",
        "tool.system.read.network",
        "tool.system.read.sensors",
    }.issubset(tool_names)


def test_runtime_app_factory_api_can_select_tool_react_loop() -> None:
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        from assistant_core.app_factory import create_runtime_app

        await _truncate_runtime_app(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        structured_provider = FakeModelProvider(
            structured_text_responses=[
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "api"},
                    },
                ),
                json.dumps({"action": "final_answer", "final_answer": "api"}),
            ],
        )
        runtime_app = create_runtime_app(
            database_url=database_url,
            settings=settings,
            providers={
                "local_structured": structured_provider,
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        try:
            app = runtime_app.app
            _, conversation_raw = await _request(
                app,
                "POST",
                "/v1/conversations",
                {"title": "factory api", "active_project_namespace": "project.personal_assistant"},
            )
            conversation = json.loads(conversation_raw)
            status_code, message_raw = await _request(
                app,
                "POST",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
                {
                    "client_message_id": "client-factory-api-tool-loop",
                    "content": "use fake echo",
                    "sensitivity": "project",
                    "loop_strategy": LoopStrategyName.TOOL_REACT_LOOP.value,
                },
            )
            submitted = json.loads(message_raw)
            _, stream_raw = await _request(
                app,
                "GET",
                f"/v1/requests/{submitted['request_id']}/stream",
            )
            event_log = PostgresEventLog(runtime_app.engine)
            events = await event_log.query(EventFilter(request_id=submitted["request_id"]))
            return (
                status_code,
                _sse_events(stream_raw),
                _sse_event_payloads(stream_raw),
                [event.event_type for event in events],
                structured_provider.structured_calls,
            )
        finally:
            await runtime_app.dispose()

    status_code, stream_events, stream_payloads, event_types, structured_calls = asyncio.run(
        scenario(),
    )

    assert status_code == 202
    assert stream_events[-1] == "request.processing.completed"
    assert stream_payloads[-1][1]["event_id"]
    assert EventType.TOOL_CALL_COMPLETED in event_types
    assert structured_calls == 2
