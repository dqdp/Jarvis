from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.domain.events import EventType
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.system_diagnostics import SensorReading, SensorSnapshot
from assistant_core.domain.system_diagnostics import SystemDiagnosticsFamily
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.models.router import ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.agent_runtime import AgentRuntime, RuntimeTurnCommand
from assistant_core.runtime.loops import LoopStrategyRegistry, MemoryAugmentedAnswerLoop
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry
from assistant_core.tools.shell_read import ShellExecutionResult
from assistant_core.tools.system_diagnostics import SystemDiagnosticsTool


pytestmark = [pytest.mark.e2e, pytest.mark.db]


class FakeDiagnosticsExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> ShellExecutionResult:
        self.calls.append(
            {
                "argv": argv,
                "cwd": cwd,
                "env": env,
                "timeout_seconds": timeout_seconds,
            },
        )
        return ShellExecutionResult(
            exit_code=0,
            stdout="123 ollama\n456 jarvis\n",
            stderr="",
        )


class FakeSensorProvider:
    def __init__(self, snapshot: SensorSnapshot) -> None:
        self.snapshot = snapshot

    async def snapshot_temperatures(self) -> SensorSnapshot:
        return self.snapshot


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
                    "truncate table approvals, content_embeddings, content_chunks, "
                    "content_sources, memory_embeddings, memory_candidates, memories, "
                    "model_invocations, assistant_requests, messages, conversations, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


async def _accepted_turn(conversation_store: PostgresConversationStore):
    from assistant_core.domain.conversations import CreateConversationCommand, MessageSubmissionCommand

    conversation = await conversation_store.create_conversation(
        CreateConversationCommand(
            user_id="user-1",
            title="system diagnostics loop",
            active_project_namespace="project.personal_assistant",
        ),
    )
    submission = await conversation_store.submit_user_message(
        MessageSubmissionCommand(
            conversation_id=conversation.conversation_id,
            client_message_id="client-system-diagnostics-loop",
            content="inspect local runtime",
            sensitivity=Sensitivity.INFRA,
        ),
    )
    return submission


async def _run_system_diagnostics_loop(
    structured_responses: list[dict],
    *,
    sensor_snapshot: SensorSnapshot | None = None,
):
    database_url = _database_url()
    assert_test_database_url(database_url)
    await _truncate_e2e(database_url)
    engine = create_database_engine(database_url)
    settings = ConfigLoader(Path("config")).load("test")
    event_log = PostgresEventLog(engine)
    policy = ConfigPolicyEngine(settings)
    conversation_store = PostgresConversationStore(engine)
    memory_store = PostgresMemoryStore(
        engine=engine,
        settings=settings,
        policy=policy,
        embedding_port=FakeEmbeddingProvider(),
    )
    invocation_repository = PostgresModelInvocationRepository(engine)
    router = ModelRouter(
        settings=settings,
        policy=policy,
        invocation_repository=invocation_repository,
        event_log=event_log,
        providers={
            "local_openai_compatible": FakeModelProvider(
                structured_text_responses=[json.dumps(response) for response in structured_responses],
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
    diagnostics_executor = FakeDiagnosticsExecutor()
    sensor_provider = FakeSensorProvider(
        sensor_snapshot
        or SensorSnapshot(
            source="fake",
            readings=[SensorReading(label="cpu", value=54.0, unit="C")],
        ),
    )
    tool_gateway = ToolGateway(
        registry=ToolRegistry(
            [
                SystemDiagnosticsTool(
                    family=SystemDiagnosticsFamily.PROCESS,
                    allowed_roots=[Path.cwd()],
                    executor=diagnostics_executor,
                    max_stdout_bytes=2000,
                    max_stderr_bytes=2000,
                    max_lines=20,
                    timeout_seconds=1.0,
                    platform="linux",
                ),
                SystemDiagnosticsTool(
                    family=SystemDiagnosticsFamily.SENSORS,
                    allowed_roots=[Path.cwd()],
                    executor=diagnostics_executor,
                    sensor_provider=sensor_provider,
                    max_stdout_bytes=2000,
                    max_stderr_bytes=2000,
                    max_lines=20,
                    timeout_seconds=1.0,
                    platform="linux",
                ),
            ],
        ),
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
    try:
        submission = await _accepted_turn(conversation_store)
        command = RuntimeTurnCommand(
            request_id=submission.request.request_id,
            conversation_id=submission.request.conversation_id,
            user_message_id=submission.user_message.message_id,
            user_id="user-1",
            user_input=submission.user_message.content,
            active_project_namespace="project.personal_assistant",
            loop_strategy="tool_react_loop",
            model_profile="local_structured",
            permission_mode="developer_local",
            working_directory=str(Path.cwd()),
        )
        try:
            result = await runtime.run_turn(command)
        except Exception as exc:  # noqa: BLE001 - e2e returns failure state for assertions.
            result = exc
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        request = await conversation_store.get_assistant_request(submission.request.request_id)
        return result, request, events, diagnostics_executor
    finally:
        await engine.dispose()


def test_agent_can_use_process_snapshot_and_answer_with_observation() -> None:
    result, request, events, diagnostics_executor = asyncio.run(
        _run_system_diagnostics_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.system.read.process",
                    "arguments": {
                        "argv": ["ps", "-Ao", "pid,comm,command"],
                        "cwd": str(Path.cwd()),
                    },
                },
                {"action": "final_answer", "final_answer": "Ollama and Jarvis are running."},
            ],
        ),
    )

    assert not isinstance(result, Exception)
    assert request.status == RequestStatus.COMPLETED
    assert result.response_text == "Ollama and Jarvis are running."
    assert diagnostics_executor.calls[0]["argv"] == ["ps", "-Ao", "pid,comm,command"]
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED in [event.event_type for event in events]


def test_agent_can_use_temperature_snapshot_when_available() -> None:
    result, request, events, _diagnostics_executor = asyncio.run(
        _run_system_diagnostics_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.system.read.sensors",
                    "arguments": {"argv": ["thermal-sysfs"], "cwd": str(Path.cwd())},
                },
                {"action": "final_answer", "final_answer": "CPU temperature is 54 C."},
            ],
        ),
    )

    assert not isinstance(result, Exception)
    assert request.status == RequestStatus.COMPLETED
    assert result.response_text == "CPU temperature is 54 C."
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED in [event.event_type for event in events]


def test_agent_handles_unavailable_temperature_backend() -> None:
    result, request, events, _diagnostics_executor = asyncio.run(
        _run_system_diagnostics_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.system.read.sensors",
                    "arguments": {"argv": ["thermal-sysfs"], "cwd": str(Path.cwd())},
                },
                {"action": "final_answer", "final_answer": "Temperature backend is unavailable."},
            ],
            sensor_snapshot=SensorSnapshot.unavailable(source="thermal-sysfs", reason="not available"),
        ),
    )

    assert not isinstance(result, Exception)
    assert request.status == RequestStatus.COMPLETED
    assert result.response_text == "Temperature backend is unavailable."
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE in [event.event_type for event in events]


def test_agent_cannot_use_denied_diagnostics_command() -> None:
    result, request, events, diagnostics_executor = asyncio.run(
        _run_system_diagnostics_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.system.read.process",
                    "arguments": {"argv": ["kill", "123"], "cwd": str(Path.cwd())},
                },
            ],
        ),
    )

    assert isinstance(result, Exception)
    assert request.status == RequestStatus.FAILED
    assert diagnostics_executor.calls == []
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED in [event.event_type for event in events]
    observation_event = next(
        event
        for event in events
        if event.event_type == EventType.TOOL_OBSERVATION_RECORDED
    )
    assert observation_event.payload["status"] == ToolObservationStatus.DENIED.value
