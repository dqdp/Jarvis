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
from assistant_core.tools.shell_read import ProjectShellReadTool, ShellExecutionResult


pytestmark = [pytest.mark.e2e, pytest.mark.db]


class FakeShellExecutor:
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
            stdout="docs/37_post_mvp_tdd_slices_plan.md: ToolGatewayPort\n",
            stderr="",
        )


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
            title="project shell loop",
            active_project_namespace="project.personal_assistant",
        ),
    )
    submission = await conversation_store.submit_user_message(
        MessageSubmissionCommand(
            conversation_id=conversation.conversation_id,
            client_message_id="client-project-shell-loop",
            content="inspect project docs",
            sensitivity=Sensitivity.PROJECT,
        ),
    )
    return submission


async def _run_project_shell_loop(
    structured_responses: list[dict],
    *,
    chat_response: str = "fake response",
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
            "local_main": FakeModelProvider(
                chat_response=chat_response,
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
    shell_executor = FakeShellExecutor()
    tool_gateway = ToolGateway(
        registry=ToolRegistry(
            [
                ProjectShellReadTool(
                    allowed_roots=[Path.cwd()],
                    executor=shell_executor,
                    max_stdout_bytes=2000,
                    max_stderr_bytes=2000,
                    max_lines=20,
                    timeout_seconds=1.0,
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
            model_profile="local_main",
            permission_mode="developer_local",
            working_directory=str(Path.cwd()),
            metadata={
                "agent_tool_policy": "available",
                "agent_allowed_tool_names": ["tool.shell.read.project"],
            },
        )
        try:
            result = await runtime.run_turn(command)
        except Exception as exc:  # noqa: BLE001 - e2e returns failure state for assertions.
            result = exc
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        request = await conversation_store.get_assistant_request(submission.request.request_id)
        return result, request, events, shell_executor
    finally:
        await engine.dispose()


def test_agent_can_use_project_rg_tool_and_answer_with_observation() -> None:
    result, request, events, shell_executor = asyncio.run(
        _run_project_shell_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.shell.read.project",
                    "arguments": {
                        "argv": ["rg", "ToolGatewayPort", "docs"],
                        "cwd": str(Path.cwd()),
                    },
                },
                {"action": "final_answer"},
            ],
            chat_response="ToolGatewayPort is documented.",
        ),
    )

    assert not isinstance(result, Exception)
    assert request.status == RequestStatus.COMPLETED
    assert result.response_text == "ToolGatewayPort is documented."
    assert shell_executor.calls[0]["argv"] == ["rg", "ToolGatewayPort", "docs"]
    assert EventType.TOOL_SHELL_COMPLETED in [event.event_type for event in events]


def test_agent_cannot_use_denied_shell_command() -> None:
    result, request, events, shell_executor = asyncio.run(
        _run_project_shell_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.shell.read.project",
                    "arguments": {
                        "argv": ["curl", "https://example.com"],
                        "cwd": str(Path.cwd()),
                    },
                },
            ],
        ),
    )

    assert isinstance(result, Exception)
    assert request.status == RequestStatus.FAILED
    assert shell_executor.calls == []
    assert EventType.TOOL_SHELL_DENIED in [event.event_type for event in events]
    assert EventType.TOOL_SHELL_STARTED not in [event.event_type for event in events]


def test_shell_denial_is_returned_as_tool_observation() -> None:
    _result, _request, events, _shell_executor = asyncio.run(
        _run_project_shell_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "tool.shell.read.project",
                    "arguments": {
                        "argv": ["git", "commit", "-m", "nope"],
                        "cwd": str(Path.cwd()),
                    },
                },
            ],
        ),
    )

    observation_event = next(
        event
        for event in events
        if event.event_type == EventType.TOOL_OBSERVATION_RECORDED
    )
    assert observation_event.payload["status"] == ToolObservationStatus.DENIED.value
    assert observation_event.payload["error_code"] == "git_subcommand_denied"
