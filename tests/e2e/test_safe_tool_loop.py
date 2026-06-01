from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.domain.events import EventType
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
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
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository
from assistant_core.tools.fake import fake_echo_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry


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


async def _accepted_turn(conversation_store: PostgresConversationStore):
    from assistant_core.domain.conversations import CreateConversationCommand, MessageSubmissionCommand

    conversation = await conversation_store.create_conversation(
        CreateConversationCommand(
            user_id="user-1",
            title="tool loop",
            active_project_namespace="project.personal_assistant",
        ),
    )
    submission = await conversation_store.submit_user_message(
        MessageSubmissionCommand(
            conversation_id=conversation.conversation_id,
            client_message_id="client-tool-loop",
            content="use fake echo",
            sensitivity=Sensitivity.PROJECT,
        ),
    )
    return submission


async def _run_tool_loop(structured_responses: list[dict]):
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
    tool_gateway = ToolGateway(
        registry=ToolRegistry([fake_echo_tool()]),
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
            metadata={
                "agent_tool_policy": "available",
                "agent_allowed_tool_names": ["fake.echo"],
            },
        )
        try:
            result = await runtime.run_turn(command)
        except Exception as exc:  # noqa: BLE001 - e2e returns failure state for assertions.
            result = exc
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        request = await conversation_store.get_assistant_request(submission.request.request_id)
        return result, request, events
    finally:
        await engine.dispose()


def test_safe_tool_loop_user_turn_with_fake_tool_completes() -> None:
    result, request, events = asyncio.run(
        _run_tool_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "fake.echo",
                    "arguments": {"message": "hello"},
                },
                {"action": "final_answer", "final_answer": "hello"},
            ],
        ),
    )

    assert not isinstance(result, Exception)
    assert request.status == RequestStatus.COMPLETED
    assert result.response_text == "hello"
    assert EventType.TOOL_CALL_COMPLETED in [event.event_type for event in events]


def test_safe_tool_loop_malformed_tool_request_fails_safely() -> None:
    result, request, events = asyncio.run(_run_tool_loop([{"action": "tool_call"}]))

    assert isinstance(result, Exception)
    assert request.status == RequestStatus.FAILED
    assert EventType.AGENT_STEP_FAILED in [event.event_type for event in events]
    assert EventType.TOOL_CALL_STARTED not in [event.event_type for event in events]


def test_safe_tool_loop_budget_exhaustion_fails_safely() -> None:
    result, request, events = asyncio.run(
        _run_tool_loop(
            [
                {
                    "action": "tool_call",
                    "tool_name": "fake.echo",
                    "arguments": {"message": "first"},
                },
                {
                    "action": "tool_call",
                    "tool_name": "fake.echo",
                    "arguments": {"message": "second"},
                },
                {
                    "action": "tool_call",
                    "tool_name": "fake.echo",
                    "arguments": {"message": "third"},
                },
            ],
        ),
    )

    assert isinstance(result, Exception)
    assert request.status == RequestStatus.FAILED
    completed_tool_calls = [
        event for event in events if event.event_type == EventType.TOOL_CALL_COMPLETED
    ]
    assert len(completed_tool_calls) == 2
