from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from pathlib import Path

import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.domain.context import AssembledContext, ContextManifest, ContextSection
from assistant_core.domain.conversations import (
    CompleteAssistantResponseCommand,
    CreateConversationCommand,
    MessageSubmissionCommand,
    RecentMessagesQuery,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.memory import MemoryQuery
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import ChatModelResponse
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.models.router import ModelProviderError, ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.memory import MemoryRetrievalError
from assistant_core.runtime.agent_runtime import AgentRuntime, RuntimeTurnCommand
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_runtime(database_url: str) -> None:
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


class EmptyMemoryRead:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def retrieve(self, query: MemoryQuery):
        if self.fail:
            raise MemoryRetrievalError("memory unavailable")
        return []


class RecordingContextAssembler:
    def __init__(self) -> None:
        self.calls = 0

    async def assemble(self, request) -> AssembledContext:
        self.calls += 1
        manifest = ContextManifest(
            context_manifest_id="33333333-3333-3333-3333-333333333333",
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            loop_strategy=request.loop_strategy,
            model_profile=request.model_profile,
            section_names=["current_user_message"],
            used_message_ids=[],
            used_memory_ids=[],
            dropped_refs=[],
            token_estimate=2,
            active_namespaces=[],
            retrieval_parameters={},
            max_sensitivity=Sensitivity.PROJECT,
            sources_by_sensitivity={"project": ["current_user_message"]},
            degraded=False,
            full_prompt_stored=False,
        )
        return AssembledContext(
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[TextPart(text=request.current_user_message)],
                    sensitivity=Sensitivity.PROJECT,
                ),
            ],
            sections=[ContextSection("current_user_message", request.current_user_message, 2)],
            manifest=manifest,
            token_estimate=2,
        )


class SlowContextAssembler(RecordingContextAssembler):
    async def assemble(self, request) -> AssembledContext:
        await asyncio.sleep(1)
        return await super().assemble(request)


class CancellingAfterAssistantAppendStore:
    def __init__(self, delegate: PostgresConversationStore) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    async def append_message(self, command):
        message = await self._delegate.append_message(command)
        if command.role == MessageRole.ASSISTANT:
            await self._cancel(command.request_id)
        return message

    async def complete_assistant_response(self, command: CompleteAssistantResponseCommand):
        completion = await self._delegate.complete_assistant_response(command)
        await self._cancel(command.request_id)
        return completion

    async def _cancel(self, request_id: str | None) -> None:
        if request_id is None:
            return
        try:
            await self._delegate.update_assistant_request_status(
                UpdateAssistantRequestStatusCommand(
                    request_id=request_id,
                    status=RequestStatus.CANCELLED,
                    error_code="cancelled",
                    error_message="request cancelled",
                ),
            )
        except Exception:
            return


@pytest.fixture
def runtime_parts():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_runtime(database_url))
    engine = create_database_engine(database_url)
    settings = ConfigLoader(Path("config")).load("test")
    conversation_store = PostgresConversationStore(engine)
    event_log = PostgresEventLog(engine)
    invocation_repository = PostgresModelInvocationRepository(engine)

    def make_runtime(
        *,
        memory_fails: bool = False,
        model_provider: FakeModelProvider | None = None,
        assembler=None,
        runtime_settings=None,
        conversation_store_override=None,
    ):
        policy = ConfigPolicyEngine(settings)
        router = ModelRouter(
            settings=settings,
            policy=policy,
            invocation_repository=invocation_repository,
            event_log=event_log,
            providers={
                "local_openai_compatible": model_provider or FakeModelProvider(chat_response="answer"),
                "local_embedding": FakeEmbeddingProvider(),
            },
        )
        context_assembler = assembler or DeterministicContextAssembler(
            conversation_store=conversation_store,
            memory_read=EmptyMemoryRead(fail=memory_fails),
            event_log=event_log,
            policy=policy,
        )
        return AgentRuntime(
            conversation_store=conversation_store_override or conversation_store,
            context_assembler=context_assembler,
            model_router=router,
            event_log=event_log,
            settings=runtime_settings or settings,
        )

    try:
        yield engine, conversation_store, event_log, invocation_repository, make_runtime
    finally:
        asyncio.run(engine.dispose())


async def _accepted_turn(
    conversation_store: PostgresConversationStore,
    *,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
    content: str = "hello runtime",
    client_message_id: str = "client-runtime",
):
    conversation = await conversation_store.create_conversation(
        command=CreateConversationCommand(
            conversation_id="22222222-2222-2222-2222-222222222222",
            user_id="user-1",
            title="runtime",
            active_project_namespace="project.personal_assistant",
        ),
    )
    submission = await conversation_store.submit_user_message(
        MessageSubmissionCommand(
            conversation_id=conversation.conversation_id,
            client_message_id=client_message_id,
            content=content,
            sensitivity=sensitivity,
            request_id="11111111-1111-1111-1111-111111111111",
        ),
    )
    return conversation, submission


def test_runtime_persists_event_chain_success(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        runtime = make_runtime()
        await runtime.run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        return await event_log.query(EventFilter(request_id=submission.request.request_id))

    events = asyncio.run(scenario())

    core_events = [
        event.event_type
        for event in events
        if event.event_type != EventType.POLICY_DECISION_RECORDED
    ]
    assert core_events == [
        EventType.USER_MESSAGE_CREATED,
        EventType.REQUEST_PROCESSING_STARTED,
        EventType.CONTEXT_ASSEMBLY_STARTED,
        EventType.MEMORY_RETRIEVED,
        EventType.CONTEXT_ASSEMBLED,
        EventType.MODEL_REQUEST_CREATED,
        EventType.MODEL_RESPONSE_RECEIVED,
        EventType.ASSISTANT_MESSAGE_CREATED,
        EventType.REQUEST_PROCESSING_COMPLETED,
    ]
    assert any(event.event_type == EventType.POLICY_DECISION_RECORDED for event in events)


def test_runtime_uses_context_assembler(runtime_parts) -> None:
    async def scenario():
        _, store, _, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        assembler = RecordingContextAssembler()
        runtime = make_runtime(assembler=assembler)
        await runtime.run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        return assembler.calls

    assert asyncio.run(scenario()) == 1


def test_runtime_calls_model_router_once(runtime_parts) -> None:
    async def scenario():
        _, store, _, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        provider = FakeModelProvider(chat_response="answer")
        runtime = make_runtime(model_provider=provider)
        await runtime.run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        return provider.chat_calls

    assert asyncio.run(scenario()) == 1


def test_runtime_max_model_calls_one(runtime_parts) -> None:
    async def scenario():
        _, store, _, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        runtime = make_runtime()
        result = await runtime.run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        return result.model_calls

    assert asyncio.run(scenario()) == 1


def test_runtime_memory_retrieval_failure_degraded(runtime_parts) -> None:
    async def scenario():
        _, store, _, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        runtime = make_runtime(memory_fails=True)
        return await runtime.run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )

    result = asyncio.run(scenario())

    assert result.degraded is True
    assert result.response_text == "answer"


def test_runtime_model_failure_marks_request_failed(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        runtime = make_runtime(model_provider=FakeModelProvider(fail_chat_times=1))
        with pytest.raises(ModelProviderError):
            await runtime.run_turn(
                RuntimeTurnCommand(
                    request_id=submission.request.request_id,
                    conversation_id=submission.request.conversation_id,
                    user_message_id=submission.user_message.message_id,
                    user_id="user-1",
                    user_input=submission.user_message.content,
                    active_project_namespace="project.personal_assistant",
                ),
            )
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=submission.request.conversation_id, limit=10),
        )
        return submission.request.request_id, events, messages

    request_id, events, messages = asyncio.run(scenario())

    assert events[-1].event_type == EventType.REQUEST_PROCESSING_FAILED
    assert all(message.role != MessageRole.ASSISTANT for message in messages)


def test_runtime_uses_canonical_event_type_enum(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        await make_runtime().run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        return await event_log.query(EventFilter(request_id=submission.request.request_id))

    assert all(isinstance(event.event_type, EventType) for event in asyncio.run(scenario()))


def test_runtime_context_manifest_id_links_model_invocation_to_context_event(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, invocations, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        await make_runtime().run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        invocation_rows = await invocations.list_recent(limit=10)
        context_event = next(event for event in events if event.event_type == EventType.CONTEXT_ASSEMBLED)
        return context_event.payload["context_manifest_id"], invocation_rows[0].context_manifest_id

    context_manifest_id, invocation_context_manifest_id = asyncio.run(scenario())

    assert invocation_context_manifest_id == context_manifest_id


def test_runtime_context_events_preserve_causation_chain(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        await make_runtime().run_turn(
            RuntimeTurnCommand(
                request_id=submission.request.request_id,
                conversation_id=submission.request.conversation_id,
                user_message_id=submission.user_message.message_id,
                user_id="user-1",
                user_input=submission.user_message.content,
                active_project_namespace="project.personal_assistant",
            ),
        )
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        return {event.event_type: event for event in events}

    events = asyncio.run(scenario())

    assert events[EventType.MEMORY_RETRIEVED].causation_id == events[
        EventType.CONTEXT_ASSEMBLY_STARTED
    ].event_id
    assert events[EventType.CONTEXT_ASSEMBLED].causation_id == events[
        EventType.MEMORY_RETRIEVED
    ].event_id
    assert events[EventType.MODEL_REQUEST_CREATED].causation_id == events[
        EventType.CONTEXT_ASSEMBLED
    ].event_id
    assert events[EventType.MODEL_RESPONSE_RECEIVED].causation_id == events[
        EventType.MODEL_REQUEST_CREATED
    ].event_id
    assert events[EventType.ASSISTANT_MESSAGE_CREATED].causation_id == events[
        EventType.MODEL_RESPONSE_RECEIVED
    ].event_id
    assert events[EventType.REQUEST_PROCESSING_COMPLETED].causation_id == events[
        EventType.ASSISTANT_MESSAGE_CREATED
    ].event_id


def test_runtime_secret_policy_denial_is_audited(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(
            store,
            sensitivity=Sensitivity.SECRET,
            content="secret runtime",
        )
        runtime = make_runtime()
        with pytest.raises(Exception):
            await runtime.run_turn(
                RuntimeTurnCommand(
                    request_id=submission.request.request_id,
                    conversation_id=submission.request.conversation_id,
                    user_message_id=submission.user_message.message_id,
                    user_id="user-1",
                    user_input=submission.user_message.content,
                    active_project_namespace="project.personal_assistant",
                    current_message_sensitivity=Sensitivity.SECRET,
                ),
            )
        return await event_log.query(EventFilter(request_id=submission.request.request_id))

    events = asyncio.run(scenario())

    policy_event = next(
        event for event in events if event.event_type == EventType.POLICY_DECISION_RECORDED
    )
    assert policy_event.payload["allowed"] is False
    assert policy_event.payload["source_ref"] == "current_user_message"
    assert events[-1].event_type == EventType.REQUEST_PROCESSING_FAILED


def test_runtime_context_assembly_timeout_marks_request_failed(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        base_settings = ConfigLoader(Path("config")).load("test")
        budget = replace(
            base_settings.runtime_budgets["memory_augmented_answer"],
            max_context_assembly_seconds=0,
        )
        settings = replace(
            base_settings,
            runtime_budgets={"memory_augmented_answer": budget},
        )
        runtime = make_runtime(
            assembler=SlowContextAssembler(),
            runtime_settings=settings,
        )
        with pytest.raises(TimeoutError):
            await runtime.run_turn(
                RuntimeTurnCommand(
                    request_id=submission.request.request_id,
                    conversation_id=submission.request.conversation_id,
                    user_message_id=submission.user_message.message_id,
                    user_id="user-1",
                    user_input=submission.user_message.content,
                    active_project_namespace="project.personal_assistant",
                ),
            )
        request = await store.get_assistant_request(submission.request.request_id)
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        return request, events

    request, events = asyncio.run(scenario())

    assert request.status == RequestStatus.FAILED
    assert request.error_code == "runtime_timeout"
    assert events[-1].event_type == EventType.REQUEST_PROCESSING_FAILED


def test_no_assistant_message_on_system_failure(runtime_parts) -> None:
    async def scenario():
        _, store, _, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(store)
        runtime = make_runtime(model_provider=FakeModelProvider(fail_chat_times=1))
        with pytest.raises(ModelProviderError):
            await runtime.run_turn(
                RuntimeTurnCommand(
                    request_id=submission.request.request_id,
                    conversation_id=submission.request.conversation_id,
                    user_message_id=submission.user_message.message_id,
                    user_id="user-1",
                    user_input=submission.user_message.content,
                    active_project_namespace="project.personal_assistant",
                ),
            )
        return await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=submission.request.conversation_id, limit=10),
        )

    messages = asyncio.run(scenario())

    assert [message.role for message in messages] == [MessageRole.USER]


def test_stream_cancel_after_model_response_does_not_persist_assistant_side_effects(
    runtime_parts,
) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(
            store,
            client_message_id="client-cancel-after-model-response",
        )
        runtime = make_runtime(model_provider=FakeModelProvider(stream_tokens=["answer"]))
        command = RuntimeTurnCommand(
            request_id=submission.request.request_id,
            conversation_id=submission.request.conversation_id,
            user_message_id=submission.user_message.message_id,
            user_id="user-1",
            user_input=submission.user_message.content,
            active_project_namespace="project.personal_assistant",
        )

        emitted = []
        async for event in runtime.stream_turn(command):
            emitted.append(event)
            if event.event_type == EventType.MODEL_RESPONSE_RECEIVED.value:
                await store.update_assistant_request_status(
                    UpdateAssistantRequestStatusCommand(
                        request_id=submission.request.request_id,
                        status=RequestStatus.CANCELLED,
                        error_code="cancelled",
                        error_message="request cancelled",
                    ),
                )

        request = await store.get_assistant_request(submission.request.request_id)
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=submission.request.conversation_id, limit=10),
        )
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        return request, messages, events, emitted

    request, messages, events, emitted = asyncio.run(scenario())

    assert request.status == RequestStatus.CANCELLED
    assert request.assistant_message_id is None
    assert [message.role for message in messages] == [MessageRole.USER]
    assert EventType.ASSISTANT_MESSAGE_CREATED not in [event.event_type for event in events]
    assert EventType.REQUEST_PROCESSING_COMPLETED not in [event.event_type for event in events]
    assert "assistant.message.created" not in [event.event_type for event in emitted]


def test_stream_completion_wins_atomically_after_assistant_message_append(
    runtime_parts,
) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(
            store,
            client_message_id="client-cancel-after-append",
        )
        runtime = make_runtime(
            model_provider=FakeModelProvider(stream_tokens=["answer"]),
            conversation_store_override=CancellingAfterAssistantAppendStore(store),
        )
        command = RuntimeTurnCommand(
            request_id=submission.request.request_id,
            conversation_id=submission.request.conversation_id,
            user_message_id=submission.user_message.message_id,
            user_id="user-1",
            user_input=submission.user_message.content,
            active_project_namespace="project.personal_assistant",
        )

        emitted = [event async for event in runtime.stream_turn(command)]
        request = await store.get_assistant_request(submission.request.request_id)
        messages = await store.load_recent_messages(
            RecentMessagesQuery(conversation_id=submission.request.conversation_id, limit=10),
        )
        events = await event_log.query(EventFilter(request_id=submission.request.request_id))
        return request, messages, events, emitted

    request, messages, events, emitted = asyncio.run(scenario())

    assert request.status == RequestStatus.COMPLETED
    assert request.assistant_message_id is not None
    assert [message.role for message in messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert events[-1].event_type == EventType.REQUEST_PROCESSING_COMPLETED
    assert emitted[-1].event_type == EventType.REQUEST_PROCESSING_COMPLETED.value


def test_stream_model_failure_emits_model_failed_before_request_failed(runtime_parts) -> None:
    async def scenario():
        _, store, event_log, _, make_runtime = runtime_parts
        _, submission = await _accepted_turn(
            store,
            client_message_id="client-stream-model-failure-chain",
        )
        runtime = make_runtime(model_provider=FakeModelProvider(fail_stream_times=1))
        command = RuntimeTurnCommand(
            request_id=submission.request.request_id,
            conversation_id=submission.request.conversation_id,
            user_message_id=submission.user_message.message_id,
            user_id="user-1",
            user_input=submission.user_message.content,
            active_project_namespace="project.personal_assistant",
        )

        emitted = [event async for event in runtime.stream_turn(command)]
        persisted = await event_log.query(EventFilter(request_id=submission.request.request_id))
        return emitted, persisted

    emitted, persisted = asyncio.run(scenario())
    emitted_types = [event.event_type for event in emitted]
    persisted_types = [event.event_type for event in persisted]

    assert EventType.MODEL_REQUEST_FAILED.value in emitted_types
    assert emitted_types[-1] == EventType.REQUEST_PROCESSING_FAILED.value
    assert persisted_types[-2:] == [
        EventType.MODEL_REQUEST_FAILED,
        EventType.REQUEST_PROCESSING_FAILED,
    ]
