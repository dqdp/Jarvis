from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.context import AssembledContext, ContextManifest, ContextSection
from assistant_core.domain.conversations import (
    AssistantRequest,
    AssistantResponseCompletion,
    ConversationMessage,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import ChatModelResponse
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.runtime.agent_runtime import AgentRuntime, RuntimeTurnCommand


pytestmark = pytest.mark.unit


class PromptContextAssembler:
    async def assemble(self, request) -> AssembledContext:
        sections = [
            ContextSection(
                name="system_identity",
                content="You are Jarvis.",
                token_estimate=3,
            ),
            ContextSection(
                name="project_or_environment_memory",
                content="remembered project fact",
                token_estimate=3,
                source_refs=["mem-1"],
            ),
            ContextSection(
                name="current_user_message",
                content=request.current_user_message,
                token_estimate=2,
            ),
        ]
        manifest = ContextManifest(
            context_manifest_id="33333333-3333-3333-3333-333333333333",
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            loop_strategy=request.loop_strategy,
            model_profile=request.model_profile,
            section_names=[section.name for section in sections],
            used_message_ids=[],
            used_memory_ids=["mem-1"],
            dropped_refs=[],
            token_estimate=8,
            active_namespaces=["project.personal_assistant"],
            retrieval_parameters={"max_hits_total": 8},
            max_sensitivity=Sensitivity.PROJECT,
            sources_by_sensitivity={"project": ["current_user_message", "mem-1"]},
            degraded=False,
            full_prompt_stored=False,
        )
        return AssembledContext(
            messages=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=[TextPart(text="You are Jarvis.\nremembered project fact")],
                    sensitivity=Sensitivity.PROJECT,
                    metadata={"source": "context_assembler"},
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=[TextPart(text=request.current_user_message)],
                    sensitivity=Sensitivity.PROJECT,
                ),
            ],
            sections=sections,
            manifest=manifest,
            token_estimate=8,
        )


class ContentEventContextAssembler(PromptContextAssembler):
    def __init__(self, event_log: InMemoryEventLog) -> None:
        self._event_log = event_log

    async def assemble(self, request) -> AssembledContext:
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id="content-retrieved-event",
                event_seq=0,
                event_type=EventType.CONTENT_RETRIEVED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                correlation_id=request.request_id,
                causation_id=request.causation_event_id,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="unit_test_context_assembler",
                source_node=None,
                sensitivity=Sensitivity.PROJECT,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "retrieved_content_refs": [
                        {
                            "source_id": "source-1",
                            "chunk_id": "chunk-1",
                            "citation": "README.md:1-3",
                            "score": 0.8,
                            "content_hash": "hash",
                        }
                    ],
                    "full_content_stored": False,
                },
                metadata={},
            ),
        )
        return await super().assemble(request)


class RecordingModelRouter:
    def __init__(self) -> None:
        self.chat_messages: list[ChatMessage] | None = None
        self.stream_messages: list[ChatMessage] | None = None

    async def chat(self, request):
        self.chat_messages = request.messages
        return ChatModelResponse(text="answer")

    async def stream_chat(self, request):
        self.stream_messages = request.messages
        yield type("StreamEvent", (), {"event_type": "token", "delta": "answer"})()


class FakeConversationStore:
    def __init__(self) -> None:
        self.request = AssistantRequest(
            request_id="11111111-1111-1111-1111-111111111111",
            conversation_id="22222222-2222-2222-2222-222222222222",
            user_message_id="user-message-1",
            assistant_message_id=None,
            status=RequestStatus.ACCEPTED,
            client_message_id="client-1",
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
            error_code=None,
            error_message=None,
        )

    async def update_assistant_request_status(self, command):
        self.request = replace(
            self.request,
            status=command.status,
            assistant_message_id=command.assistant_message_id or self.request.assistant_message_id,
            error_code=command.error_code,
            error_message=command.error_message,
        )

    async def complete_assistant_response(self, command):
        now = datetime.now(UTC)
        message = ConversationMessage(
            message_id="assistant-message-1",
            conversation_id=command.conversation_id,
            request_id=command.request_id,
            event_id=None,
            client_message_id=None,
            role=MessageRole.ASSISTANT,
            content=command.content,
            content_hash="hash",
            sensitivity=command.sensitivity,
            created_at=now,
        )
        self.request = replace(
            self.request,
            status=RequestStatus.COMPLETED,
            assistant_message_id=message.message_id,
            completed_at=now,
        )
        return AssistantResponseCompletion(message=message, request=self.request)

    async def get_assistant_request(self, request_id: str):
        assert request_id == self.request.request_id
        return self.request


def _runtime(
    model_router: RecordingModelRouter,
    store: FakeConversationStore,
    *,
    context_assembler=None,
    event_log=None,
) -> AgentRuntime:
    event_log = event_log or InMemoryEventLog()
    return AgentRuntime(
        conversation_store=store,
        context_assembler=context_assembler or PromptContextAssembler(),
        model_router=model_router,
        event_log=event_log,
        settings=ConfigLoader("config").load("test"),
    )


def _command() -> RuntimeTurnCommand:
    return RuntimeTurnCommand(
        request_id="11111111-1111-1111-1111-111111111111",
        conversation_id="22222222-2222-2222-2222-222222222222",
        user_message_id="user-message-1",
        user_id="user-1",
        user_input="current question",
        active_project_namespace="project.personal_assistant",
        loop_strategy=LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value,
    )


def _message_text(messages: list[ChatMessage]) -> str:
    return "\n".join(part.text for message in messages for part in message.content)


def test_run_turn_passes_context_assembler_messages_to_model_prompt() -> None:
    async def scenario():
        model_router = RecordingModelRouter()
        store = FakeConversationStore()
        await _runtime(model_router, store).run_turn(_command())
        return model_router.chat_messages

    messages = asyncio.run(scenario())

    assert messages is not None
    assert len(messages) == 2
    prompt_text = _message_text(messages)
    assert "You are Jarvis." in prompt_text
    assert "remembered project fact" in prompt_text
    assert messages[0].metadata == {"source": "context_assembler"}
    assert messages[-1].role == MessageRole.USER
    assert messages[-1].content[0].text == "current question"


def test_stream_turn_passes_context_assembler_messages_to_model_prompt() -> None:
    async def scenario():
        model_router = RecordingModelRouter()
        store = FakeConversationStore()
        async for _ in _runtime(model_router, store).stream_turn(_command()):
            pass
        return model_router.stream_messages

    messages = asyncio.run(scenario())

    assert messages is not None
    assert len(messages) == 2
    prompt_text = _message_text(messages)
    assert "You are Jarvis." in prompt_text
    assert "remembered project fact" in prompt_text
    assert messages[0].metadata == {"source": "context_assembler"}
    assert messages[-1].role == MessageRole.USER
    assert messages[-1].content[0].text == "current question"


def test_stream_turn_emits_content_retrieved_phase_event_from_context_assembly() -> None:
    async def scenario():
        model_router = RecordingModelRouter()
        store = FakeConversationStore()
        event_log = InMemoryEventLog()
        runtime = _runtime(
            model_router,
            store,
            context_assembler=ContentEventContextAssembler(event_log),
            event_log=event_log,
        )
        return [event async for event in runtime.stream_turn(_command())]

    emitted = asyncio.run(scenario())

    content_event = next(
        event for event in emitted if event.event_type == EventType.CONTENT_RETRIEVED.value
    )
    assert content_event.data["event_id"] == "content-retrieved-event"
    assert content_event.data["retrieved_content_refs"][0]["chunk_id"] == "chunk-1"
