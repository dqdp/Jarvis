from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.conversations import (
    AssistantRequest,
    Conversation,
    ConversationMessage,
    ConversationStatus,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.messages import MessageRole
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.request_command import RuntimeTurnCommandBuilder
from assistant_core.runtime.agent_runtime import RuntimeStreamEvent, RuntimeTurnCommand
from assistant_core.runtime.request_execution import RequestExecutionManager
from assistant_core.runtime.request_lifecycle import RequestLifecycleService
from assistant_core.runtime.request_stream_buffer import RequestStreamBuffer


pytestmark = pytest.mark.unit


def _request_record(*, status: RequestStatus = RequestStatus.RUNNING) -> AssistantRequest:
    now = datetime.now(UTC)
    return AssistantRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        user_message_id="message-user",
        assistant_message_id=None,
        status=status,
        client_message_id="client-1",
        created_at=now,
        started_at=now,
        completed_at=None,
        error_code=None,
        error_message=None,
        metadata={
            "model_profile": "local_small",
            "loop_strategy": "tool_react_loop",
            "working_directory": "/tmp/jarvis-project",
            "agent_tool_policy": "available",
            "agent_allowed_tool_names": ["datetime.now"],
        },
    )


def _message() -> ConversationMessage:
    now = datetime.now(UTC)
    return ConversationMessage(
        message_id="message-user",
        conversation_id="conversation-1",
        request_id="request-1",
        event_id=None,
        client_message_id="client-1",
        role=MessageRole.USER,
        content="hello runtime",
        content_hash="hash",
        sensitivity=Sensitivity.PROJECT,
        created_at=now,
    )


def _conversation() -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        conversation_id="conversation-1",
        user_id="user-1",
        title="Test",
        active_project_namespace="project.personal_assistant",
        status=ConversationStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


class FakeConversationStore:
    def __init__(self) -> None:
        self.request = _request_record()
        self.status_updates = []

    async def get_assistant_request(self, request_id: str):
        assert request_id == self.request.request_id
        return self.request

    async def load_recent_messages(self, query):
        assert query.conversation_id == self.request.conversation_id
        return [_message()]

    async def get_conversation(self, conversation_id: str):
        assert conversation_id == self.request.conversation_id
        return _conversation()

    async def update_assistant_request_status(self, command):
        self.status_updates.append(command)
        self.request = replace(
            self.request,
            status=command.status,
            error_code=command.error_code,
            error_message=command.error_message,
        )
        return self.request


class FakeEventLog:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event):
        self.events.append(event)
        return event

    async def query(self, event_filter):
        return [
            event
            for event in self.events
            if event_filter.request_id is None or event.request_id == event_filter.request_id
        ]


class FastRuntime:
    async def stream_turn(self, command):
        yield RuntimeStreamEvent(
            EventType.REQUEST_PROCESSING_STARTED.value,
            {"request_id": command.request_id, "event_id": "runtime-started"},
        )
        await asyncio.sleep(0)
        yield RuntimeStreamEvent(
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {
                "request_id": command.request_id,
                "event_id": "runtime-completed",
                "assistant_message_id": "message-assistant",
            },
        )


class TokenAfterStartedRuntime:
    async def stream_turn(self, command):
        yield RuntimeStreamEvent(
            EventType.REQUEST_PROCESSING_STARTED.value,
            {"request_id": command.request_id, "event_id": "runtime-started"},
        )
        yield RuntimeStreamEvent("token", {"delta": "answer"})
        yield RuntimeStreamEvent(
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {
                "request_id": command.request_id,
                "event_id": "runtime-completed",
                "assistant_message_id": "message-assistant",
            },
        )


def _event(event_type: EventType, payload: dict) -> EventEnvelope:
    now = datetime.now(UTC)
    return EventEnvelope(
        event_id=f"event-{event_type.value}",
        event_seq=0,
        event_type=event_type,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id="conversation-1",
        request_id="request-1",
        correlation_id="request-1",
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=None,
        source_component="unit_test",
        source_node=None,
        sensitivity=Sensitivity.PROJECT,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload=payload,
        metadata={},
    )


def test_request_stream_buffer_filters_and_replays_public_events() -> None:
    async def scenario():
        buffer = RequestStreamBuffer()
        await buffer.publish("request-1", "debug.internal", {"secret": "hidden"})
        await buffer.publish("request-1", "token", {"delta": "hi", "secret": "hidden"})
        await buffer.publish(
            "request-1",
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {"assistant_message_id": "message-assistant", "secret": "hidden"},
        )

        return buffer.events_from("request-1", 0)

    events = asyncio.run(scenario())

    assert [event.event_type for event in events] == [
        "token",
        EventType.REQUEST_PROCESSING_COMPLETED.value,
    ]
    assert events[0].data == {"request_id": "request-1", "delta": "hi"}
    assert "secret" not in events[1].data


def test_request_stream_buffer_replays_public_tool_lifecycle_events_without_raw_output() -> None:
    async def scenario():
        buffer = RequestStreamBuffer()
        published = await buffer.publish(
            "request-1",
            EventType.TOOL_SHELL_COMPLETED.value,
            {
                "tool_name": "tool.shell.read.project",
                "argv": ["rg", "needle", "docs"],
                "cwd": "/Users/alex/Jarvis",
                "exit_code": 0,
                "output_bytes": 42,
                "stdout": "raw output must not stream",
            },
        )
        return published, buffer.events_from("request-1", 0)

    published, events = asyncio.run(scenario())

    assert published is True
    assert events[0].event_type == EventType.TOOL_SHELL_COMPLETED.value
    assert events[0].data == {
        "request_id": "request-1",
        "tool_name": "tool.shell.read.project",
        "argv": ["rg", "needle", "docs"],
        "exit_code": 0,
        "output_bytes": 42,
    }


def test_request_stream_buffer_replays_loop_selection_events_without_raw_prompt() -> None:
    async def scenario():
        buffer = RequestStreamBuffer()
        published = await buffer.publish(
            "request-1",
            EventType.LOOP_SELECTION_STARTED.value,
            {
                "requested_mode": "auto",
                "raw_prompt": "must not stream",
            },
        )
        return published, buffer.events_from("request-1", 0)

    published, events = asyncio.run(scenario())

    assert published is True
    assert events[0].event_type == EventType.LOOP_SELECTION_STARTED.value
    assert events[0].data == {
        "request_id": "request-1",
        "requested_mode": "auto",
    }


def test_request_stream_buffer_replays_content_retrieval_without_raw_query() -> None:
    async def scenario():
        buffer = RequestStreamBuffer()
        published = await buffer.publish(
            "request-1",
            EventType.CONTENT_RETRIEVED.value,
            {
                "hit_count": 2,
                "query": "must not stream",
            },
        )
        return published, buffer.events_from("request-1", 0)

    published, events = asyncio.run(scenario())

    assert published is True
    assert events[0].event_type == EventType.CONTENT_RETRIEVED.value
    assert events[0].data == {
        "request_id": "request-1",
        "hit_count": 2,
    }


def test_request_stream_buffer_derives_content_hit_count_without_streaming_refs() -> None:
    async def scenario():
        buffer = RequestStreamBuffer()
        published = await buffer.publish(
            "request-1",
            EventType.CONTENT_RETRIEVED.value,
            {
                "retrieved_content_refs": [
                    {"chunk_id": "chunk-1", "content_hash": "hash-1"},
                    {"chunk_id": "chunk-2", "content_hash": "hash-2"},
                ],
                "full_content_stored": False,
            },
        )
        return published, buffer.events_from("request-1", 0)

    published, events = asyncio.run(scenario())

    assert published is True
    assert events[0].data == {
        "request_id": "request-1",
        "hit_count": 2,
        "full_content_stored": False,
    }


def test_request_execution_stream_preserves_started_first_then_replays_pre_start_events() -> None:
    async def scenario():
        settings = ConfigLoader("config").load("test")
        store = FakeConversationStore()
        store.request = _request_record(status=RequestStatus.ACCEPTED)
        event_log = FakeEventLog()
        event_log.events.extend(
            [
                _event(
                    EventType.LOOP_SELECTION_STARTED,
                    {"requested_mode": "auto", "raw_prompt": "must not stream"},
                ),
                _event(
                    EventType.LOOP_SELECTION_COMPLETED,
                    {
                        "requested_mode": "auto",
                        "selected_loop_strategy": "tool_react_loop",
                        "request_plan_status": "selected",
                        "request_plan_reason_code": "request_plan_auto_agent_loop",
                        "raw_prompt": "must not stream",
                    },
                ),
            ],
        )
        manager = RequestExecutionManager(
            runtime=FastRuntime(),
            conversation_store=store,
            event_log=event_log,
            settings=settings,
        )
        await manager.start(store.request)
        events = []
        try:
            async for event in manager.stream("request-1"):
                events.append(event)
                if event.event_type == EventType.REQUEST_PROCESSING_COMPLETED.value:
                    break
        finally:
            await manager.shutdown()
        return events

    events = asyncio.run(scenario())

    assert [event.event_type for event in events[:3]] == [
        EventType.REQUEST_PROCESSING_STARTED.value,
        EventType.LOOP_SELECTION_STARTED.value,
        EventType.LOOP_SELECTION_COMPLETED.value,
    ]
    assert events[1].data == {
        "request_id": "request-1",
        "event_id": f"event-{EventType.LOOP_SELECTION_STARTED.value}",
        "requested_mode": "auto",
    }
    assert "raw_prompt" not in events[2].data


def test_request_execution_seed_does_not_publish_terminal_events_before_tokens() -> None:
    async def scenario():
        settings = ConfigLoader("config").load("test")
        store = FakeConversationStore()
        store.request = _request_record(status=RequestStatus.ACCEPTED)
        event_log = FakeEventLog()
        event_log.events.extend(
            [
                _event(
                    EventType.LOOP_SELECTION_STARTED,
                    {"requested_mode": "auto"},
                ),
                _event(
                    EventType.REQUEST_PROCESSING_COMPLETED,
                    {"assistant_message_id": "message-assistant-from-log"},
                ),
            ],
        )
        manager = RequestExecutionManager(
            runtime=TokenAfterStartedRuntime(),
            conversation_store=store,
            event_log=event_log,
            settings=settings,
        )
        await manager.start(store.request)
        events = []
        try:
            async for event in manager.stream("request-1"):
                events.append(event)
                if event.event_type == EventType.REQUEST_PROCESSING_COMPLETED.value:
                    break
        finally:
            await manager.shutdown()
        return events

    events = asyncio.run(scenario())

    assert [event.event_type for event in events] == [
        EventType.REQUEST_PROCESSING_STARTED.value,
        EventType.LOOP_SELECTION_STARTED.value,
        "token",
        EventType.REQUEST_PROCESSING_COMPLETED.value,
    ]
    assert events[-1].data["assistant_message_id"] == "message-assistant"


def test_runtime_turn_command_builder_uses_request_metadata_and_user_message() -> None:
    async def scenario():
        settings = ConfigLoader("config").load("test")
        store = FakeConversationStore()
        command = await RuntimeTurnCommandBuilder(
            conversation_store=store,
            settings=settings,
        ).build(store.request)
        return command

    command = asyncio.run(scenario())

    assert command.request_id == "request-1"
    assert command.user_input == "hello runtime"
    assert command.active_project_namespace == "project.personal_assistant"
    assert command.model_profile == "local_small"
    assert command.loop_strategy == "tool_react_loop"
    assert command.working_directory == "/tmp/jarvis-project"
    assert command.metadata["agent_allowed_tool_names"] == ["datetime.now"]


def test_runtime_turn_command_builder_defaults_missing_loop_to_agent_loop() -> None:
    async def scenario():
        settings = ConfigLoader("config").load("test")
        store = FakeConversationStore()
        store.request = replace(
            store.request,
            metadata={
                key: value
                for key, value in store.request.metadata.items()
                if key != "loop_strategy"
            },
        )
        command = await RuntimeTurnCommandBuilder(
            conversation_store=store,
            settings=settings,
        ).build(store.request)
        return command

    command = asyncio.run(scenario())

    assert command.loop_strategy == LoopStrategyName.TOOL_REACT_LOOP.value


def test_runtime_turn_command_default_is_agent_loop() -> None:
    command = RuntimeTurnCommand(
        request_id="request-1",
        conversation_id="conversation-1",
        user_message_id="message-user",
        user_id="user-1",
        user_input="hello runtime",
        active_project_namespace="project.personal_assistant",
    )

    assert command.loop_strategy == LoopStrategyName.TOOL_REACT_LOOP.value


def test_request_lifecycle_service_marks_failure_and_publishes_terminal_event() -> None:
    async def scenario():
        store = FakeConversationStore()
        event_log = FakeEventLog()
        buffer = RequestStreamBuffer()
        lifecycle = RequestLifecycleService(
            conversation_store=store,
            event_log=event_log,
            stream_buffer=buffer,
        )

        failed = await lifecycle.mark_failed(
            store.request,
            code="background_task_failed",
            message="request failed in background execution",
        )
        return failed, event_log.events, buffer.events_from("request-1", 0)

    failed, events, stream_events = asyncio.run(scenario())

    assert failed.status == RequestStatus.FAILED
    assert events[0].event_type == EventType.REQUEST_PROCESSING_FAILED
    assert stream_events[0].event_type == EventType.REQUEST_PROCESSING_FAILED.value
    assert stream_events[0].data["error"]["code"] == "background_task_failed"
