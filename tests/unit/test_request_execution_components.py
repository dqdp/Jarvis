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
from assistant_core.domain.events import EventType
from assistant_core.domain.messages import MessageRole
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.request_command import RuntimeTurnCommandBuilder
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
