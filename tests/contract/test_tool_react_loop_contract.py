from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from assistant_core.domain.context import AssembledContext, ContextManifest
from assistant_core.domain.conversations import (
    AssistantRequest,
    AssistantResponseCompletion,
    ConversationMessage,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.loops import LoopBudget, LoopExecutionRequest, LoopStrategyName
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import StructuredModelResponse
from assistant_core.domain.policy import PolicyDecision, PolicyDecisionOutcome
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.tools.builtin import datetime_now_tool
from assistant_core.tools.fake import fake_echo_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry


pytestmark = pytest.mark.contract


class AllowPolicy:
    def __init__(self, outcome: PolicyDecisionOutcome = PolicyDecisionOutcome.ALLOW) -> None:
        self.outcome = outcome

    async def evaluate_capability_request(self, request):
        return PolicyDecision(
            allowed=self.outcome == PolicyDecisionOutcome.ALLOW,
            code=self.outcome.value,
            reason="contract policy decision",
            outcome=self.outcome,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


class ScriptedRouter:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls = 0

    async def structured(self, request):
        self.calls += 1
        return StructuredModelResponse(value=self.responses[self.calls - 1])


class RecordingContextAssembler:
    def __init__(self) -> None:
        self.tool_ref_counts: list[int] = []

    async def assemble(self, request):
        self.tool_ref_counts.append(len(request.tool_observation_refs))
        return AssembledContext(
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[TextPart(text=request.current_user_message)],
                    sensitivity=Sensitivity.PROJECT,
                ),
            ],
            sections=[],
            manifest=ContextManifest(
                context_manifest_id=f"manifest-{len(self.tool_ref_counts)}",
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                loop_strategy=request.loop_strategy,
                model_profile=request.model_profile,
                section_names=[],
                used_message_ids=[],
                used_memory_ids=[],
                dropped_refs=[],
                token_estimate=1,
                active_namespaces=[],
                retrieval_parameters={},
                max_sensitivity=Sensitivity.PROJECT,
                sources_by_sensitivity={"project": ["current_user_message"]},
                degraded=False,
            ),
            token_estimate=1,
        )


class RecordingConversationStore:
    def __init__(self) -> None:
        self.request = AssistantRequest(
            request_id="request-tool-react",
            conversation_id="conversation-tool-react",
            user_message_id="message-user",
            assistant_message_id=None,
            status=RequestStatus.ACCEPTED,
            client_message_id="client-tool-react",
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
            error_code=None,
            error_message=None,
        )
        self.messages: list[ConversationMessage] = []

    async def update_assistant_request_status(self, command):
        self.request = replace(
            self.request,
            status=command.status,
            error_code=command.error_code,
            error_message=command.error_message,
        )

    async def complete_assistant_response(self, command):
        message = ConversationMessage(
            message_id="assistant-message",
            conversation_id=command.conversation_id,
            request_id=command.request_id,
            event_id=None,
            client_message_id=None,
            role=MessageRole.ASSISTANT,
            content=command.content,
            content_hash="hash",
            sensitivity=command.sensitivity,
            created_at=datetime.now(UTC),
        )
        self.messages.append(message)
        self.request = replace(
            self.request,
            status=RequestStatus.COMPLETED,
            assistant_message_id=message.message_id,
            completed_at=datetime.now(UTC),
        )
        return AssistantResponseCompletion(message=message, request=self.request)


def _budget() -> LoopBudget:
    return LoopBudget(
        max_steps=4,
        max_model_calls=4,
        max_tool_calls=2,
        max_wall_time_seconds=60,
        max_context_assembly_seconds=10,
        max_model_call_seconds=60,
        max_consecutive_failures=1,
    )


def _request(*, sensitivity: Sensitivity = Sensitivity.PROJECT) -> LoopExecutionRequest:
    return LoopExecutionRequest(
        request_id="request-tool-react",
        conversation_id="conversation-tool-react",
        user_message_id="message-user",
        user_id="user-1",
        user_input="use a safe tool",
        active_project_namespace="project.personal_assistant",
        current_message_sensitivity=sensitivity,
        model_profile="local_structured",
        strategy_name=LoopStrategyName.TOOL_REACT_LOOP,
        budget=_budget(),
    )


def _loop(*, router: ScriptedRouter, policy: AllowPolicy | None = None):
    store = RecordingConversationStore()
    assembler = RecordingContextAssembler()
    event_log = InMemoryEventLog()
    gateway = ToolGateway(
        registry=ToolRegistry([fake_echo_tool(), datetime_now_tool()]),
        policy=policy or AllowPolicy(),
        event_log=event_log,
    )
    return (
        ToolReactLoop(
            conversation_store=store,
            context_assembler=assembler,
            model_router=router,
            event_log=event_log,
            tool_gateway=gateway,
        ),
        store,
        assembler,
        event_log,
    )


def test_tool_react_loop_executes_fake_tool_then_final_answer() -> None:
    async def scenario():
        loop, store, assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                    {"action": "final_answer", "final_answer": "tool says hello"},
                ],
            ),
        )
        result = await loop.run_turn(_request(sensitivity=Sensitivity.PUBLIC))
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, store, assembler, events

    result, store, assembler, events = asyncio.run(scenario())

    assert result.response_text == "tool says hello"
    assert result.used_model_calls == 2
    assert result.used_tool_calls == 1
    assert store.messages[-1].content == "tool says hello"
    assert all(message.content != "hello" for message in store.messages[:-1])
    assert assembler.tool_ref_counts == [0, 1]
    assert EventType.TOOL_CALL_COMPLETED in [event.event_type for event in events]


def test_tool_react_loop_executes_datetime_tool_then_final_answer() -> None:
    async def scenario():
        loop, _store, _assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {"action": "tool_call", "tool_name": "datetime.now", "arguments": {}},
                    {"action": "final_answer", "final_answer": "time checked"},
                ],
            ),
        )
        result = await loop.run_turn(_request(sensitivity=Sensitivity.PUBLIC))
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, events

    result, events = asyncio.run(scenario())

    assert result.response_text == "time checked"
    assert EventType.TOOL_OBSERVATION_RECORDED in [event.event_type for event in events]


def test_tool_react_loop_handles_denied_tool_observation() -> None:
    async def scenario():
        loop, store, _assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                ],
            ),
            policy=AllowPolicy(PolicyDecisionOutcome.DENY),
        )
        with pytest.raises(RuntimeError):
            await loop.run_turn(_request())
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return store, events

    store, events = asyncio.run(scenario())

    assert store.messages == []
    assert store.request.status == RequestStatus.FAILED
    assert EventType.TOOL_CALL_DENIED in [event.event_type for event in events]
    assert EventType.AGENT_STEP_FAILED in [event.event_type for event in events]


def test_tool_react_loop_handles_approval_required_observation_without_execution() -> None:
    async def scenario():
        loop, store, _assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                ],
            ),
            policy=AllowPolicy(PolicyDecisionOutcome.APPROVAL_REQUIRED),
        )
        with pytest.raises(RuntimeError):
            await loop.run_turn(_request())
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return store, events

    store, events = asyncio.run(scenario())

    assert store.messages == []
    assert store.request.status == RequestStatus.FAILED
    observation = next(event for event in events if event.event_type == EventType.TOOL_OBSERVATION_RECORDED)
    assert observation.payload["status"] == ToolObservationStatus.APPROVAL_REQUIRED.value


def test_tool_react_loop_records_step_events_and_observation_refs() -> None:
    async def scenario():
        loop, _store, _assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                    {"action": "final_answer", "final_answer": "done"},
                ],
            ),
        )
        result = await loop.run_turn(_request())
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, events

    result, events = asyncio.run(scenario())
    event_types = [event.event_type for event in events]
    completed = next(event for event in events if event.event_type == EventType.AGENT_LOOP_COMPLETED)

    assert result.tool_observation_refs
    assert EventType.AGENT_STEP_STARTED in event_types
    assert EventType.AGENT_STEP_COMPLETED in event_types
    assert completed.payload["used_tool_calls"] == 1
    assert completed.payload["tool_observation_refs"] == [
        result.tool_observation_refs[0].tool_call_id,
    ]
