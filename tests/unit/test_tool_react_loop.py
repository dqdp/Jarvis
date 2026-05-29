from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    LoopStatus,
    LoopStrategyName,
    ToolProposal,
    ToolProposalParseError,
    parse_tool_proposal,
)
from assistant_core.domain.policy import PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.runtime.loops.tool_react import ToolReactLoop


pytestmark = pytest.mark.unit


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


def _request(
    *,
    budget: LoopBudget | None = None,
    permission_mode: PermissionMode | None = None,
) -> LoopExecutionRequest:
    return LoopExecutionRequest(
        request_id="request-tool-react",
        conversation_id="conversation-tool-react",
        user_message_id="message-user",
        user_id="user-1",
        user_input="use a tool",
        active_project_namespace="project.personal_assistant",
        current_message_sensitivity=Sensitivity.PROJECT,
        model_profile="local_main",
        strategy_name=LoopStrategyName.TOOL_REACT_LOOP,
        budget=budget or _budget(),
        permission_mode=permission_mode,
    )


def test_tool_react_loop_requires_toolgateway() -> None:
    with pytest.raises(ValueError):
        ToolReactLoop(
            conversation_store=object(),
            context_assembler=object(),
            model_router=object(),
            event_log=object(),
            tool_gateway=None,
        )


def test_tool_react_loop_budget_requires_positive_step_and_tool_limits() -> None:
    loop = ToolReactLoop(
        conversation_store=object(),
        context_assembler=object(),
        model_router=object(),
        event_log=object(),
        tool_gateway=object(),
    )

    with pytest.raises(ValueError):
        loop.validate_budget(replace(_budget(), max_steps=0))
    with pytest.raises(ValueError):
        loop.validate_budget(replace(_budget(), max_tool_calls=0))


def test_parse_tool_proposal_accepts_tool_call_and_final_answer() -> None:
    tool_call = parse_tool_proposal(
        {"action": "tool_call", "tool_name": "fake.echo", "arguments": {"message": "hi"}},
    )
    final = parse_tool_proposal({"action": "final_answer", "final_answer": "done"})

    assert tool_call == ToolProposal(
        action="tool_call",
        tool_name="fake.echo",
        arguments={"message": "hi"},
    )
    assert final.final_answer == "done"


def test_tool_react_loop_rejects_malformed_tool_proposal() -> None:
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal({"action": "tool_call", "arguments": {"message": "hi"}})
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal({"action": "final_answer"})


def test_tool_react_loop_rejects_unknown_tool_name() -> None:
    class Gateway:
        invoked = False

        async def get_tool(self, tool_name: str):
            return None

        async def invoke(self, request):
            from datetime import UTC, datetime
            from assistant_core.domain.tools import ToolObservation

            self.invoked = True
            now = datetime.now(UTC)
            return ToolObservation.empty(
                tool_name="<redacted>",
                status=ToolObservationStatus.FAILED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
                error={"code": "unknown_tool", "message": "tool is not registered"},
            )

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "missing.tool", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError, match="tool_observation_failed"):
            await loop.run_turn(_request())
        return gateway.invoked

    assert asyncio.run(scenario()) is True


def test_tool_react_loop_stops_on_max_tool_calls() -> None:
    loop = ToolReactLoop(
        conversation_store=object(),
        context_assembler=object(),
        model_router=object(),
        event_log=object(),
        tool_gateway=object(),
    )

    with pytest.raises(RuntimeError):
        loop.ensure_tool_budget(used_tool_calls=1, budget=replace(_budget(), max_tool_calls=1))


def test_tool_react_loop_does_not_downlabel_tool_call_sensitivity() -> None:
    class Gateway:
        seen_sensitivity: Sensitivity | None = None

        async def get_tool(self, tool_name: str):
            from assistant_core.domain.policy import Capability, RiskClass
            from assistant_core.domain.tools import ToolSpec

            return ToolSpec(
                name="public.safe",
                display_name="Public Safe",
                description="Public ceiling safe tool.",
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE}),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                adapter_name="public.safe",
                sensitivity_ceiling=Sensitivity.PUBLIC,
            )

        async def invoke(self, request):
            from datetime import UTC, datetime
            from assistant_core.domain.tools import ToolObservation

            self.seen_sensitivity = request.sensitivity
            now = datetime.now(UTC)
            return ToolObservation.empty(
                tool_name=request.tool_name,
                status=ToolObservationStatus.DENIED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
                error={"code": "sensitivity_ceiling_exceeded", "message": "denied"},
            )

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "public.safe", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError):
            await loop.run_turn(_request())
        return gateway.seen_sensitivity

    assert asyncio.run(scenario()) == Sensitivity.PROJECT


def test_tool_react_loop_passes_permission_mode_to_tool_gateway() -> None:
    class Gateway:
        seen_permission_mode: PermissionMode | str | None = None

        async def get_tool(self, tool_name: str):
            return None

        async def invoke(self, request):
            from datetime import UTC, datetime
            from assistant_core.domain.tools import ToolObservation

            self.seen_permission_mode = request.permission_mode
            now = datetime.now(UTC)
            return ToolObservation.empty(
                tool_name=request.tool_name,
                status=ToolObservationStatus.DENIED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
                error={"code": "denied", "message": "denied"},
            )

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "fake.echo", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError):
            await loop.run_turn(_request(permission_mode=PermissionMode.LOCKED_DOWN))
        return gateway.seen_permission_mode

    assert asyncio.run(scenario()) == PermissionMode.LOCKED_DOWN


def test_tool_react_loop_passes_step_event_causation_to_tool_gateway() -> None:
    class Gateway:
        seen_causation_event_id: str | None = None

        async def get_tool(self, tool_name: str):
            return None

        async def invoke(self, request):
            from datetime import UTC, datetime
            from assistant_core.domain.tools import ToolObservation

            self.seen_causation_event_id = request.causation_event_id
            now = datetime.now(UTC)
            return ToolObservation.empty(
                tool_name=request.tool_name,
                status=ToolObservationStatus.DENIED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
                error={"code": "denied", "message": "denied"},
            )

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "fake.echo", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError):
            await loop.run_turn(_request())
        return gateway.seen_causation_event_id

    assert asyncio.run(scenario()) == "event-agent.step.started"


def test_tool_react_loop_enforces_wall_clock_budget() -> None:
    class SlowStructuredRouter(FakeStructuredRouter):
        async def structured(self, request):
            await asyncio.sleep(0.05)
            return await super().structured(request)

    async def scenario():
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=SlowStructuredRouter([{"action": "final_answer", "final_answer": "late"}]),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        with pytest.raises(RuntimeError, match="max_wall_time_exceeded"):
            await loop.run_turn(
                _request(
                    budget=replace(
                        _budget(),
                        max_wall_time_seconds=0.001,
                        max_model_call_seconds=1,
                    ),
                ),
            )

    asyncio.run(scenario())


def test_tool_react_loop_streams_final_answer_token_and_terminal_event() -> None:
    async def scenario():
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "final_answer", "final_answer": "streamed answer"}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        return [event async for event in loop.stream_turn(_request())]

    events = asyncio.run(scenario())

    assert ("token", {"delta": "streamed answer"}) in [
        (event.event_type, event.data) for event in events
    ]
    assert events[-1].event_type == "request.processing.completed"
    assert events[-1].data["event_id"] == "event-request.processing.completed"


class FakeConversationStore:
    async def update_assistant_request_status(self, command):
        return None

    async def complete_assistant_response(self, command):
        from datetime import UTC, datetime
        from assistant_core.domain.conversations import (
            AssistantRequest,
            AssistantResponseCompletion,
            ConversationMessage,
        )
        from assistant_core.domain.messages import MessageRole
        from assistant_core.domain.requests import RequestStatus

        message = ConversationMessage(
            message_id="assistant-message-tool-react",
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
        request = AssistantRequest(
            request_id=command.request_id,
            conversation_id=command.conversation_id,
            user_message_id="message-user",
            assistant_message_id=message.message_id,
            status=RequestStatus.COMPLETED,
            client_message_id=None,
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=datetime.now(UTC),
            error_code=None,
            error_message=None,
        )
        return AssistantResponseCompletion(message=message, request=request)


class FakeContextAssembler:
    async def assemble(self, request):
        from assistant_core.domain.context import AssembledContext, ContextManifest
        from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart

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
                context_manifest_id="manifest-tool-react",
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


class FakeStructuredRouter:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls = 0

    async def structured(self, request):
        from assistant_core.domain.models import StructuredModelResponse

        self.calls += 1
        return StructuredModelResponse(value=self.responses[self.calls - 1])


class FakeEventLog:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event):
        stored = replace(event, event_id=f"event-{event.event_type.value}")
        self.events.append(stored)
        return stored

    async def query(self, event_filter):
        return [
            event
            for event in self.events
            if event_filter.request_id is None or event.request_id == event_filter.request_id
        ]
