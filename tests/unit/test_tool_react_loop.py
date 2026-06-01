from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
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
from assistant_core.runtime.loops.tool_react import TOOL_PROPOSAL_SCHEMA, ToolReactLoop


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
    metadata: dict | None = None,
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
        metadata=metadata or {},
    )


def _tool_plan_metadata(*tool_names: str, policy: str = "available") -> dict:
    return {
        "agent_tool_policy": policy,
        "agent_allowed_tool_names": list(tool_names),
    }


def test_tool_react_loop_requires_toolgateway() -> None:
    with pytest.raises(ValueError):
        ToolReactLoop(
            conversation_store=object(),
            context_assembler=object(),
            model_router=object(),
            event_log=object(),
            tool_gateway=None,
        )


def test_tool_react_loop_budget_requires_positive_step_and_model_limits() -> None:
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
        loop.validate_budget(replace(_budget(), max_model_calls=0))


def test_tool_react_loop_allows_final_answer_when_tools_are_disabled_by_plan() -> None:
    async def scenario():
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeChatRouter("chat answer"),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        return await loop.run_turn(
            _request(
                budget=replace(_budget(), max_tool_calls=0),
                metadata={"agent_tool_policy": "disabled"},
            )
        )

    result = asyncio.run(scenario())

    assert result.response_text == "chat answer"
    assert result.used_tool_calls == 0


def test_tool_react_loop_uses_chat_model_when_tools_are_disabled_by_plan() -> None:
    class ContractContextAssembler(FakeContextAssembler):
        seen_contract: str | None = "not-called"

        async def assemble(self, request):
            self.seen_contract = getattr(request, "output_contract", None)
            return await super().assemble(request)

    async def scenario():
        assembler = ContractContextAssembler()
        router = FakeChatRouter("plain chat answer")
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=assembler,
            model_router=router,
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        result = await loop.run_turn(_request(metadata={"agent_tool_policy": "disabled"}))
        return result, router, assembler

    result, router, assembler = asyncio.run(scenario())

    assert result.response_text == "plain chat answer"
    assert result.used_model_calls == 1
    assert router.chat_calls == 1
    assert router.structured_calls == 0
    assert assembler.seen_contract is None


def test_tool_react_loop_ignores_legacy_direct_plan_when_tools_disabled_by_plan() -> None:
    class Gateway:
        invoked = False

        async def invoke(self, request):
            self.invoked = True
            raise AssertionError("legacy direct plan must not bypass agent request plan")

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeChatRouter("chat answer"),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        result = await loop.run_turn(
            _request(
                metadata={
                    "agent_tool_policy": "disabled",
                    "loop_selection_direct_tool_plan": {
                        "version": 1,
                        "scenario": "current_time",
                        "tool_names": ["datetime.now"],
                        "capabilities": ["tool.safe"],
                        "classification_source": "deterministic",
                        "provenance": ["legacy_fixture"],
                        "required_arguments": {},
                    },
                },
            ),
        )
        return result, gateway.invoked

    result, invoked = asyncio.run(scenario())

    assert result.response_text == "chat answer"
    assert invoked is False


def test_tool_react_loop_does_not_offer_tools_when_tools_are_disabled_by_plan() -> None:
    class Gateway:
        invoked = False

        async def invoke(self, request):
            self.invoked = True
            raise AssertionError("disabled tools must not reach gateway")

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeChatRouter("chat answer"),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        result = await loop.run_turn(_request(metadata={"agent_tool_policy": "disabled"}))
        return result, gateway.invoked

    result, invoked = asyncio.run(scenario())

    assert result.response_text == "chat answer"
    assert invoked is False


def test_tool_react_loop_blocks_tool_call_when_request_plan_metadata_is_missing() -> None:
    class Gateway:
        invoked = False

        async def invoke(self, request):
            self.invoked = True
            raise AssertionError("missing request plan must not reach gateway")

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "datetime.now", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError, match="request_plan_missing_tool_policy"):
            await loop.run_turn(_request(metadata={}))
        return gateway.invoked

    assert asyncio.run(scenario()) is False


def test_tool_react_loop_blocks_tool_call_when_request_plan_policy_is_invalid() -> None:
    class Gateway:
        invoked = False

        async def invoke(self, request):
            self.invoked = True
            raise AssertionError("invalid request plan must not reach gateway")

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "datetime.now", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError, match="request_plan_invalid_tool_policy"):
            await loop.run_turn(_request(metadata={"agent_tool_policy": "legacy"}))
        return gateway.invoked

    assert asyncio.run(scenario()) is False


def test_tool_react_loop_blocks_tool_call_outside_request_plan_allowlist() -> None:
    class Gateway:
        invoked = False

        async def invoke(self, request):
            self.invoked = True
            raise AssertionError("disallowed tools must not reach gateway")

    async def scenario():
        gateway = Gateway()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "tool_call", "tool_name": "daemon.status", "arguments": {}}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        with pytest.raises(RuntimeError, match="tool_not_allowed_by_request_plan"):
            await loop.run_turn(
                _request(
                    metadata={
                        "agent_tool_policy": "available",
                        "agent_allowed_tool_names": ["datetime.now"],
                    },
                )
            )
        return gateway.invoked

    assert asyncio.run(scenario()) is False


def test_tool_react_loop_requires_tool_call_for_required_tool_policy() -> None:
    async def scenario():
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredRouter(
                [{"action": "final_answer"}],
            ),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        with pytest.raises(RuntimeError, match="required_tool_call_missing"):
            await loop.run_turn(
                _request(
                    metadata={
                        "agent_tool_policy": "required",
                        "agent_allowed_tool_names": ["datetime.now"],
                    },
                )
            )

    asyncio.run(scenario())


def test_tool_react_loop_checks_final_chat_budget_before_final_context_assembly() -> None:
    class ContractContextAssembler(FakeContextAssembler):
        calls = 0

        async def assemble(self, request):
            self.calls += 1
            return await super().assemble(request)

    async def scenario():
        assembler = ContractContextAssembler()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=assembler,
            model_router=FakeStructuredAndChatRouter(
                [{"action": "final_answer"}],
                chat_response="would exceed budget",
            ),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        with pytest.raises(RuntimeError, match="max_model_calls_exceeded"):
            await loop.run_turn(
                _request(
                    budget=replace(_budget(), max_model_calls=1),
                    metadata=_tool_plan_metadata("datetime.now"),
                )
            )
        return assembler.calls

    assert asyncio.run(scenario()) == 1


def test_tool_react_loop_passes_tool_proposal_contract_to_context_assembler() -> None:
    class ContractContextAssembler(FakeContextAssembler):
        seen_contracts: list[str | None]

        def __init__(self) -> None:
            self.seen_contracts = []

        async def assemble(self, request):
            self.seen_contracts.append(getattr(request, "output_contract", None))
            return await super().assemble(request)

    async def scenario():
        assembler = ContractContextAssembler()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=assembler,
            model_router=FakeStructuredAndChatRouter(
                [{"action": "final_answer"}],
                chat_response="done",
            ),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        result = await loop.run_turn(
            _request(
                metadata={
                    "agent_tool_policy": "available",
                    "agent_allowed_tool_names": ["datetime.now"],
                },
            )
        )
        return result, assembler.seen_contracts

    result, contracts = asyncio.run(scenario())

    assert result.response_text == "done"
    contract = contracts[0]
    assert contract is not None
    assert "Return only a JSON object" in contract
    assert 'Use {"action":"final_answer"} when ready to answer without another tool.' in contract
    assert '"final_answer":"..."' not in contract
    assert "datetime.now" in contract
    assert "Do not wrap the JSON in markdown" in contract


def test_tool_react_loop_uses_chat_model_for_final_answer_after_proposal() -> None:
    class ContractContextAssembler(FakeContextAssembler):
        seen_contracts: list[str | None]

        def __init__(self) -> None:
            self.seen_contracts = []

        async def assemble(self, request):
            self.seen_contracts.append(getattr(request, "output_contract", None))
            return await super().assemble(request)

    async def scenario():
        assembler = ContractContextAssembler()
        router = FakeStructuredAndChatRouter(
            [{"action": "final_answer"}],
            chat_response="main model answer",
        )
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=assembler,
            model_router=router,
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        result = await loop.run_turn(
            _request(
                metadata={
                    "agent_tool_policy": "available",
                    "agent_allowed_tool_names": ["datetime.now"],
                },
            )
        )
        return result, router, assembler

    result, router, assembler = asyncio.run(scenario())

    assert result.response_text == "main model answer"
    assert result.used_model_calls == 2
    assert router.structured_calls == 1
    assert router.chat_calls == 1
    assert assembler.seen_contracts[0] is not None
    assert "Return only a JSON object" in assembler.seen_contracts[0]
    assert assembler.seen_contracts[1] is None


def test_tool_react_loop_uses_final_chat_when_tool_budget_is_exhausted() -> None:
    class Gateway:
        calls = 0

        async def invoke(self, request):
            from datetime import UTC, datetime
            from assistant_core.domain.tools import ToolObservation

            self.calls += 1
            now = datetime.now(UTC)
            return ToolObservation.empty(
                tool_name=request.tool_name,
                status=ToolObservationStatus.COMPLETED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
            )

    async def scenario():
        gateway = Gateway()
        router = FakeStructuredAndChatRouter(
            [{"action": "tool_call", "tool_name": "datetime.now", "arguments": {}}],
            chat_response="final after budget",
        )
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=router,
            event_log=FakeEventLog(),
            tool_gateway=gateway,
        )
        result = await loop.run_turn(
            _request(
                budget=replace(_budget(), max_tool_calls=1),
                metadata=_tool_plan_metadata("datetime.now"),
            )
        )
        return result, gateway, router

    result, gateway, router = asyncio.run(scenario())

    assert result.response_text == "final after budget"
    assert result.used_tool_calls == 1
    assert gateway.calls == 1
    assert router.structured_calls == 1
    assert router.chat_calls == 1


def test_tool_react_loop_falls_back_to_chat_when_available_proposal_is_non_tool_shape() -> None:
    async def scenario():
        router = FakeStructuredAndChatRouter(
            [{"answer": "not a tool proposal"}],
            chat_response="plain joke",
        )
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=router,
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        result = await loop.run_turn(_request(metadata=_tool_plan_metadata("datetime.now")))
        return result, router

    result, router = asyncio.run(scenario())

    assert result.response_text == "plain joke"
    assert result.used_tool_calls == 0
    assert router.structured_calls == 1
    assert router.chat_calls == 1


def test_parse_tool_proposal_accepts_tool_call_and_final_answer() -> None:
    tool_call = parse_tool_proposal(
        {"action": "tool_call", "tool_name": "fake.echo", "arguments": {"message": "hi"}},
    )
    final = parse_tool_proposal({"action": "final_answer"})

    assert tool_call == ToolProposal(
        action="tool_call",
        tool_name="fake.echo",
        arguments={"message": "hi"},
    )
    assert final.final_answer is None


def test_tool_react_loop_rejects_malformed_tool_proposal() -> None:
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal({"action": "tool_call", "arguments": {"message": "hi"}})
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal({"action": "tool_call", "tool_name": "datetime.now"})
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal({"action": "final_answer", "final_answer": "old structured text"})
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal(
            {"action": "final_answer", "tool_name": "datetime.now", "arguments": {}},
        )
    with pytest.raises(ToolProposalParseError):
        parse_tool_proposal(
            {
                "action": "tool_call",
                "tool_name": "datetime.now",
                "arguments": {},
                "answer": "extra text",
            },
        )


def test_tool_proposal_schema_does_not_request_final_answer_text() -> None:
    schema_text = str(TOOL_PROPOSAL_SCHEMA)
    assert "final_answer" in schema_text
    assert "tool_name" in schema_text
    assert "arguments" in schema_text
    final_answer_schema = next(
        item
        for item in TOOL_PROPOSAL_SCHEMA["oneOf"]
        if item["properties"]["action"]["const"] == "final_answer"
    )
    assert set(final_answer_schema["properties"]) == {"action"}
    assert final_answer_schema["additionalProperties"] is False


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
            await loop.run_turn(_request(metadata=_tool_plan_metadata("missing.tool")))
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
            await loop.run_turn(_request(metadata=_tool_plan_metadata("public.safe")))
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
            await loop.run_turn(
                _request(
                    permission_mode=PermissionMode.LOCKED_DOWN,
                    metadata=_tool_plan_metadata("fake.echo"),
                )
            )
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
            await loop.run_turn(_request(metadata=_tool_plan_metadata("fake.echo")))
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
            model_router=SlowStructuredRouter([{"action": "final_answer"}]),
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
            model_router=FakeStructuredAndChatRouter(
                [{"action": "final_answer"}],
                chat_response="streamed answer",
            ),
            event_log=FakeEventLog(),
            tool_gateway=object(),
        )
        return [
            event
            async for event in loop.stream_turn(
                _request(metadata=_tool_plan_metadata("tool.shell.read.project"))
            )
        ]

    events = asyncio.run(scenario())

    assert ("token", {"delta": "streamed answer"}) in [
        (event.event_type, event.data) for event in events
    ]
    assert events[-1].event_type == "request.processing.completed"
    assert events[-1].data["event_id"] == "event-request.processing.completed"


def test_tool_react_loop_streams_public_tool_lifecycle_events() -> None:
    class Gateway:
        def __init__(self, event_log: FakeEventLog) -> None:
            self._event_log = event_log

        async def get_tool(self, tool_name: str):
            return None

        async def invoke(self, request):
            from assistant_core.domain.tools import ToolObservation

            now = datetime.now(UTC)
            await self._event_log.append(
                EventEnvelope(
                    event_id="",
                    event_seq=0,
                    event_type=EventType.TOOL_SHELL_STARTED,
                    event_version=1,
                    occurred_at=now,
                    recorded_at=now,
                    conversation_id=request.conversation_id,
                    request_id=request.request_id,
                    correlation_id=request.correlation_id,
                    causation_id=request.causation_event_id,
                    parent_event_id=None,
                    actor_type=ActorType.TOOL,
                    actor_id=None,
                    source_component="test",
                    source_node=None,
                    sensitivity=Sensitivity.PROJECT,
                    visibility=EventVisibility.USER_VISIBLE,
                    idempotency_key=None,
                    payload={
                        "tool_name": request.tool_name,
                        "capability": "tool.shell.read",
                        "argv": ["rg", "needle", "docs"],
                        "stdout": "raw output must not stream",
                    },
                )
            )
            return ToolObservation.empty(
                tool_name=request.tool_name,
                status=ToolObservationStatus.COMPLETED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
            )

    async def scenario():
        event_log = FakeEventLog()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=FakeContextAssembler(),
            model_router=FakeStructuredAndChatRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "tool.shell.read.project",
                        "arguments": {},
                    },
                    {"action": "final_answer"},
                ],
                chat_response="done",
            ),
            event_log=event_log,
            tool_gateway=Gateway(event_log),
        )
        return [
            event
            async for event in loop.stream_turn(
                _request(metadata=_tool_plan_metadata("tool.shell.read.project"))
            )
        ]

    events = asyncio.run(scenario())
    tool_events = [event for event in events if event.event_type == EventType.TOOL_SHELL_STARTED.value]

    assert tool_events
    assert tool_events[0].data["tool_name"] == "tool.shell.read.project"
    assert tool_events[0].data["argv"] == ["rg", "needle", "docs"]
    assert "raw output must not stream" not in repr(tool_events[0].data)
    assert events[-1].event_type == EventType.REQUEST_PROCESSING_COMPLETED.value


def test_tool_react_loop_streams_context_phase_events() -> None:
    class RetrievalEventContextAssembler(FakeContextAssembler):
        def __init__(self, event_log: FakeEventLog) -> None:
            self._event_log = event_log

        async def assemble(self, request):
            now = datetime.now(UTC)
            for event_type, payload in [
                (EventType.MEMORY_RETRIEVED, {"retrieved_memory_ids": ["mem-1"]}),
                (
                    EventType.CONTENT_RETRIEVED,
                    {
                        "retrieved_content_refs": [
                            {"chunk_id": "chunk-1", "content_hash": "hash-1"}
                        ],
                        "full_content_stored": False,
                    },
                ),
            ]:
                await self._event_log.append(
                    EventEnvelope(
                        event_id="",
                        event_seq=0,
                        event_type=event_type,
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
                        source_component="test",
                        source_node=None,
                        sensitivity=Sensitivity.PROJECT,
                        visibility=EventVisibility.INTERNAL,
                        idempotency_key=None,
                        payload=payload,
                        metadata={},
                    )
                )
            return await super().assemble(request)

    async def scenario():
        event_log = FakeEventLog()
        loop = ToolReactLoop(
            conversation_store=FakeConversationStore(),
            context_assembler=RetrievalEventContextAssembler(event_log),
            model_router=FakeStructuredAndChatRouter(
                [{"action": "final_answer"}],
                chat_response="done",
            ),
            event_log=event_log,
            tool_gateway=object(),
        )
        return [event async for event in loop.stream_turn(_request())]

    events = asyncio.run(scenario())
    event_types = [event.event_type for event in events]

    assert events[0].event_type == EventType.REQUEST_PROCESSING_STARTED.value
    assert EventType.CONTEXT_ASSEMBLY_STARTED.value in event_types
    assert EventType.MEMORY_RETRIEVED.value in event_types
    assert EventType.CONTENT_RETRIEVED.value in event_types
    assert events[-1].event_type == EventType.REQUEST_PROCESSING_COMPLETED.value
    content_event = next(
        event for event in events if event.event_type == EventType.CONTENT_RETRIEVED.value
    )
    assert content_event.data["hit_count"] == 1
    assert "retrieved_content_refs" not in content_event.data


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


class FakeChatRouter:
    def __init__(self, response: str) -> None:
        self.response = response
        self.chat_calls = 0
        self.structured_calls = 0

    async def chat(self, request):
        from assistant_core.domain.models import ChatModelResponse

        self.chat_calls += 1
        return ChatModelResponse(text=self.response)

    async def structured(self, request):
        self.structured_calls += 1
        raise AssertionError("tools-disabled chat mode must not call structured")


class FakeStructuredAndChatRouter:
    def __init__(self, responses: list[dict], *, chat_response: str) -> None:
        self.responses = responses
        self.chat_response = chat_response
        self.structured_calls = 0
        self.chat_calls = 0

    async def structured(self, request):
        from assistant_core.domain.models import StructuredModelResponse

        self.structured_calls += 1
        return StructuredModelResponse(value=self.responses[self.structured_calls - 1])

    async def chat(self, request):
        from assistant_core.domain.models import ChatModelResponse

        self.chat_calls += 1
        return ChatModelResponse(text=self.chat_response)


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
