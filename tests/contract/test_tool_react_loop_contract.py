from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from assistant_core.approvals.in_memory import InMemoryApprovalStore
from assistant_core.domain.context import AssembledContext, ContextManifest
from assistant_core.domain.conversations import (
    AssistantRequest,
    AssistantResponseCompletion,
    ConversationMessage,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.loops import LoopBudget, LoopExecutionRequest, LoopStrategyName
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import ChatModelResponse, StructuredModelResponse
from assistant_core.domain.policy import Capability, PolicyDecision, PolicyDecisionOutcome, RiskClass
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolInvocationResult, ToolObservationStatus, ToolParseStatus, ToolSpec
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.tools.builtin import datetime_now_tool
from assistant_core.tools.fake import fake_echo_tool, fake_fail_tool
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
    def __init__(self, responses: list[dict], *, chat_text: str | None = None) -> None:
        self.responses = responses
        self.calls = 0
        self.chat_text = chat_text or "chat answer"
        self.chat_calls = 0

    async def structured(self, request):
        self.calls += 1
        value = self.responses[self.calls - 1]
        return StructuredModelResponse(value=value)

    async def chat(self, request):
        self.chat_calls += 1
        return ChatModelResponse(text=self.chat_text)


def _typed_json_result(
    content: dict,
    *,
    structured_schema: str,
    structured_content: dict | None,
    parse_status: ToolParseStatus = ToolParseStatus.PARSED,
    parse_warnings: tuple[str, ...] = (),
) -> ToolInvocationResult:
    encoded = json.dumps(content, sort_keys=True)
    return ToolInvocationResult(
        content=encoded,
        content_type="application/json",
        truncated=False,
        output_bytes=len(encoded.encode("utf-8")),
        structured_content=structured_content,
        structured_schema=structured_schema,
        structured_schema_version=1,
        parse_status=parse_status,
        parse_warnings=parse_warnings,
    )


def _untyped_json_result(content: dict) -> ToolInvocationResult:
    encoded = json.dumps(content, sort_keys=True)
    return ToolInvocationResult(
        content=encoded,
        content_type="application/json",
        truncated=False,
        output_bytes=len(encoded.encode("utf-8")),
        parse_status=ToolParseStatus.NOT_APPLICABLE,
    )


def _direct_plan_metadata(
    *,
    scenario: str,
    tool_names: list[str],
    capabilities: list[str],
    scope_hint: str | None = None,
    required_arguments: dict[str, str] | None = None,
) -> dict:
    return {
        "loop_selection_direct_tool_plan": {
            "version": 1,
            "scenario": scenario,
            "tool_names": tool_names,
            "capabilities": capabilities,
            "scope_hint": scope_hint,
            "classification_source": "deterministic",
            "provenance": ["contract_fixture"],
            "required_arguments": required_arguments or {},
        }
    }


def _tool_plan_metadata(*tool_names: str, policy: str = "available") -> dict:
    return {
        "agent_tool_policy": policy,
        "agent_allowed_tool_names": list(tool_names),
    }


def _typed_sensor_result(response: dict | ToolInvocationResult) -> ToolInvocationResult:
    if isinstance(response, ToolInvocationResult):
        return response
    return _typed_json_result(
        response,
        structured_schema="system.sensor_snapshot",
        structured_content=response,
    )


def _default_resources_response(arguments: dict) -> ToolInvocationResult:
    argv = arguments.get("argv", ())
    if argv == ["vm_stat"]:
        content = {
            "exit_code": 0,
            "stdout": (
                "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                "Pages free:                               262144.\n"
                "Pages speculative:                        262144.\n"
            ),
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
        return _typed_json_result(
            content,
            structured_schema="system.memory_overview",
            structured_content={"free": "1.00 GiB", "available": "2.00 GiB", "source": "vm_stat"},
            parse_status=ToolParseStatus.PARTIAL,
            parse_warnings=("total_memory_unavailable",),
        )
    if argv == ["top", "-l", "1", "-n", "0"]:
        content = {
            "exit_code": 0,
            "stdout": "Processes: 637 total\nCPU usage: 40.92% user, 28.16% sys, 30.90% idle\n",
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
        return _typed_json_result(
            content,
            structured_schema="system.cpu_overview",
            structured_content={
                "user_percent": 40.92,
                "system_percent": 28.16,
                "idle_percent": 30.9,
                "source": "top",
            },
            parse_status=ToolParseStatus.PARTIAL,
            parse_warnings=("core_count_unavailable",),
        )
    content = {
        "exit_code": 0,
        "stdout": (
            "              total        used        free      shared  buff/cache   available\n"
            "Mem:          32768       12000        1024         128       19744       18000\n"
        ),
        "stderr": "",
        "truncated": {"stdout": False, "stderr": False},
    }
    return _typed_json_result(
        content,
        structured_schema="system.memory_overview",
        structured_content={
            "total": "32768 MiB",
            "used": "12000 MiB",
            "free": "1024 MiB",
            "available": "18000 MiB",
            "used_percent": 36.6,
            "source": "free",
        },
    )


def _default_hardware_response(arguments: dict) -> ToolInvocationResult:
    argv = arguments.get("argv", ())
    if argv == ["lscpu"]:
        content = {
            "exit_code": 0,
            "stdout": "CPU(s):              10\n",
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
    else:
        content = {
            "exit_code": 0,
            "stdout": "10\n",
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
    return _typed_json_result(
        content,
        structured_schema="system.cpu_overview",
        structured_content={"logical_cores": 10, "source": "fake-hardware"},
        parse_status=ToolParseStatus.PARTIAL,
        parse_warnings=("load_unavailable",),
    )


def _default_network_response(arguments: dict) -> ToolInvocationResult:
    argv = arguments.get("argv", ())
    if argv == ["ip", "addr"]:
        content = {
            "exit_code": 0,
            "stdout": "7: wg0: <POINTOPOINT,NOARP,UP,LOWER_UP> mtu 1420 state UP group default\n",
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
        service = "wg0"
    else:
        content = {
            "exit_code": 0,
            "stdout": "* (Connected)   JarvisVPN               [VPN]\n",
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
        service = "JarvisVPN"
    return _typed_json_result(
        content,
        structured_schema="system.vpn_status",
        structured_content={
            "connected": True,
            "interface_or_service": service,
            "evidence": [content["stdout"].strip()],
            "source": "fake-network",
        },
    )


def _typed_process_result(response: dict | ToolInvocationResult) -> ToolInvocationResult:
    if isinstance(response, ToolInvocationResult):
        return response
    if response.get("exit_code") not in {0, 1}:
        return _typed_json_result(
            response,
            structured_schema="system.process_name_search",
            structured_content={
                "query": "HFT",
                "matches": [],
                "error": response.get("stderr") or response.get("stdout") or "process search failed",
                "source": "fake-process",
            },
            parse_status=ToolParseStatus.PARTIAL,
            parse_warnings=("process_search_failed",),
        )
    matches = (
        [{"pid": 12345, "name": "HFT-strategy-runner"}]
        if "HFT-strategy-runner" in str(response.get("stdout", ""))
        else []
    )
    return _typed_json_result(
        response,
        structured_schema="system.process_name_search",
        structured_content={"query": "HFT", "matches": matches, "source": "fake-process"},
    )


class FakeSensorsTool:
    content_type = "application/json"

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response or {
            "source": "fake-sensors",
            "available": True,
            "reason": None,
            "readings": [
                {
                    "label": "cpu",
                    "value": 54.0,
                    "unit": "C",
                    "source": "fake",
                    "metadata": {},
                },
            ],
        }
        self.spec = ToolSpec(
            name="tool.system.read.sensors",
            display_name="System Sensors Diagnostics",
            description="Fake read-only sensor diagnostics.",
            capability=Capability.TOOL_SYSTEM_READ_SENSORS,
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name="fake.system.sensors",
            sensitivity_ceiling=Sensitivity.INFRA,
        )

    async def invoke(self, arguments):
        self.calls.append(arguments)
        return _typed_sensor_result(self.response)


class FakeResourcesTool:
    content_type = "application/json"

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response
        self.spec = ToolSpec(
            name="tool.system.read.resources",
            display_name="System Resources Diagnostics",
            description="Fake read-only resource diagnostics.",
            capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name="fake.system.resources",
            sensitivity_ceiling=Sensitivity.INFRA,
        )

    async def invoke(self, arguments):
        self.calls.append(arguments)
        response = self.response if self.response is not None else _default_resources_response(arguments)
        return response if isinstance(response, ToolInvocationResult) else _untyped_json_result(response)


class FakeHardwareTool:
    content_type = "application/json"

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response
        self.spec = ToolSpec(
            name="tool.system.read.hardware",
            display_name="System Hardware Diagnostics",
            description="Fake read-only hardware diagnostics.",
            capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name="fake.system.hardware",
            sensitivity_ceiling=Sensitivity.INFRA,
        )

    async def invoke(self, arguments):
        self.calls.append(arguments)
        response = self.response if self.response is not None else _default_hardware_response(arguments)
        return response if isinstance(response, ToolInvocationResult) else _untyped_json_result(response)


class FakeProcessTool:
    content_type = "application/json"

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response or {
            "exit_code": 0,
            "stdout": "12345 HFT-strategy-runner\n",
            "stderr": "",
            "truncated": {"stdout": False, "stderr": False},
        }
        self.spec = ToolSpec(
            name="tool.system.read.process",
            display_name="System Process Diagnostics",
            description="Fake read-only process diagnostics.",
            capability=Capability.TOOL_SYSTEM_READ_PROCESS,
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name="fake.system.process",
            sensitivity_ceiling=Sensitivity.INFRA,
        )

    async def invoke(self, arguments):
        self.calls.append(arguments)
        return _typed_process_result(self.response)


class FakeNetworkTool:
    content_type = "application/json"

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response
        self.spec = ToolSpec(
            name="tool.system.read.network",
            display_name="System Network Diagnostics",
            description="Fake read-only network diagnostics.",
            capability=Capability.TOOL_SYSTEM_READ_NETWORK,
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name="fake.system.network",
            sensitivity_ceiling=Sensitivity.INFRA,
        )

    async def invoke(self, arguments):
        self.calls.append(arguments)
        response = self.response if self.response is not None else _default_network_response(arguments)
        return response if isinstance(response, ToolInvocationResult) else _untyped_json_result(response)


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
        self.status_history: list[RequestStatus] = [self.request.status]

    async def update_assistant_request_status(self, command):
        self.request = replace(
            self.request,
            status=command.status,
            error_code=command.error_code,
            error_message=command.error_message,
        )
        self.status_history.append(command.status)

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
        self.status_history.append(RequestStatus.COMPLETED)
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


def _request(
    *,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
    user_input: str = "use a safe tool",
    metadata: dict | None = None,
    working_directory: str | None = None,
) -> LoopExecutionRequest:
    return LoopExecutionRequest(
        request_id="request-tool-react",
        conversation_id="conversation-tool-react",
        user_message_id="message-user",
        user_id="user-1",
        user_input=user_input,
        active_project_namespace="project.personal_assistant",
        current_message_sensitivity=sensitivity,
        model_profile="local_structured",
        strategy_name=LoopStrategyName.TOOL_REACT_LOOP,
        budget=_budget(),
        metadata=metadata or {},
        working_directory=working_directory,
    )


def _loop(
    *,
    router: ScriptedRouter,
    policy: AllowPolicy | None = None,
    approval_store: InMemoryApprovalStore | None = None,
    extra_tools: list | None = None,
):
    store = RecordingConversationStore()
    assembler = RecordingContextAssembler()
    event_log = InMemoryEventLog()
    gateway = ToolGateway(
        registry=ToolRegistry([fake_echo_tool(), datetime_now_tool(), *(extra_tools or [])]),
        policy=policy or AllowPolicy(),
        event_log=event_log,
        approval_store=approval_store,
    )
    return (
        ToolReactLoop(
            conversation_store=store,
            context_assembler=assembler,
            model_router=router,
            event_log=event_log,
            tool_gateway=gateway,
            approval_store=approval_store,
        ),
        store,
        assembler,
        event_log,
    )


def test_tool_react_loop_executes_fake_tool_then_final_answer() -> None:
    async def scenario():
        router = ScriptedRouter(
            [
                {
                    "action": "tool_call",
                    "tool_name": "fake.echo",
                    "arguments": {"message": "hello"},
                },
                {"action": "final_answer"},
            ],
            chat_text="tool says hello",
        )
        loop, store, assembler, event_log = _loop(
            router=router,
        )
        result = await loop.run_turn(
            _request(
                sensitivity=Sensitivity.PUBLIC,
                metadata=_tool_plan_metadata("fake.echo"),
            )
        )
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, store, assembler, router, events

    result, store, assembler, router, events = asyncio.run(scenario())

    assert result.response_text == "tool says hello"
    assert result.used_model_calls == 3
    assert result.used_tool_calls == 1
    assert store.messages[-1].content == "tool says hello"
    assert all(message.content != "hello" for message in store.messages[:-1])
    assert assembler.tool_ref_counts == [0, 1, 1]
    assert router.chat_calls == 1
    assert EventType.TOOL_CALL_COMPLETED in [event.event_type for event in events]


def test_tool_react_loop_executes_datetime_tool_then_final_answer() -> None:
    async def scenario():
        loop, _store, _assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {"action": "tool_call", "tool_name": "datetime.now", "arguments": {}},
                    {"action": "final_answer"},
                ],
                chat_text="time checked",
            ),
        )
        result = await loop.run_turn(
            _request(
                sensitivity=Sensitivity.PUBLIC,
                metadata=_tool_plan_metadata("datetime.now"),
            )
        )
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, events

    result, events = asyncio.run(scenario())

    assert result.response_text == "time checked"
    assert EventType.TOOL_OBSERVATION_RECORDED in [event.event_type for event in events]


def test_tool_react_loop_uses_chat_model_when_tools_are_disabled() -> None:
    async def scenario():
        router = ScriptedRouter([], chat_text="plain answer")
        loop, _store, assembler, event_log = _loop(router=router)
        result = await loop.run_turn(
            _request(
                metadata={
                    "agent_tool_policy": "disabled",
                    "agent_allowed_tool_names": [],
                },
            )
        )
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, router, assembler, events

    result, router, assembler, events = asyncio.run(scenario())

    assert result.response_text == "plain answer"
    assert result.used_model_calls == 1
    assert result.used_tool_calls == 0
    assert router.chat_calls == 1
    assert router.calls == 0
    assert assembler.tool_ref_counts == [0]
    assert EventType.TOOL_CALL_STARTED not in [event.event_type for event in events]


def test_tool_react_loop_ignores_legacy_direct_plan_metadata() -> None:
    async def scenario():
        router = ScriptedRouter(
            [{"action": "final_answer"}],
            chat_text="answer from agent loop",
        )
        loop, _store, assembler, event_log = _loop(router=router)
        result = await loop.run_turn(
            _request(
                user_input="Сколько время?",
                metadata=_direct_plan_metadata(
                    scenario="current_time",
                    tool_names=["datetime.now"],
                    capabilities=["tool.safe"],
                ),
            )
        )
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, router, assembler, events

    result, router, assembler, events = asyncio.run(scenario())

    assert result.response_text == "answer from agent loop"
    assert router.calls == 1
    assert router.chat_calls == 1
    assert assembler.tool_ref_counts == [0, 0]
    assert result.used_model_calls == 2
    assert result.used_tool_calls == 0
    assert EventType.TOOL_OBSERVATION_RECORDED not in [event.event_type for event in events]


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
            await loop.run_turn(_request(metadata=_tool_plan_metadata("fake.echo")))
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return store, events

    store, events = asyncio.run(scenario())

    assert store.messages == []
    assert store.request.status == RequestStatus.FAILED
    assert EventType.TOOL_CALL_DENIED in [event.event_type for event in events]
    assert EventType.AGENT_STEP_FAILED in [event.event_type for event in events]


def test_tool_react_loop_recovers_optional_failed_tool_observation() -> None:
    async def scenario():
        loop, store, assembler, event_log = _loop(
            router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.fail",
                        "arguments": {},
                    },
                ],
                chat_text="Recovered from optional tool failure.",
            ),
            extra_tools=[fake_fail_tool()],
        )
        result = await loop.run_turn(_request(metadata=_tool_plan_metadata("fake.fail")))
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, store, assembler, events

    result, store, assembler, events = asyncio.run(scenario())

    assert result.response_text == "Recovered from optional tool failure."
    assert result.used_tool_calls == 1
    assert result.tool_observation_refs[0].status == ToolObservationStatus.FAILED
    assert store.request.status == RequestStatus.COMPLETED
    assert assembler.tool_ref_counts[-1] == 1
    assert EventType.TOOL_CALL_FAILED in [event.event_type for event in events]
    assert EventType.REQUEST_PROCESSING_FAILED not in [event.event_type for event in events]


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
            await loop.run_turn(_request(metadata=_tool_plan_metadata("fake.echo")))
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return store, events

    store, events = asyncio.run(scenario())

    assert store.messages == []
    assert store.request.status == RequestStatus.FAILED
    observation = next(event for event in events if event.event_type == EventType.TOOL_OBSERVATION_RECORDED)
    assert observation.payload["status"] == ToolObservationStatus.APPROVAL_REQUIRED.value


def test_tool_react_loop_retries_after_granted_approval() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        approval_store = InMemoryApprovalStore(event_log=event_log)
        store = RecordingConversationStore()
        assembler = RecordingContextAssembler()
        gateway = ToolGateway(
            registry=ToolRegistry([fake_echo_tool()]),
            policy=AllowPolicy(PolicyDecisionOutcome.APPROVAL_REQUIRED),
            event_log=event_log,
            approval_store=approval_store,
        )
        loop = ToolReactLoop(
            conversation_store=store,
            context_assembler=assembler,
            model_router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                    {"action": "final_answer"},
                ],
                chat_text="approved",
            ),
            event_log=event_log,
            tool_gateway=gateway,
            approval_store=approval_store,
        )
        task = asyncio.create_task(
            loop.run_turn(
                _request(
                    sensitivity=Sensitivity.PUBLIC,
                    metadata=_tool_plan_metadata("fake.echo"),
                )
            )
        )
        approval_id = None
        for _ in range(100):
            events = await event_log.query(EventFilter(request_id="request-tool-react"))
            approval_event = next(
                (event for event in events if event.event_type == EventType.APPROVAL_REQUIRED),
                None,
            )
            if approval_event is not None:
                approval_id = approval_event.payload["approval_id"]
                await approval_store.grant_approval(approval_id, actor_id="user-1")
                break
            await asyncio.sleep(0.01)
        assert approval_id is not None
        result = await task
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return result, store, events

    result, store, events = asyncio.run(scenario())

    assert result.response_text == "approved"
    assert store.request.status == RequestStatus.COMPLETED
    assert EventType.TOOL_CALL_APPROVED in [event.event_type for event in events]


def test_tool_react_loop_marks_request_waiting_approval_until_decision() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        approval_store = InMemoryApprovalStore(event_log=event_log)
        store = RecordingConversationStore()
        gateway = ToolGateway(
            registry=ToolRegistry([fake_echo_tool()]),
            policy=AllowPolicy(PolicyDecisionOutcome.APPROVAL_REQUIRED),
            event_log=event_log,
            approval_store=approval_store,
        )
        loop = ToolReactLoop(
            conversation_store=store,
            context_assembler=RecordingContextAssembler(),
            model_router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                    {"action": "final_answer"},
                ],
                chat_text="approved",
            ),
            event_log=event_log,
            tool_gateway=gateway,
            approval_store=approval_store,
        )
        task = asyncio.create_task(
            loop.run_turn(
                _request(
                    sensitivity=Sensitivity.PUBLIC,
                    metadata=_tool_plan_metadata("fake.echo"),
                )
            )
        )
        approval_id = None
        for _ in range(100):
            events = await event_log.query(EventFilter(request_id="request-tool-react"))
            approval_event = next(
                (event for event in events if event.event_type == EventType.APPROVAL_REQUIRED),
                None,
            )
            if approval_event is not None and store.request.status == RequestStatus.WAITING_APPROVAL:
                approval_id = approval_event.payload["approval_id"]
                break
            await asyncio.sleep(0.01)
        assert approval_id is not None
        status_while_waiting = store.request.status
        await approval_store.grant_approval(approval_id, actor_id="user-1")
        result = await task
        return status_while_waiting, result, store.status_history

    status_while_waiting, result, status_history = asyncio.run(scenario())

    assert status_while_waiting == RequestStatus.WAITING_APPROVAL
    assert result.response_text == "approved"
    assert status_history == [
        RequestStatus.ACCEPTED,
        RequestStatus.RUNNING,
        RequestStatus.WAITING_APPROVAL,
        RequestStatus.RUNNING,
        RequestStatus.COMPLETED,
    ]


def test_tool_react_loop_streams_failed_terminal_after_denied_approval() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        approval_store = InMemoryApprovalStore(event_log=event_log)
        store = RecordingConversationStore()
        gateway = ToolGateway(
            registry=ToolRegistry([fake_echo_tool()]),
            policy=AllowPolicy(PolicyDecisionOutcome.APPROVAL_REQUIRED),
            event_log=event_log,
            approval_store=approval_store,
        )
        loop = ToolReactLoop(
            conversation_store=store,
            context_assembler=RecordingContextAssembler(),
            model_router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                ],
            ),
            event_log=event_log,
            tool_gateway=gateway,
            approval_store=approval_store,
        )
        emitted = []
        async for event in loop.stream_turn(
            _request(
                sensitivity=Sensitivity.PUBLIC,
                metadata=_tool_plan_metadata("fake.echo"),
            )
        ):
            emitted.append(event)
            if event.event_type == EventType.APPROVAL_REQUIRED.value:
                await approval_store.deny_approval(event.data["approval_id"], actor_id="user-1")
        return emitted, store

    emitted, store = asyncio.run(scenario())

    assert store.request.status == RequestStatus.FAILED
    assert [event.event_type for event in emitted] == [
        EventType.REQUEST_PROCESSING_STARTED.value,
        EventType.CONTEXT_ASSEMBLY_STARTED.value,
        EventType.MODEL_REQUEST_CREATED.value,
        EventType.MODEL_RESPONSE_RECEIVED.value,
        EventType.APPROVAL_REQUIRED.value,
        EventType.APPROVAL_DENIED.value,
        EventType.REQUEST_PROCESSING_FAILED.value,
    ]
    assert [event.event_type for event in emitted].count(
        EventType.REQUEST_PROCESSING_FAILED.value,
    ) == 1
    failed = emitted[-1].data["error"]
    assert failed["code"] == "approval_denied"


def test_tool_react_loop_cancels_pending_approval_when_request_is_cancelled() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        approval_store = InMemoryApprovalStore(event_log=event_log)
        loop = ToolReactLoop(
            conversation_store=RecordingConversationStore(),
            context_assembler=RecordingContextAssembler(),
            model_router=ScriptedRouter(
                [
                    {
                        "action": "tool_call",
                        "tool_name": "fake.echo",
                        "arguments": {"message": "hello"},
                    },
                ],
            ),
            event_log=event_log,
            tool_gateway=ToolGateway(
                registry=ToolRegistry([fake_echo_tool()]),
                policy=AllowPolicy(PolicyDecisionOutcome.APPROVAL_REQUIRED),
                event_log=event_log,
                approval_store=approval_store,
            ),
            approval_store=approval_store,
        )
        task = asyncio.create_task(
            loop.run_turn(
                _request(
                    sensitivity=Sensitivity.PUBLIC,
                    metadata=_tool_plan_metadata("fake.echo"),
                )
            )
        )
        approval_id = None
        for _ in range(100):
            events = await event_log.query(EventFilter(request_id="request-tool-react"))
            approval_event = next(
                (event for event in events if event.event_type == EventType.APPROVAL_REQUIRED),
                None,
            )
            if approval_event is not None:
                approval_id = approval_event.payload["approval_id"]
                break
            await asyncio.sleep(0.01)
        assert approval_id is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        approval = await approval_store.get_approval(approval_id)
        events = await event_log.query(EventFilter(request_id="request-tool-react"))
        return approval, events

    approval, events = asyncio.run(scenario())

    assert approval.status.value == "cancelled"
    assert EventType.APPROVAL_CANCELLED in [event.event_type for event in events]


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
                    {"action": "final_answer"},
                ],
                chat_text="done",
            ),
        )
        result = await loop.run_turn(_request(metadata=_tool_plan_metadata("fake.echo")))
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
