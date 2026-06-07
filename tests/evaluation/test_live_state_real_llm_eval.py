from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from assistant_core.app_factory import build_local_providers
from assistant_core.config.settings import ConfigLoader
from assistant_core.context_assembly.rendering import tool_observation_content
from assistant_core.domain.context import AssembledContext, ContextManifest
from assistant_core.domain.conversations import (
    AssistantRequest,
    AssistantResponseCompletion,
    ConversationMessage,
)
from assistant_core.domain.events import EventEnvelope
from assistant_core.domain.loops import LoopBudget, LoopExecutionRequest, LoopStatus, LoopStrategyName
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.model_invocations import ModelInvocationRecord
from assistant_core.domain.policy import PolicyDecision
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import (
    ToolCallRequest,
    ToolInvocationResult,
    ToolObservation,
    ToolObservationStatus,
    ToolParseStatus,
)
from assistant_core.models.router import ModelRouter
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.runtime.routing import CapabilityRoutingRegistry
from assistant_core.tools.builtin import calendar_diff_tool, calculator_tool, datetime_diff_tool


pytestmark = pytest.mark.evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXED_NOW_ISO = "2026-06-07T20:17:00+03:00"


@pytest.mark.parametrize(
    ("prompt", "expected_tools"),
    [
        ("Сколько времени?", ("datetime.now",)),
        (
            "Десятичный логарифм количества часов, прошедших с "
            "2025-09-01T00:00:00+03:00.",
            ("datetime.now", "datetime.diff", "calculator.evaluate"),
        ),
        (
            "Двоичный логарифм количества часов, прошедших с "
            "2025-09-11T00:00:00+03:00.",
            ("datetime.now", "datetime.diff", "calculator.evaluate"),
        ),
    ],
)
def test_real_local_llm_uses_live_state_tools_for_time_delta_prompt(
    prompt: str,
    expected_tools: tuple[str, ...],
) -> None:
    result, gateway = asyncio.run(_run_real_llm_turn(prompt))

    assert result.status is LoopStatus.COMPLETED
    assert result.error is None
    assert [tool_name for tool_name, _arguments in gateway.calls] == list(expected_tools)


def test_real_local_llm_uses_calendar_diff_for_relative_calendar_interval_prompt() -> None:
    result, gateway = asyncio.run(
        _run_real_llm_turn(
            "Количество микросекунд между последним днём Благодарения "
            "(2025-11-27T00:00:00+00:00) и Пасхой "
            "(2026-04-05T00:00:00+00:00).",
            tool_names=("calendar.diff", "datetime.now"),
            live_state_tool_names=("calendar.diff", "datetime.now"),
        )
    )

    assert result.status is LoopStatus.COMPLETED
    assert result.error is None
    assert [tool_name for tool_name, _arguments in gateway.calls] == ["calendar.diff"]


async def _run_real_llm_turn(
    prompt: str,
    *,
    tool_names: tuple[str, ...] = (
        "calculator.evaluate",
        "datetime.diff",
        "datetime.now",
    ),
    live_state_tool_names: tuple[str, ...] = ("datetime.diff", "datetime.now"),
):
    settings = ConfigLoader(PROJECT_ROOT / "config").load(
        profile="ollama",
    )
    profile = settings.model_profiles["local_main"]
    settings = replace(
        settings,
        model_profiles={
            **settings.model_profiles,
            "local_main": replace(
                profile,
                temperature=0.0,
                max_output_tokens=768,
                timeout_seconds=120,
            ),
        },
    )
    event_log = _RecordingEventLog()
    router = ModelRouter(
        settings=settings,
        policy=_AllowAllPolicy(),
        invocation_repository=_InMemoryModelInvocations(),
        providers=build_local_providers(settings),
        event_log=event_log,
    )
    gateway = _TypedEvaluationGateway()
    loop = ToolReactLoop(
        conversation_store=_FakeConversationStore(),
        context_assembler=_EvaluationContextAssembler(),
        model_router=router,
        event_log=event_log,
        tool_gateway=gateway,
    )
    try:
        result = await loop.run_turn(
            LoopExecutionRequest(
                request_id=f"evaluation-request-{uuid4()}",
                conversation_id=f"evaluation-conversation-{uuid4()}",
                user_message_id=f"evaluation-user-message-{uuid4()}",
                user_id="evaluation-user",
                user_input=prompt,
                active_project_namespace="project.personal_assistant",
                current_message_sensitivity=Sensitivity.PROJECT,
                model_profile="local_main",
                strategy_name=LoopStrategyName.TOOL_REACT_LOOP,
                budget=LoopBudget(
                    max_steps=8,
                    max_model_calls=8,
                    max_tool_calls=3,
                    max_wall_time_seconds=120,
                    max_context_assembly_seconds=10,
                    max_model_call_seconds=120,
                    max_consecutive_failures=1,
                ),
                metadata=_tool_plan_metadata(
                    *tool_names,
                    live_state_tool_names=live_state_tool_names,
                ),
            )
        )
    except Exception as exc:
        raise AssertionError(f"real LLM evaluation failed after tool calls: {gateway.calls}") from exc
    return result, gateway


def _tool_plan_metadata(
    *tool_names: str,
    live_state_tool_names: tuple[str, ...],
) -> dict:
    registry = CapabilityRoutingRegistry(enabled_tool_names=frozenset(tool_names))
    return {
        "agent_tool_policy": "available",
        "agent_allowed_tool_names": list(tool_names),
        "agent_allowed_tool_summaries": list(registry.available_tools_summary()),
        "agent_live_state_tool_names": list(live_state_tool_names),
    }


class _EvaluationContextAssembler:
    async def assemble(self, request) -> AssembledContext:
        observation_text = tool_observation_content(list(request.tool_observation_refs))
        system_parts = [
            "You are running a Jarvis local model evaluation.",
            "Follow the output contract exactly.",
            "For tool proposals, return only one JSON object and no markdown.",
            (
                "Tool schemas: datetime.now takes {}; datetime.diff takes "
                '{"from_iso":"timezone-aware ISO","to_iso":"timezone-aware ISO",'
                '"unit":"microseconds|milliseconds|seconds|minutes|hours|days|weeks"}; '
                'calendar.diff takes {"from_iso":"timezone-aware ISO",'
                '"to_iso":"timezone-aware ISO",'
                '"unit":"microseconds|milliseconds|seconds|minutes|hours|days|weeks|months|quarters|decades"}; '
                'calculator.evaluate takes {"expression":"bounded arithmetic expression"}.'
            ),
            (
                "The calculator supports ln(x), log2(x), log10(x), sqrt(x), cbrt(x), "
                "arithmetic operators and numeric constants."
            ),
            f"Output contract: {request.output_contract}",
        ]
        if observation_text:
            system_parts.append("Completed typed tool observations:\n" + observation_text)
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[TextPart(text="\n".join(system_parts))],
                sensitivity=Sensitivity.PROJECT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[TextPart(text=request.current_user_message)],
                sensitivity=Sensitivity.PROJECT,
            ),
        ]
        return AssembledContext(
            messages=messages,
            sections=[],
            manifest=ContextManifest(
                context_manifest_id=f"evaluation-manifest-{uuid4()}",
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                loop_strategy=request.loop_strategy,
                model_profile=request.model_profile,
                section_names=[],
                used_message_ids=[],
                used_memory_ids=[],
                dropped_refs=[],
                token_estimate=sum(
                    len(part.text.split())
                    for message in messages
                    for part in message.content
                ),
                active_namespaces=[],
                retrieval_parameters={},
                max_sensitivity=Sensitivity.PROJECT,
                sources_by_sensitivity={"project": ["current_user_message", "output_contract"]},
                degraded=False,
            ),
            token_estimate=1,
        )


class _TypedEvaluationGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._calendar_diff = calendar_diff_tool()
        self._calculator = calculator_tool()
        self._datetime_diff = datetime_diff_tool()

    async def invoke(self, request: ToolCallRequest) -> ToolObservation:
        self.calls.append((request.tool_name, dict(request.arguments)))
        now = datetime.now(UTC)
        if request.tool_name == "datetime.now":
            return _typed_observation(
                request,
                now=now,
                content=json.dumps({"iso": FIXED_NOW_ISO}, sort_keys=True),
                content_type="application/json",
                structured_schema="datetime.now",
                structured_content={"iso": FIXED_NOW_ISO},
                parse_status=ToolParseStatus.PARSED,
            )
        if request.tool_name == "datetime.diff":
            result = await self._datetime_diff.invoke(request.arguments)
            return _observation_from_result(request, result, now=now)
        if request.tool_name == "calendar.diff":
            result = await self._calendar_diff.invoke(request.arguments)
            return _observation_from_result(request, result, now=now)
        if request.tool_name == "calculator.evaluate":
            result = await self._calculator.invoke(request.arguments)
            return _observation_from_result(request, result, now=now)
        return ToolObservation.empty(
            tool_name=request.tool_name,
            status=ToolObservationStatus.FAILED,
            sensitivity=request.sensitivity,
            started_at=now,
            completed_at=now,
            error={"code": "unknown_tool", "message": "unknown tool"},
        )


def _observation_from_result(
    request: ToolCallRequest,
    result: object,
    *,
    now: datetime,
) -> ToolObservation:
    if isinstance(result, ToolInvocationResult):
        return _typed_observation(
            request,
            now=now,
            content=result.content,
            content_type=result.content_type,
            structured_schema=result.structured_schema,
            structured_content=result.structured_content,
            structured_schema_version=result.structured_schema_version,
            parse_status=result.parse_status,
        )
    content = str(result)
    return _typed_observation(
        request,
        now=now,
        content=content,
        content_type="text/plain",
    )


def _typed_observation(
    request: ToolCallRequest,
    *,
    now: datetime,
    content: str,
    content_type: str,
    structured_schema: str | None = None,
    structured_content: object | None = None,
    structured_schema_version: int | None = 1,
    parse_status: ToolParseStatus | str = ToolParseStatus.NOT_APPLICABLE,
) -> ToolObservation:
    return ToolObservation(
        tool_call_id=f"evaluation-tool-call-{uuid4()}",
        tool_name=request.tool_name,
        status=ToolObservationStatus.COMPLETED,
        content=content,
        content_type=content_type,
        sensitivity=request.sensitivity,
        truncated=False,
        output_bytes=len(content.encode("utf-8")),
        started_at=now,
        completed_at=now,
        duration_ms=0,
        structured_schema=structured_schema,
        structured_schema_version=structured_schema_version if structured_schema else None,
        structured_content=structured_content,
        parse_status=parse_status,
    )


class _AllowAllPolicy:
    async def evaluate_model_request(self, request):
        return PolicyDecision(allowed=True, code="allow", reason="evaluation")

    async def evaluate_memory_write(self, request):
        return PolicyDecision(allowed=True, code="allow", reason="evaluation")

    async def evaluate_context_inclusion(self, request):
        return PolicyDecision(allowed=True, code="allow", reason="evaluation")

    async def evaluate_capability_request(self, request):
        return PolicyDecision(allowed=True, code="allow", reason="evaluation")


class _RecordingEventLog:
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def append(self, event: EventEnvelope) -> EventEnvelope:
        stored = replace(event, event_id=event.event_id or f"evaluation-event-{uuid4()}")
        self.events.append(stored)
        return stored

    async def query(self, event_filter):
        return [
            event
            for event in self.events
            if event_filter.request_id is None or event.request_id == event_filter.request_id
        ]


class _InMemoryModelInvocations:
    def __init__(self) -> None:
        self.records: dict[str, ModelInvocationRecord] = {}

    async def start(self, command) -> ModelInvocationRecord:
        now = datetime.now(UTC)
        record = ModelInvocationRecord(
            model_invocation_id=f"evaluation-model-{uuid4()}",
            request_id=command.request_id,
            conversation_id=command.conversation_id,
            profile=command.profile,
            provider=command.provider,
            model=command.model,
            purpose=command.purpose,
            sensitivity=command.sensitivity,
            status="started",
            started_at=now,
            finished_at=None,
            latency_ms=None,
            input_token_estimate=command.input_token_estimate,
            input_tokens_reported=None,
            output_tokens_reported=None,
            streaming=command.streaming,
            error_type=None,
            error_message=None,
            context_manifest_id=command.context_manifest_id,
            metadata=command.metadata,
        )
        self.records[record.model_invocation_id] = record
        return record

    async def finish(self, command) -> ModelInvocationRecord:
        started = self.records[command.model_invocation_id]
        now = datetime.now(UTC)
        record = replace(
            started,
            status=command.status,
            finished_at=now,
            latency_ms=max(0, int((now - started.started_at).total_seconds() * 1000)),
            input_tokens_reported=command.input_tokens_reported,
            output_tokens_reported=command.output_tokens_reported,
            error_type=command.error_type,
            error_message=command.error_message,
            metadata=command.metadata or started.metadata,
        )
        self.records[record.model_invocation_id] = record
        return record


class _FakeConversationStore:
    async def update_assistant_request_status(self, command):
        return None

    async def complete_assistant_response(self, command) -> AssistantResponseCompletion:
        message = ConversationMessage(
            message_id=f"evaluation-assistant-message-{uuid4()}",
            conversation_id=command.conversation_id,
            request_id=command.request_id,
            event_id=None,
            client_message_id=None,
            role=MessageRole.ASSISTANT,
            content=command.content,
            content_hash="evaluation-hash",
            sensitivity=command.sensitivity,
            created_at=datetime.now(UTC),
        )
        request = AssistantRequest(
            request_id=command.request_id,
            conversation_id=command.conversation_id,
            user_message_id="evaluation-user-message",
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
