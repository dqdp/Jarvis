from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
import json
import re
import sys
from typing import Any
from uuid import uuid4

from assistant_core.domain.context import ContextAssemblyRequest
from assistant_core.domain.conversations import (
    CompleteAssistantResponseCommand,
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    LoopExecutionResult,
    LoopStreamEvent,
    LoopStatus,
    LoopStrategyName,
    ToolObservationRef,
    ToolProposal,
    ToolProposalParseError,
    parse_tool_proposal,
)
from assistant_core.domain.models import StructuredModelRequest
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import SENSITIVITY_ORDER, ToolObservationStatus
from assistant_core.ports.approvals import ApprovalStorePort
from assistant_core.ports.context_assembler import ContextAssemblerPort
from assistant_core.ports.conversation_store import ConversationStorePort
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.model_router import ModelRouterPort
from assistant_core.ports.tools import ToolGatewayPort
from assistant_core.runtime.loops.tool_approval import ApprovalWaiter
from assistant_core.runtime.loops.tool_proposal_executor import ToolProposalExecutor
from assistant_core.runtime.request_streaming import public_stream_data


TOOL_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"enum": ["tool_call", "final_answer"]},
        "tool_name": {"type": "string"},
        "arguments": {"type": "object"},
        "final_answer": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}

_DIRECT_PATTERN_SHELL_SYNTAX_MARKERS = (
    "|",
    ";",
    "&&",
    "||",
    ">",
    "<",
    "`",
    "$(",
    "\n",
    "\r",
)


class ToolReactLoop:
    strategy_name = LoopStrategyName.TOOL_REACT_LOOP

    def __init__(
        self,
        *,
        conversation_store: ConversationStorePort,
        context_assembler: ContextAssemblerPort,
        model_router: ModelRouterPort,
        event_log: EventLogPort,
        tool_gateway: ToolGatewayPort | None,
        approval_store: ApprovalStorePort | None = None,
    ) -> None:
        if tool_gateway is None:
            raise ValueError("tool_gateway is required")
        self._conversation_store = conversation_store
        self._context_assembler = context_assembler
        self._model_router = model_router
        self._event_log = event_log
        self._tool_gateway = tool_gateway
        self._approval_store = approval_store
        self._proposal_executor = ToolProposalExecutor(
            tool_gateway=tool_gateway,
            conversation_store=conversation_store,
            approval_waiter=ApprovalWaiter(approval_store) if approval_store is not None else None,
        )

    def validate_budget(self, budget: LoopBudget) -> None:
        if budget.max_steps <= 0:
            raise ValueError("tool_react_loop requires positive max_steps")
        if budget.max_model_calls <= 0:
            raise ValueError("tool_react_loop requires positive max_model_calls")
        if budget.max_tool_calls <= 0:
            raise ValueError("tool_react_loop requires positive max_tool_calls")

    def ensure_tool_budget(self, *, used_tool_calls: int, budget: LoopBudget) -> None:
        if used_tool_calls >= budget.max_tool_calls:
            raise RuntimeError("max_tool_calls_exceeded")

    async def run_turn(self, request: LoopExecutionRequest) -> LoopExecutionResult:
        self.validate_budget(request.budget)
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        request_started = await self._append_event(
            EventType.REQUEST_PROCESSING_STARTED,
            request,
            payload={},
        )
        loop_started = await self._append_event(
            EventType.AGENT_LOOP_STARTED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "budget": _budget_payload(request.budget),
            },
            causation_id=request_started.event_id,
        )

        used_model_calls = 0
        used_tool_calls = 0
        consecutive_failures = 0
        context_manifest_refs: list[str] = []
        tool_observation_refs: list[ToolObservationRef] = []
        loop_deadline = asyncio.get_running_loop().time() + float(
            request.budget.max_wall_time_seconds,
        )
        direct_tool_names = _direct_tool_names(request)
        if direct_tool_names:
            return await self._run_direct_tools(
                request,
                tool_names=direct_tool_names,
                loop_started=loop_started,
                loop_deadline=loop_deadline,
            )

        for step_index in range(1, request.budget.max_steps + 1):
            _raise_if_wall_time_exceeded(loop_deadline)
            step_id = str(uuid4())
            step_started = await self._append_event(
                EventType.AGENT_STEP_STARTED,
                request,
                payload={
                    "strategy_name": request.strategy_name.value,
                    "step_id": step_id,
                    "step_index": step_index,
                    "used_model_calls": used_model_calls,
                    "used_tool_calls": used_tool_calls,
                },
                causation_id=loop_started.event_id,
            )
            try:
                context = await asyncio.wait_for(
                    self._context_assembler.assemble(
                        ContextAssemblyRequest(
                            request_id=request.request_id,
                            conversation_id=request.conversation_id,
                            user_id=request.user_id,
                            current_user_message=request.user_input,
                            active_project_namespace=request.active_project_namespace,
                            loop_strategy=request.strategy_name.value,
                            model_profile=request.model_profile,
                            current_message_sensitivity=request.current_message_sensitivity,
                            current_user_message_id=request.user_message_id,
                            causation_event_id=step_started.event_id,
                            permission_mode=request.permission_mode,
                            tool_observation_refs=tuple(tool_observation_refs),
                        ),
                    ),
                    timeout=_remaining_timeout(
                        loop_deadline,
                        request.budget.max_context_assembly_seconds,
                    ),
                )
                context_manifest_refs.append(context.manifest.context_manifest_id)
                if used_model_calls >= request.budget.max_model_calls:
                    raise RuntimeError("max_model_calls_exceeded")
                used_model_calls += 1
                try:
                    model_response = await asyncio.wait_for(
                        self._model_router.structured(
                            StructuredModelRequest(
                                profile=request.model_profile,
                                messages=context.messages,
                                schema=TOOL_PROPOSAL_SCHEMA,
                                sensitivity=context.manifest.max_sensitivity,
                                request_id=request.request_id,
                                conversation_id=request.conversation_id,
                                context_manifest_id=context.manifest.context_manifest_id,
                            ),
                        ),
                        timeout=_remaining_timeout(
                            loop_deadline,
                            request.budget.max_model_call_seconds,
                        ),
                    )
                except TimeoutError as exc:
                    if _wall_time_expired(loop_deadline):
                        raise RuntimeError("max_wall_time_exceeded") from exc
                    raise
                proposal = parse_tool_proposal(model_response.value)
                if proposal.action == "final_answer":
                    return await self._complete(
                        request,
                        proposal,
                        step_started=step_started,
                        used_model_calls=used_model_calls,
                        used_tool_calls=used_tool_calls,
                        context_manifest_refs=tuple(context_manifest_refs),
                        tool_observation_refs=tuple(tool_observation_refs),
                        sensitivity=context.manifest.max_sensitivity,
                    )

                observation_ref = await self._proposal_executor.execute(
                    request,
                    proposal,
                    step_id=step_id,
                    causation_event_id=step_started.event_id,
                    used_tool_calls=used_tool_calls,
                    loop_deadline=loop_deadline,
                )
                used_tool_calls += 1
                tool_observation_refs.append(observation_ref)
                if observation_ref.status != ToolObservationStatus.COMPLETED:
                    raise RuntimeError(f"tool_observation_{observation_ref.status.value}")
                await self._append_event(
                    EventType.AGENT_STEP_COMPLETED,
                    request,
                    payload={
                        "strategy_name": request.strategy_name.value,
                        "step_id": step_id,
                        "step_index": step_index,
                        "action": "tool_call",
                        "tool_name": proposal.tool_name,
                        "tool_call_id": observation_ref.tool_call_id,
                    },
                    causation_id=step_started.event_id,
                )
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                await self._append_event(
                    EventType.AGENT_STEP_FAILED,
                    request,
                    payload={
                        "strategy_name": request.strategy_name.value,
                        "step_id": step_id,
                        "step_index": step_index,
                        "error_code": _error_code(exc),
                        "error_type": type(exc).__name__,
                    },
                    causation_id=step_started.event_id,
                )
                await self._fail(
                    request,
                    exc,
                    causation_id=step_started.event_id,
                    used_model_calls=used_model_calls,
                    used_tool_calls=used_tool_calls,
                    context_manifest_refs=tuple(context_manifest_refs),
                    tool_observation_refs=tuple(tool_observation_refs),
                )
                raise
            if consecutive_failures > request.budget.max_consecutive_failures:
                break

        exc = RuntimeError("max_steps_exceeded")
        await self._fail(
            request,
            exc,
            causation_id=loop_started.event_id,
            used_model_calls=used_model_calls,
            used_tool_calls=used_tool_calls,
            context_manifest_refs=tuple(context_manifest_refs),
            tool_observation_refs=tuple(tool_observation_refs),
        )
        raise exc

    async def _run_direct_tools(
        self,
        request: LoopExecutionRequest,
        *,
        tool_names: tuple[str, ...],
        loop_started: EventEnvelope,
        loop_deadline: float,
    ) -> LoopExecutionResult:
        step_id = str(uuid4())
        step_started = await self._append_event(
            EventType.AGENT_STEP_STARTED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": step_id,
                "step_index": 1,
                "used_model_calls": 0,
                "used_tool_calls": 0,
            },
            causation_id=loop_started.event_id,
        )
        used_tool_calls = 0
        tool_observation_refs: tuple[ToolObservationRef, ...] = ()
        try:
            for tool_name in tool_names:
                observation_ref = await self._proposal_executor.execute(
                    request,
                    ToolProposal(
                        action="tool_call",
                        tool_name=tool_name,
                        arguments=_direct_tool_arguments(tool_name, request),
                    ),
                    step_id=step_id,
                    causation_event_id=step_started.event_id,
                    used_tool_calls=used_tool_calls,
                    loop_deadline=loop_deadline,
                )
                used_tool_calls += 1
                tool_observation_refs = (*tool_observation_refs, observation_ref)
                if observation_ref.status != ToolObservationStatus.COMPLETED:
                    raise RuntimeError(f"tool_observation_{observation_ref.status.value}")
            return await self._complete(
                request,
                ToolProposal(
                    action="final_answer",
                    final_answer=_direct_tools_answer(tool_names, tool_observation_refs, request),
                ),
                step_started=step_started,
                used_model_calls=0,
                used_tool_calls=used_tool_calls,
                context_manifest_refs=(),
                tool_observation_refs=tool_observation_refs,
                sensitivity=_max_ref_sensitivity(
                    request.current_message_sensitivity,
                    tool_observation_refs,
                ),
            )
        except Exception as exc:
            await self._append_event(
                EventType.AGENT_STEP_FAILED,
                request,
                payload={
                    "strategy_name": request.strategy_name.value,
                    "step_id": step_id,
                    "step_index": 1,
                    "error_code": _error_code(exc),
                    "error_type": type(exc).__name__,
                },
                causation_id=step_started.event_id,
            )
            await self._fail(
                request,
                exc,
                causation_id=step_started.event_id,
                used_model_calls=0,
                used_tool_calls=used_tool_calls,
                context_manifest_refs=(),
                tool_observation_refs=tool_observation_refs,
            )
            raise

    async def stream_turn(self, request: LoopExecutionRequest):
        task = asyncio.create_task(self.run_turn(request))
        seen_stream_events: set[str] = set()
        try:
            while not task.done():
                async for event in self._public_stream_events(request, seen_stream_events):
                    yield event
                await asyncio.wait({task}, timeout=0.05)
            async for event in self._public_stream_events(request, seen_stream_events):
                yield event
            result = await task
        except Exception:
            failed_event = await self._latest_event(
                request.request_id,
                EventType.REQUEST_PROCESSING_FAILED,
            )
            yield LoopStreamEvent(
                EventType.REQUEST_PROCESSING_FAILED.value,
                _failed_stream_payload(request, failed_event),
            )
            return
        except BaseException:
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
            raise
        if result.response_text:
            yield LoopStreamEvent("token", {"delta": result.response_text})
        completed_event = await self._latest_event(
            request.request_id,
            EventType.REQUEST_PROCESSING_COMPLETED,
        )
        yield LoopStreamEvent(
            EventType.REQUEST_PROCESSING_COMPLETED.value,
            {
                "request_id": request.request_id,
                "event_id": completed_event.event_id if completed_event is not None else None,
                "assistant_message_id": (
                    result.assistant_message.message_id
                    if result.assistant_message is not None
                    else None
                ),
            },
        )

    async def _latest_event(
        self,
        request_id: str,
        event_type: EventType,
    ) -> EventEnvelope | None:
        events = await self._event_log.query(EventFilter(request_id=request_id))
        for event in reversed(events):
            if event.event_type == event_type:
                return event
        return None

    async def _public_stream_events(
        self,
        request: LoopExecutionRequest,
        seen_event_ids: set[str],
    ):
        events = await self._event_log.query(EventFilter(request_id=request.request_id))
        for event in events:
            if event.event_type not in _USER_STREAM_EVENT_TYPES:
                continue
            if event.event_id in seen_event_ids:
                continue
            seen_event_ids.add(event.event_id)
            yield LoopStreamEvent(
                event.event_type.value,
                public_stream_data(
                    event.event_type.value,
                    {
                        "request_id": request.request_id,
                        "event_id": event.event_id,
                        **event.payload,
                    },
                ),
            )

    async def _complete(
        self,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        *,
        step_started: EventEnvelope,
        used_model_calls: int,
        used_tool_calls: int,
        context_manifest_refs: tuple[str, ...],
        tool_observation_refs: tuple[ToolObservationRef, ...],
        sensitivity: Sensitivity,
    ) -> LoopExecutionResult:
        assert proposal.final_answer is not None
        completion = await self._conversation_store.complete_assistant_response(
            CompleteAssistantResponseCommand(
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                content=proposal.final_answer,
                sensitivity=sensitivity,
            ),
        )
        assistant_event = await self._append_event(
            EventType.ASSISTANT_MESSAGE_CREATED,
            request,
            payload={
                "message_id": completion.message.message_id,
                "content_hash": completion.message.content_hash,
            },
            causation_id=step_started.event_id,
            sensitivity=sensitivity,
        )
        await self._append_event(
            EventType.AGENT_STEP_COMPLETED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": step_started.payload["step_id"],
                "step_index": step_started.payload["step_index"],
                "action": "final_answer",
            },
            causation_id=step_started.event_id,
            sensitivity=sensitivity,
        )
        loop_completed = await self._append_event(
            EventType.AGENT_LOOP_COMPLETED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "status": LoopStatus.COMPLETED.value,
                "used_model_calls": used_model_calls,
                "used_tool_calls": used_tool_calls,
                "context_manifest_refs": list(context_manifest_refs),
                "tool_observation_refs": [
                    ref.tool_call_id for ref in tool_observation_refs
                ],
            },
            causation_id=assistant_event.event_id,
            sensitivity=sensitivity,
        )
        await self._append_event(
            EventType.REQUEST_PROCESSING_COMPLETED,
            request,
            payload={"assistant_message_id": completion.message.message_id},
            causation_id=loop_completed.event_id,
            sensitivity=sensitivity,
        )
        return LoopExecutionResult(
            status=LoopStatus.COMPLETED,
            response_text=proposal.final_answer,
            assistant_message=completion.message,
            used_model_calls=used_model_calls,
            used_tool_calls=used_tool_calls,
            context_manifest_refs=context_manifest_refs,
            tool_observation_refs=tool_observation_refs,
            degraded=False,
        )

    async def _fail(
        self,
        request: LoopExecutionRequest,
        exc: Exception,
        *,
        causation_id: str | None,
        used_model_calls: int,
        used_tool_calls: int,
        context_manifest_refs: tuple[str, ...],
        tool_observation_refs: tuple[ToolObservationRef, ...],
    ) -> None:
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error_code=_error_code(exc),
                error_message="tool loop failed",
            ),
        )
        loop_failed = await self._append_event(
            EventType.AGENT_LOOP_FAILED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "status": LoopStatus.FAILED.value,
                "error_code": _error_code(exc),
                "error_type": type(exc).__name__,
                "used_model_calls": used_model_calls,
                "used_tool_calls": used_tool_calls,
                "context_manifest_refs": list(context_manifest_refs),
                "tool_observation_refs": [
                    ref.tool_call_id for ref in tool_observation_refs
                ],
            },
            causation_id=causation_id,
        )
        await self._append_event(
            EventType.REQUEST_PROCESSING_FAILED,
            request,
            payload={
                "error_type": type(exc).__name__,
                "error_code": _error_code(exc),
                "error": {
                    "code": _error_code(exc),
                    "message": "tool loop failed",
                    "request_id": request.request_id,
                    "details": {},
                },
            },
            causation_id=loop_failed.event_id,
        )

    async def _append_event(
        self,
        event_type: EventType,
        request: LoopExecutionRequest,
        *,
        payload: dict[str, Any],
        causation_id: str | None = None,
        sensitivity: Sensitivity = Sensitivity.PROJECT,
    ) -> EventEnvelope:
        now = datetime.now(UTC)
        return await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=event_type,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                correlation_id=request.correlation_id or request.request_id,
                causation_id=causation_id,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="tool_react_loop",
                source_node=None,
                sensitivity=sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload=payload,
                metadata={},
            ),
        )


def _budget_payload(budget: LoopBudget) -> dict[str, int]:
    return {
        "max_steps": budget.max_steps,
        "max_model_calls": budget.max_model_calls,
        "max_tool_calls": budget.max_tool_calls,
        "max_wall_time_seconds": budget.max_wall_time_seconds,
    }


def _remaining_timeout(deadline: float, operation_timeout: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise RuntimeError("max_wall_time_exceeded")
    return min(float(operation_timeout), remaining)


def _raise_if_wall_time_exceeded(deadline: float) -> None:
    if _wall_time_expired(deadline):
        raise RuntimeError("max_wall_time_exceeded")


def _wall_time_expired(deadline: float) -> bool:
    return asyncio.get_running_loop().time() >= deadline


def _direct_tool_names(request: LoopExecutionRequest) -> tuple[str, ...]:
    tool_names = request.metadata.get("loop_selection_direct_tool_names")
    if isinstance(tool_names, list):
        accepted = tuple(tool_name for tool_name in tool_names if _is_direct_tool_name(tool_name))
        return accepted if len(accepted) == len(tool_names) else ()
    tool_name = request.metadata.get("loop_selection_direct_tool_name")
    if _is_direct_tool_name(tool_name):
        if (
            tool_name == "tool.system.read.process"
            and request.metadata.get("loop_selection_direct_scenario") == "process_name_search"
            and _process_search_pattern(request.user_input) is None
        ):
            return ()
        return (tool_name,)
    return ()


def _is_direct_tool_name(tool_name: object) -> bool:
    return tool_name in {
        "datetime.now",
        "tool.system.read.hardware",
        "tool.system.read.network",
        "tool.system.read.process",
        "tool.system.read.resources",
        "tool.system.read.sensors",
    }


def _max_sensitivity(first: Sensitivity, second: Sensitivity) -> Sensitivity:
    return first if SENSITIVITY_ORDER[first] >= SENSITIVITY_ORDER[second] else second


def _max_ref_sensitivity(
    base: Sensitivity,
    refs: tuple[ToolObservationRef, ...],
) -> Sensitivity:
    result = base
    for ref in refs:
        result = _max_sensitivity(result, ref.sensitivity)
    return result


def _direct_tool_arguments(tool_name: str, request: LoopExecutionRequest) -> dict[str, Any]:
    if tool_name == "datetime.now":
        return {}
    if tool_name == "tool.system.read.hardware":
        return {
            "argv": _hardware_snapshot_argv(request),
            "cwd": request.working_directory or ".",
        }
    if tool_name == "tool.system.read.sensors":
        return {
            "argv": _sensor_snapshot_argv(),
            "cwd": request.working_directory or ".",
        }
    if tool_name == "tool.system.read.process":
        return {
            "argv": _process_name_search_argv(request),
            "cwd": request.working_directory or ".",
        }
    if tool_name == "tool.system.read.resources":
        return {
            "argv": _resource_snapshot_argv(request),
            "cwd": request.working_directory or ".",
        }
    if tool_name == "tool.system.read.network":
        return {
            "argv": _network_snapshot_argv(request),
            "cwd": request.working_directory or ".",
        }
    return {}


def _hardware_snapshot_argv(request: LoopExecutionRequest) -> list[str]:
    if request.metadata.get("loop_selection_direct_scenario") == "os_version":
        return _os_version_argv()
    if request.metadata.get("loop_selection_direct_scenario") == "battery_charge":
        return _battery_snapshot_argv()
    if sys.platform == "darwin":
        return ["sysctl", "-n", "hw.logicalcpu"]
    if sys.platform.startswith("linux"):
        return ["lscpu"]
    return ["lscpu"]


def _os_version_argv() -> list[str]:
    if sys.platform == "darwin":
        return ["sw_vers"]
    if sys.platform.startswith("linux"):
        return ["uname", "-a"]
    return ["uname", "-a"]


def _battery_snapshot_argv() -> list[str]:
    if sys.platform == "darwin":
        return ["pmset", "-g", "batt"]
    return ["upower", "-i", "/org/freedesktop/UPower/devices/DisplayDevice"]


def _sensor_snapshot_argv() -> list[str]:
    if sys.platform == "darwin":
        return ["powermetrics", "--samplers", "thermal", "-n", "1"]
    if sys.platform.startswith("linux"):
        return ["thermal-sysfs"]
    return ["sensors"]


def _resource_snapshot_argv(request: LoopExecutionRequest) -> list[str]:
    if request.metadata.get("loop_selection_direct_scenario") == "disk_free":
        return ["df", "-h"]
    if sys.platform == "darwin":
        if "tool.system.read.hardware" in _direct_tool_names(request):
            return ["top", "-l", "1", "-n", "0"]
        return ["vm_stat"]
    if sys.platform.startswith("linux"):
        if "tool.system.read.hardware" in _direct_tool_names(request):
            return ["top", "-b", "-n", "1"]
        return ["free", "-m"]
    return ["uptime"]


def _network_snapshot_argv(request: LoopExecutionRequest) -> list[str]:
    if request.metadata.get("loop_selection_direct_scenario") == "vpn_status":
        if sys.platform == "darwin":
            return ["scutil", "--nc", "list"]
        if sys.platform.startswith("linux"):
            return ["ip", "addr"]
    if sys.platform == "darwin":
        return ["ifconfig"]
    return ["ip", "addr"]


def _process_name_search_argv(request: LoopExecutionRequest) -> list[str]:
    pattern = _process_search_pattern(request.user_input)
    if pattern is None:
        pattern = "__jarvis_missing_process_name__"
    return ["pgrep", "-l", re.escape(pattern)]


def _process_search_pattern(text: str) -> str | None:
    return _quoted_process_pattern(text) or _unquoted_process_pattern(text)


def _quoted_process_pattern(text: str) -> str | None:
    for pattern in (
        r'"([^"]+)"',
        r"'([^']+)'",
        r"«([^»]+)»",
    ):
        match = re.search(pattern, text)
        if match is None:
            continue
        value = match.group(1).strip()
        if _safe_direct_pattern(value):
            return value
    return None


def _unquoted_process_pattern(text: str) -> str | None:
    for pattern in (
        r"(?:process|процесс(?:а|е|ом)?)(?:\s+(?:named|called|с именем|имени|под названием))?\s+(?P<value>[A-Za-z0-9_.:-]{1,128})",
        r"(?P<value>[A-Za-z0-9_.:-]{1,128})\s+(?:process|процесс(?:а|е|ом)?)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        value = match.group("value").strip()
        if value.casefold() in {"process", "процесс", "now", "сейчас"}:
            continue
        if _safe_direct_pattern(value):
            return value
    return None


def _safe_direct_pattern(value: str) -> bool:
    return 0 < len(value) <= 128 and not any(
        marker in value for marker in _DIRECT_PATTERN_SHELL_SYNTAX_MARKERS
    )


def _direct_tools_answer(
    tool_names: tuple[str, ...],
    observation_refs: tuple[ToolObservationRef, ...],
    request: LoopExecutionRequest,
) -> str:
    refs_by_name = {ref.tool_name: ref for ref in observation_refs}
    if (
        request.metadata.get("loop_selection_direct_scenario") == "christmas_countdown"
        and tool_names == ("datetime.now",)
    ):
        return _christmas_countdown_answer(refs_by_name["datetime.now"])
    if (
        request.metadata.get("loop_selection_direct_scenario") == "battery_charge"
        and tool_names == ("tool.system.read.hardware",)
    ):
        return _battery_charge_answer(refs_by_name["tool.system.read.hardware"])
    if (
        request.metadata.get("loop_selection_direct_scenario") == "disk_free"
        and tool_names == ("tool.system.read.resources",)
    ):
        return _disk_free_answer(refs_by_name["tool.system.read.resources"])
    if (
        request.metadata.get("loop_selection_direct_scenario") == "os_version"
        and tool_names == ("tool.system.read.hardware",)
    ):
        return _os_version_answer(refs_by_name["tool.system.read.hardware"])
    if tool_names == ("tool.system.read.hardware", "tool.system.read.resources"):
        return _cpu_overview_answer(
            refs_by_name["tool.system.read.hardware"],
            refs_by_name["tool.system.read.resources"],
        )
    if (
        request.metadata.get("loop_selection_direct_scenario") == "vpn_status"
        and tool_names == ("tool.system.read.network",)
    ):
        return _vpn_status_answer(refs_by_name["tool.system.read.network"])
    return _direct_tool_answer(tool_names[-1], observation_refs[-1])


def _direct_tool_answer(
    tool_name: str,
    observation_ref: ToolObservationRef,
) -> str:
    if tool_name == "datetime.now":
        try:
            payload = json.loads(observation_ref.content)
        except json.JSONDecodeError:
            payload = {}
        iso_timestamp = payload.get("iso") if isinstance(payload, dict) else None
        if isinstance(iso_timestamp, str) and iso_timestamp:
            return f"Текущее локальное время: {iso_timestamp}."
    if tool_name == "tool.system.read.resources":
        return _resource_snapshot_answer(observation_ref)
    if tool_name == "tool.system.read.process":
        return _process_name_search_answer(observation_ref)
    if tool_name == "tool.system.read.sensors":
        return _sensor_snapshot_answer(observation_ref)
    return observation_ref.content


def _christmas_countdown_answer(observation_ref: ToolObservationRef) -> str:
    try:
        payload = json.loads(observation_ref.content)
    except json.JSONDecodeError:
        payload = {}
    iso_timestamp = payload.get("iso") if isinstance(payload, dict) else None
    if not isinstance(iso_timestamp, str) or not iso_timestamp:
        return "Не удалось прочитать текущую дату."
    try:
        current_date = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")).date()
    except ValueError:
        return "Не удалось разобрать текущую дату."
    western = _days_until_month_day(current_date, 12, 25)
    orthodox = _days_until_month_day(current_date, 1, 7)
    return (
        f"До Рождества 25 декабря: {western} дней. "
        f"До православного Рождества 7 января: {orthodox} дней."
    )


def _days_until_month_day(current_date: date, month: int, day: int) -> int:
    target = date(current_date.year, month, day)
    if target < current_date:
        target = date(current_date.year + 1, month, day)
    return (target - current_date).days


def _battery_charge_answer(observation_ref: ToolObservationRef) -> str:
    stdout = _tool_stdout(observation_ref)
    if stdout is None:
        return "Не удалось прочитать заряд аккумулятора."
    match = re.search(r"\b(\d{1,3})%;\s*([^;\n]+)", stdout)
    if match is None:
        return "Не удалось разобрать заряд аккумулятора."
    percent = match.group(1)
    state = _battery_state_label(match.group(2).strip())
    return f"Аккумулятор: {percent}% ({state})."


def _battery_state_label(raw_state: str) -> str:
    state = raw_state.casefold()
    if "discharging" in state:
        return "разряжается"
    if "charging" in state:
        return "заряжается"
    if "charged" in state:
        return "заряжен"
    return raw_state


def _disk_free_answer(observation_ref: ToolObservationRef) -> str:
    try:
        payload = json.loads(observation_ref.content)
    except json.JSONDecodeError:
        return observation_ref.content
    if not isinstance(payload, dict):
        return observation_ref.content
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    exit_code = payload.get("exit_code")
    if not isinstance(stdout, str):
        return observation_ref.content
    if exit_code not in {0, None}:
        detail = stderr if isinstance(stderr, str) and stderr.strip() else stdout
        return f"Не удалось прочитать свободное место на диске: {detail.strip()}"
    parsed = _parse_df_snapshot(stdout)
    if parsed is None:
        compact = "\n".join(line for line in stdout.strip().splitlines()[:12])
        return f"Свободное место на диске:\n{compact}"
    return (
        f"Диск {parsed['mount']}: свободно {parsed['available']} "
        f"из {parsed['size']} (использовано {parsed['used_percent']})."
    )


def _parse_df_snapshot(stdout: str) -> dict[str, str] | None:
    rows = [line.split() for line in stdout.splitlines() if line.strip()]
    if len(rows) < 2:
        return None
    data_rows = rows[1:]
    selected = next((row for row in data_rows if row and row[-1] == "/"), data_rows[0])
    if len(selected) < 6:
        return None
    return {
        "size": selected[1],
        "available": selected[3],
        "used_percent": selected[4],
        "mount": selected[-1],
    }


def _vpn_status_answer(observation_ref: ToolObservationRef) -> str:
    stdout = _tool_stdout(observation_ref)
    if stdout is None:
        return "Не удалось прочитать статус VPN."
    connected_lines = [
        line.strip()
        for line in stdout.splitlines()
        if "(connected)" in line.casefold()
    ]
    if connected_lines:
        return f"VPN включен: {connected_lines[0]}"
    if _linux_vpn_interface_is_up(stdout):
        return "VPN включен: обнаружен активный VPN-интерфейс."
    return "VPN не включен или активное VPN-подключение не найдено."


def _linux_vpn_interface_is_up(stdout: str) -> bool:
    vpn_interface_markers = ("tun", "tap", "wg", "vpn", "utun")
    for block in re.split(r"\n(?=\d+:\s)", stdout.strip()):
        header = block.splitlines()[0].strip() if block.strip() else ""
        lowered_header = header.casefold()
        name_match = re.match(r"\d+:\s+([^:@\s]+)", header)
        interface_name = name_match.group(1).casefold() if name_match else ""
        if not any(
            marker in interface_name or marker in lowered_header
            for marker in vpn_interface_markers
        ):
            continue
        flags_match = re.search(r"<([^>]+)>", header)
        flags = {
            flag.strip().casefold()
            for flag in (flags_match.group(1).split(",") if flags_match else ())
        }
        if "state up" in lowered_header or "up" in flags:
            return True
    return False


def _cpu_overview_answer(
    hardware_ref: ToolObservationRef,
    resources_ref: ToolObservationRef,
) -> str:
    cores = _hardware_cpu_cores(hardware_ref)
    cpu_usage = _top_cpu_usage(resources_ref)
    parts = []
    if cores is not None:
        parts.append(f"CPU: {cores} логических ядер")
    if cpu_usage is not None:
        parts.append(
            "загрузка: "
            f"{cpu_usage['user']}% user, "
            f"{cpu_usage['system']}% sys, "
            f"{cpu_usage['idle']}% idle"
        )
    if parts:
        return "; ".join(parts) + "."
    return "Не удалось разобрать сведения о ядрах CPU и текущей загрузке."


def _os_version_answer(observation_ref: ToolObservationRef) -> str:
    stdout = _tool_stdout(observation_ref)
    if stdout is None:
        return "Не удалось прочитать версию операционной системы."
    sw_vers = _parse_sw_vers(stdout)
    if sw_vers:
        name = sw_vers.get("ProductName") or "macOS"
        version = sw_vers.get("ProductVersion")
        build = sw_vers.get("BuildVersion")
        if version and build:
            return f"Операционная система: {name} {version} (build {build})."
        if version:
            return f"Операционная система: {name} {version}."
    first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    if first_line:
        return f"Операционная система: {first_line}."
    return "Не удалось разобрать версию операционной системы."


def _parse_sw_vers(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _hardware_cpu_cores(observation_ref: ToolObservationRef) -> int | None:
    stdout = _tool_stdout(observation_ref)
    if stdout is None:
        return None
    first_number = re.search(r"\b(\d+)\b", stdout)
    return int(first_number.group(1)) if first_number else None


def _top_cpu_usage(observation_ref: ToolObservationRef) -> dict[str, str] | None:
    stdout = _tool_stdout(observation_ref)
    if stdout is None:
        return None
    macos = re.search(
        r"CPU usage:\s*([0-9.]+)% user,\s*([0-9.]+)% sys,\s*([0-9.]+)% idle",
        stdout,
    )
    if macos:
        return {
            "user": macos.group(1),
            "system": macos.group(2),
            "idle": macos.group(3),
        }
    linux = re.search(
        r"%Cpu\(s\):\s*([0-9.]+)\s*us,\s*([0-9.]+)\s*sy,.*?([0-9.]+)\s*id",
        stdout,
    )
    if linux:
        return {
            "user": linux.group(1),
            "system": linux.group(2),
            "idle": linux.group(3),
        }
    return None


def _tool_stdout(observation_ref: ToolObservationRef) -> str | None:
    try:
        payload = json.loads(observation_ref.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    stdout = payload.get("stdout")
    return stdout if isinstance(stdout, str) else None


def _resource_snapshot_answer(observation_ref: ToolObservationRef) -> str:
    try:
        payload = json.loads(observation_ref.content)
    except json.JSONDecodeError:
        return observation_ref.content
    if not isinstance(payload, dict):
        return observation_ref.content
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    exit_code = payload.get("exit_code")
    if not isinstance(stdout, str):
        return observation_ref.content
    if exit_code not in {0, None}:
        detail = stderr if isinstance(stderr, str) and stderr.strip() else stdout
        return f"Не удалось прочитать память: {detail.strip()}"
    if free_answer := _free_memory_answer(stdout):
        return free_answer
    if vm_stat_answer := _vm_stat_memory_answer(stdout):
        return vm_stat_answer
    compact = "\n".join(line for line in stdout.strip().splitlines()[:12])
    return f"Диагностика памяти:\n{compact}"


def _process_name_search_answer(observation_ref: ToolObservationRef) -> str:
    try:
        payload = json.loads(observation_ref.content)
    except json.JSONDecodeError:
        return observation_ref.content
    if not isinstance(payload, dict):
        return observation_ref.content
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    exit_code = payload.get("exit_code")
    if not isinstance(stdout, str):
        return observation_ref.content
    matches = [line.strip() for line in stdout.splitlines() if line.strip()]
    if exit_code == 0 and matches:
        compact = "\n".join(matches[:10])
        return f"Процесс запущен:\n{compact}"
    if exit_code in {0, 1, None}:
        return "Процесс не найден."
    detail = stderr if isinstance(stderr, str) and stderr.strip() else stdout
    return f"Не удалось проверить процессы: {detail.strip()}"


def _free_memory_answer(stdout: str) -> str | None:
    lines = [line.split() for line in stdout.splitlines() if line.strip()]
    for index, columns in enumerate(lines):
        if columns and columns[0].casefold().startswith("mem:") and index > 0:
            headers = [header.casefold() for header in lines[index - 1]]
            values = columns[1:] if columns[0].casefold() == "mem:" else columns
            free = _column_value(headers, values, "free")
            available = _column_value(headers, values, "available")
            if free is None and available is None:
                return None
            parts = []
            if free is not None:
                parts.append(f"свободно {free} MiB")
            if available is not None:
                parts.append(f"доступно {available} MiB")
            return "Память: " + ", ".join(parts) + "."
    return None


def _column_value(headers: list[str], values: list[str], name: str) -> str | None:
    try:
        index = headers.index(name)
    except ValueError:
        return None
    if index >= len(values):
        return None
    value = values[index]
    return value if re.fullmatch(r"\d+(?:\.\d+)?", value) else None


def _vm_stat_memory_answer(stdout: str) -> str | None:
    page_size_match = re.search(r"page size of (\d+) bytes", stdout)
    free_pages = _vm_stat_pages(stdout, "Pages free")
    speculative_pages = _vm_stat_pages(stdout, "Pages speculative")
    if page_size_match is None or free_pages is None:
        return None
    page_size = int(page_size_match.group(1))
    free_bytes = free_pages * page_size
    available_bytes = free_bytes + ((speculative_pages or 0) * page_size)
    if speculative_pages:
        return (
            f"Память: свободно {_format_bytes(free_bytes)}, "
            f"доступно примерно {_format_bytes(available_bytes)}."
        )
    return f"Память: свободно {_format_bytes(free_bytes)}."


def _vm_stat_pages(stdout: str, label: str) -> int | None:
    pattern = rf"^{re.escape(label)}:\s+(\d+)\."
    match = re.search(pattern, stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / (1024**2):.0f} MiB"


def _sensor_snapshot_answer(observation_ref: ToolObservationRef) -> str:
    try:
        payload = json.loads(observation_ref.content)
    except json.JSONDecodeError:
        return observation_ref.content
    if not isinstance(payload, dict):
        return observation_ref.content
    if payload.get("available") is False:
        source = payload.get("source") if isinstance(payload.get("source"), str) else "sensor"
        reason = payload.get("reason") if isinstance(payload.get("reason"), str) else "unavailable"
        return f"Не удалось прочитать температуру CPU через {source}: {reason}."
    readings = payload.get("readings")
    if not isinstance(readings, list) or not readings:
        return "Не удалось прочитать температуру CPU: датчики не вернули показания."
    cpu_reading = _select_cpu_reading(readings)
    if cpu_reading is None:
        return "Не удалось прочитать температуру CPU: в ответе нет числовых показаний."
    label = cpu_reading.get("label") if isinstance(cpu_reading.get("label"), str) else "CPU"
    value = cpu_reading.get("value")
    unit = cpu_reading.get("unit") if isinstance(cpu_reading.get("unit"), str) else "C"
    return f"Температура CPU ({label}): {value:g} °{unit}."


def _select_cpu_reading(readings: list[Any]) -> dict[str, Any] | None:
    numeric_readings = [
        reading
        for reading in readings
        if isinstance(reading, dict) and isinstance(reading.get("value"), int | float)
    ]
    for reading in numeric_readings:
        label = str(reading.get("label", "")).casefold()
        if "cpu" in label or "processor" in label or "package" in label:
            return reading
    return numeric_readings[0] if numeric_readings else None


_USER_STREAM_EVENT_TYPES = {
    EventType.APPROVAL_REQUIRED,
    EventType.APPROVAL_GRANTED,
    EventType.APPROVAL_DENIED,
    EventType.APPROVAL_EXPIRED,
    EventType.APPROVAL_CANCELLED,
    EventType.TOOL_SHELL_STARTED,
    EventType.TOOL_SHELL_COMPLETED,
    EventType.TOOL_SHELL_DENIED,
    EventType.TOOL_SHELL_FAILED,
    EventType.TOOL_SHELL_TIMEOUT,
    EventType.TOOL_SHELL_OUTPUT_TRUNCATED,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_STARTED,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_FAILED,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_TIMEOUT,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_OUTPUT_TRUNCATED,
    EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE,
}


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ToolProposalParseError):
        return "malformed_tool_proposal"
    return str(exc) or type(exc).__name__


def _failed_stream_payload(
    request: LoopExecutionRequest,
    failed_event: EventEnvelope | None,
) -> dict[str, Any]:
    if failed_event is None:
        return {
            "request_id": request.request_id,
            "event_id": None,
            "error": {
                "code": "tool_loop_failed",
                "message": "tool loop failed",
                "request_id": request.request_id,
                "details": {},
            },
        }
    error = failed_event.payload.get("error")
    if not isinstance(error, dict):
        error = {
            "code": failed_event.payload.get("error_code") or failed_event.payload.get("error_type"),
            "message": "tool loop failed",
            "request_id": request.request_id,
            "details": {},
        }
    return {
        "request_id": request.request_id,
        "event_id": failed_event.event_id,
        "error": error,
    }
