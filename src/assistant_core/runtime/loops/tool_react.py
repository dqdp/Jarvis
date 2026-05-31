from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
import json
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
from assistant_core.domain.tools import SENSITIVITY_ORDER, ToolObservationStatus, ToolParseStatus
from assistant_core.ports.approvals import ApprovalStorePort
from assistant_core.ports.context_assembler import ContextAssemblerPort
from assistant_core.ports.conversation_store import ConversationStorePort
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.model_router import ModelRouterPort
from assistant_core.ports.tools import ToolGatewayPort
from assistant_core.runtime.direct_tools import (
    DirectToolPlan,
    direct_tool_arguments,
    direct_tool_plan_from_metadata,
)
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
        direct_tool_plan = direct_tool_plan_from_metadata(request.metadata)
        if direct_tool_plan is not None:
            return await self._run_direct_tools(
                request,
                plan=direct_tool_plan,
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
                context_started = await self._append_event(
                    EventType.CONTEXT_ASSEMBLY_STARTED,
                    request,
                    payload={"step_id": step_id, "step_index": step_index},
                    causation_id=step_started.event_id,
                )
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
                            causation_event_id=context_started.event_id,
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
        plan: DirectToolPlan,
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
        used_model_calls = 0
        context_manifest_refs: tuple[str, ...] = ()
        tool_observation_refs: tuple[ToolObservationRef, ...] = ()
        try:
            for tool_name in plan.tool_names:
                observation_ref = await self._proposal_executor.execute(
                    request,
                    ToolProposal(
                        action="tool_call",
                        tool_name=tool_name,
                        arguments=direct_tool_arguments(
                            plan,
                            tool_name,
                            working_directory=request.working_directory,
                        ),
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
            if _has_unparsed_direct_observation(tool_observation_refs):
                context_started = await self._append_event(
                    EventType.CONTEXT_ASSEMBLY_STARTED,
                    request,
                    payload={"step_id": step_id, "step_index": 1},
                    causation_id=step_started.event_id,
                )
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
                            causation_event_id=context_started.event_id,
                            permission_mode=request.permission_mode,
                            tool_observation_refs=tool_observation_refs,
                        ),
                    ),
                    timeout=_remaining_timeout(
                        loop_deadline,
                        request.budget.max_context_assembly_seconds,
                    ),
                )
                context_manifest_refs = (context.manifest.context_manifest_id,)
                if used_model_calls >= request.budget.max_model_calls:
                    raise RuntimeError("max_model_calls_exceeded")
                used_model_calls += 1
                model_response = await asyncio.wait_for(
                    self._model_router.structured(
                        StructuredModelRequest(
                            profile=request.model_profile,
                            messages=context.messages,
                            schema=TOOL_PROPOSAL_SCHEMA,
                            sensitivity=_max_ref_sensitivity(
                                context.manifest.max_sensitivity,
                                tool_observation_refs,
                            ),
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
                proposal = parse_tool_proposal(model_response.value)
                if proposal.action != "final_answer":
                    raise RuntimeError("direct_unparsed_fallback_requires_final_answer")
                return await self._complete(
                    request,
                    proposal,
                    step_started=step_started,
                    used_model_calls=used_model_calls,
                    used_tool_calls=used_tool_calls,
                    context_manifest_refs=context_manifest_refs,
                    tool_observation_refs=tool_observation_refs,
                    sensitivity=_max_ref_sensitivity(
                        context.manifest.max_sensitivity,
                        tool_observation_refs,
                    ),
                )
            return await self._complete(
                request,
                ToolProposal(
                    action="final_answer",
                    final_answer=_direct_tools_answer(plan, tool_observation_refs),
                ),
                step_started=step_started,
                used_model_calls=used_model_calls,
                used_tool_calls=used_tool_calls,
                context_manifest_refs=context_manifest_refs,
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
                used_model_calls=used_model_calls,
                used_tool_calls=used_tool_calls,
                context_manifest_refs=context_manifest_refs,
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


def _direct_tools_answer(
    plan: DirectToolPlan,
    observation_refs: tuple[ToolObservationRef, ...],
) -> str:
    refs_by_name = {ref.tool_name: ref for ref in observation_refs}
    tool_names = plan.tool_names
    if plan.scenario == "christmas_countdown" and tool_names == ("datetime.now",):
        return _christmas_countdown_answer(refs_by_name["datetime.now"])
    if plan.scenario == "battery_charge" and tool_names == ("tool.system.read.hardware",):
        return _battery_charge_answer(refs_by_name["tool.system.read.hardware"])
    if plan.scenario == "disk_free" and tool_names == ("tool.system.read.resources",):
        return _disk_free_answer(refs_by_name["tool.system.read.resources"])
    if plan.scenario == "os_version" and tool_names == ("tool.system.read.hardware",):
        return _os_version_answer(refs_by_name["tool.system.read.hardware"])
    if plan.scenario == "cpu_overview" and tool_names == (
        "tool.system.read.hardware",
        "tool.system.read.resources",
    ):
        return _cpu_overview_answer(
            refs_by_name["tool.system.read.hardware"],
            refs_by_name["tool.system.read.resources"],
        )
    if plan.scenario == "vpn_status" and tool_names == ("tool.system.read.network",):
        return _vpn_status_answer(refs_by_name["tool.system.read.network"])
    return _direct_tool_answer(tool_names[-1], observation_refs[-1])


def _has_unparsed_direct_observation(observation_refs: tuple[ToolObservationRef, ...]) -> bool:
    return any(ref.parse_status == ToolParseStatus.UNPARSED for ref in observation_refs)


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
    payload = _typed_payload(observation_ref, "system.battery_charge")
    if payload is None:
        return "Не удалось разобрать заряд аккумулятора."
    percent = payload.get("percent")
    state = _battery_state_label(payload.get("state"))
    if percent is None:
        return "Не удалось разобрать заряд аккумулятора."
    return _with_partial_warning(f"Аккумулятор: {percent}% ({state}).", observation_ref)


def _battery_state_label(raw_state: object) -> str:
    state = str(raw_state or "").casefold()
    if "discharging" in state:
        return "разряжается"
    if "charging" in state:
        return "заряжается"
    if "charged" in state or "fully" in state:
        return "заряжен"
    return str(raw_state) if raw_state else "состояние неизвестно"


def _disk_free_answer(observation_ref: ToolObservationRef) -> str:
    payload = _typed_payload(observation_ref, "system.disk_free")
    if payload is None:
        return "Не удалось разобрать свободное место на диске."
    filesystems = payload.get("filesystems")
    if not isinstance(filesystems, list) or not filesystems:
        return "Не удалось разобрать свободное место на диске."
    parsed = next(
        (
            filesystem
            for filesystem in filesystems
            if isinstance(filesystem, dict) and filesystem.get("mount") == "/"
        ),
        None,
    )
    if parsed is None:
        parsed = next((filesystem for filesystem in filesystems if isinstance(filesystem, dict)), None)
    answer = (
        f"Диск {parsed.get('mount', '/')}: свободно {parsed.get('available')} "
        f"из {parsed.get('size')} (использовано {parsed.get('used_percent')})."
        if isinstance(parsed, dict)
        else "Не удалось разобрать свободное место на диске."
    )
    return _with_partial_warning(answer, observation_ref)


def _vpn_status_answer(observation_ref: ToolObservationRef) -> str:
    payload = _typed_payload(observation_ref, "system.vpn_status")
    if payload is None:
        return "Не удалось разобрать статус VPN."
    if payload.get("connected") is True:
        service = payload.get("interface_or_service")
        answer = f"VPN включен: {service}." if service else "VPN включен."
        return _with_partial_warning(answer, observation_ref)
    return _with_partial_warning(
        "VPN не включен или активное VPN-подключение не найдено.",
        observation_ref,
    )


def _cpu_overview_answer(
    hardware_ref: ToolObservationRef,
    resources_ref: ToolObservationRef,
) -> str:
    payload = _merge_typed_payloads("system.cpu_overview", hardware_ref, resources_ref)
    if not payload:
        return "Не удалось разобрать сведения о ядрах CPU и текущей загрузке."
    parts = []
    if payload.get("logical_cores") is not None:
        parts.append(f"CPU: {payload['logical_cores']} логических ядер")
    if {"user_percent", "system_percent", "idle_percent"}.issubset(payload):
        parts.append(
            "загрузка: "
            f"{payload['user_percent']}% user, "
            f"{payload['system_percent']}% sys, "
            f"{payload['idle_percent']}% idle"
        )
    if parts:
        return _with_partial_warning("; ".join(parts) + ".", hardware_ref, resources_ref)
    return "Не удалось разобрать сведения о ядрах CPU и текущей загрузке."


def _os_version_answer(observation_ref: ToolObservationRef) -> str:
    payload = _typed_payload(observation_ref, "system.os_version")
    if payload is None:
        return "Не удалось разобрать версию операционной системы."
    name = payload.get("product_name") or payload.get("platform") or "операционная система"
    version = payload.get("version")
    build = payload.get("build")
    if version and build:
        return _with_partial_warning(
            f"Операционная система: {name} {version} (build {build}).",
            observation_ref,
        )
    if version:
        return _with_partial_warning(f"Операционная система: {name} {version}.", observation_ref)
    return "Не удалось разобрать версию операционной системы."


def _resource_snapshot_answer(observation_ref: ToolObservationRef) -> str:
    if observation_ref.structured_schema == "system.memory_overview":
        return _memory_overview_answer(observation_ref)
    if observation_ref.structured_schema == "system.disk_free":
        return _disk_free_answer(observation_ref)
    if observation_ref.structured_schema == "system.cpu_overview":
        payload = _typed_payload(observation_ref, "system.cpu_overview")
        if payload and {"user_percent", "system_percent", "idle_percent"}.issubset(payload):
            return _with_partial_warning(
                "CPU загрузка: "
                f"{payload['user_percent']}% user, "
                f"{payload['system_percent']}% sys, "
                f"{payload['idle_percent']}% idle.",
                observation_ref,
            )
    return "Не удалось разобрать диагностику ресурсов."


def _memory_overview_answer(observation_ref: ToolObservationRef) -> str:
    payload = _typed_payload(observation_ref, "system.memory_overview")
    if payload is None:
        return "Не удалось разобрать сведения о памяти."
    parts = []
    if payload.get("free") is not None:
        parts.append(f"свободно {payload['free']}")
    if payload.get("available") is not None:
        parts.append(f"доступно {payload['available']}")
    if payload.get("used_percent") is not None:
        parts.append(f"использовано {payload['used_percent']}%")
    if not parts:
        return "Не удалось разобрать сведения о памяти."
    return _with_partial_warning("Память: " + ", ".join(parts) + ".", observation_ref)


def _process_name_search_answer(observation_ref: ToolObservationRef) -> str:
    payload = _typed_payload(observation_ref, "system.process_name_search")
    if payload is None:
        return "Не удалось разобрать результат поиска процесса."
    if payload.get("error"):
        return f"Не удалось проверить процессы: {payload['error']}"
    matches = payload.get("matches")
    if isinstance(matches, list) and matches:
        compact = "\n".join(
            f"{match.get('pid')} {match.get('name')}"
            for match in matches[:10]
            if isinstance(match, dict)
        )
        answer = f"Процесс запущен:\n{compact}" if compact else "Процесс запущен."
        return _with_partial_warning(answer, observation_ref)
    return _with_partial_warning("Процесс не найден.", observation_ref)


def _sensor_snapshot_answer(observation_ref: ToolObservationRef) -> str:
    payload = _typed_payload(observation_ref, "system.sensor_snapshot")
    if payload is None:
        return "Не удалось разобрать температуру CPU."
    if payload.get("available") is False:
        source = payload.get("source") if isinstance(payload.get("source"), str) else "sensor"
        reason = payload.get("reason") if isinstance(payload.get("reason"), str) else "unavailable"
        return _with_partial_warning(
            f"Не удалось прочитать температуру CPU через {source}: {reason}.",
            observation_ref,
        )
    readings = payload.get("readings")
    if not isinstance(readings, list) or not readings:
        return "Не удалось прочитать температуру CPU: датчики не вернули показания."
    cpu_reading = _select_cpu_reading(readings)
    if cpu_reading is None:
        return "Не удалось прочитать температуру CPU: в ответе нет числовых показаний."
    label = cpu_reading.get("label") if isinstance(cpu_reading.get("label"), str) else "CPU"
    value = cpu_reading.get("value")
    unit = cpu_reading.get("unit") if isinstance(cpu_reading.get("unit"), str) else "C"
    return _with_partial_warning(f"Температура CPU ({label}): {value:g} °{unit}.", observation_ref)


def _typed_payload(observation_ref: ToolObservationRef, schema: str) -> dict[str, Any] | None:
    if observation_ref.structured_schema != schema:
        return None
    if observation_ref.parse_status not in {
        ToolParseStatus.PARSED,
        ToolParseStatus.PARTIAL,
    }:
        return None
    if not isinstance(observation_ref.structured_content, dict):
        return None
    return observation_ref.structured_content


def _merge_typed_payloads(schema: str, *refs: ToolObservationRef) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for ref in refs:
        payload = _typed_payload(ref, schema)
        if payload:
            merged.update(payload)
    return merged


def _with_partial_warning(answer: str, *refs: ToolObservationRef) -> str:
    if any(ref.parse_status == ToolParseStatus.PARTIAL for ref in refs):
        return f"Данные частично разобраны; некоторые поля могут быть неполными. {answer}"
    return answer


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
    EventType.REQUEST_PROCESSING_STARTED,
    EventType.CONTEXT_ASSEMBLY_STARTED,
    EventType.MEMORY_RETRIEVED,
    EventType.MEMORY_RETRIEVAL_FAILED,
    EventType.CONTENT_RETRIEVED,
    EventType.CONTEXT_ASSEMBLED,
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
