from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from assistant_core.domain.context import ContextAssemblyRequest
from assistant_core.domain.conversations import (
    UpdateAssistantRequestStatusCommand,
)
from assistant_core.domain.events import EventEnvelope, EventType
from assistant_core.domain.loops import (
    AgentLoopState,
    AgentLoopStep,
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
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.ports.approvals import ApprovalStorePort
from assistant_core.ports.context_assembler import ContextAssemblerPort
from assistant_core.ports.conversation_store import ConversationStorePort
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.model_router import ModelRouterPort
from assistant_core.ports.tools import ToolGatewayPort
from assistant_core.runtime.loops.event_recorder import LoopEventRecorder
from assistant_core.runtime.loops.failure_policy import LoopFailureDecision, LoopFailurePolicy
from assistant_core.runtime.loops.final_answer import FinalAnswerStep, FinalAnswerStepError
from assistant_core.runtime.loops.tool_approval import ApprovalWaiter
from assistant_core.runtime.loops.tool_proposal_executor import ToolProposalExecutor
from assistant_core.runtime.request_streaming import public_stream_data


TOOL_PROPOSAL_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"const": "tool_call"},
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["action", "tool_name", "arguments"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"const": "final_answer"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    ],
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
        failure_policy: Any | None = None,
    ) -> None:
        if tool_gateway is None:
            raise ValueError("tool_gateway is required")
        self._conversation_store = conversation_store
        self._context_assembler = context_assembler
        self._model_router = model_router
        self._event_log = event_log
        self._event_recorder = LoopEventRecorder(
            event_log=event_log,
            source_component="tool_react_loop",
        )
        self._tool_gateway = tool_gateway
        self._approval_store = approval_store
        self._failure_policy = failure_policy or LoopFailurePolicy()
        self._proposal_executor = ToolProposalExecutor(
            tool_gateway=tool_gateway,
            conversation_store=conversation_store,
            approval_waiter=ApprovalWaiter(approval_store) if approval_store is not None else None,
            state_recorder=self._record_tool_state,
        )
        self._final_answer_step = FinalAnswerStep(
            conversation_store=conversation_store,
            context_assembler=context_assembler,
            model_router=model_router,
            event_recorder=self._event_recorder,
        )

    def validate_budget(self, budget: LoopBudget) -> None:
        if budget.max_steps <= 0:
            raise ValueError("tool_react_loop requires positive max_steps")
        if budget.max_model_calls <= 0:
            raise ValueError("tool_react_loop requires positive max_model_calls")

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
            state=AgentLoopState.REQUEST_STARTED,
            step=AgentLoopStep.STARTED,
        )
        loop_started = await self._append_event(
            EventType.AGENT_LOOP_STARTED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "budget": _budget_payload(request.budget),
            },
            causation_id=request_started.event_id,
            state=AgentLoopState.REQUEST_STARTED,
            step=AgentLoopStep.STARTED,
        )

        used_model_calls = 0
        used_tool_calls = 0
        consecutive_failures = 0
        request_plan = _tool_request_plan(request.metadata)
        context_manifest_refs: list[str] = []
        tool_observation_refs: list[ToolObservationRef] = []
        loop_deadline = asyncio.get_running_loop().time() + float(
            request.budget.max_wall_time_seconds,
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
                step=AgentLoopStep.STARTED,
            )
            try:
                if _should_use_final_chat_without_proposal(
                    request_plan,
                    used_tool_calls=used_tool_calls,
                    budget=request.budget,
                ):
                    return await self._final_answer_step.run(
                        request,
                        step_started=step_started,
                        used_model_calls=used_model_calls,
                        used_tool_calls=used_tool_calls,
                        context_manifest_refs=context_manifest_refs,
                        tool_observation_refs=tool_observation_refs,
                        loop_deadline=loop_deadline,
                    )

                context_started = await self._append_event(
                    EventType.CONTEXT_ASSEMBLY_STARTED,
                    request,
                    payload={
                        "step_id": step_id,
                        "step_index": step_index,
                        "purpose": "tool_proposal",
                    },
                    causation_id=step_started.event_id,
                    state=AgentLoopState.CONTEXT_ASSEMBLING,
                    step=AgentLoopStep.PROPOSAL,
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
                            purpose="tool_proposal",
                            permission_mode=request.permission_mode,
                            tool_observation_refs=tuple(tool_observation_refs),
                            output_contract=_tool_proposal_output_contract(
                                request_plan,
                                used_tool_calls=used_tool_calls,
                            ),
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
                model_started = await self._append_event(
                    EventType.MODEL_REQUEST_CREATED,
                    request,
                    payload={"context_manifest_id": context.manifest.context_manifest_id},
                    causation_id=context_started.event_id,
                    sensitivity=context.manifest.max_sensitivity,
                    state=AgentLoopState.PROPOSING,
                    step=AgentLoopStep.PROPOSAL,
                )
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
                await self._append_event(
                    EventType.MODEL_RESPONSE_RECEIVED,
                    request,
                    payload={"context_manifest_id": context.manifest.context_manifest_id},
                    causation_id=model_started.event_id,
                    sensitivity=context.manifest.max_sensitivity,
                    state=AgentLoopState.PROPOSING,
                    step=AgentLoopStep.PROPOSAL,
                )
                try:
                    proposal = parse_tool_proposal(model_response.value)
                except ToolProposalParseError as exc:
                    if not _should_fallback_to_final_chat_after_malformed_proposal(
                        model_response.value,
                        request_plan,
                        used_tool_calls=used_tool_calls,
                    ):
                        raise
                    try:
                        return await self._final_answer_step.run(
                            request,
                            step_started=step_started,
                            used_model_calls=used_model_calls,
                            used_tool_calls=used_tool_calls,
                            context_manifest_refs=context_manifest_refs,
                            tool_observation_refs=tool_observation_refs,
                            loop_deadline=loop_deadline,
                        )
                    except RuntimeError as final_exc:
                        if str(final_exc) == "max_model_calls_exceeded":
                            raise RuntimeError("max_model_calls_exceeded") from exc
                        raise
                if proposal.action == "final_answer":
                    _ensure_final_answer_allowed(
                        request_plan,
                        used_tool_calls=used_tool_calls,
                    )
                    return await self._final_answer_step.run(
                        request,
                        step_started=step_started,
                        used_model_calls=used_model_calls,
                        used_tool_calls=used_tool_calls,
                        context_manifest_refs=context_manifest_refs,
                        tool_observation_refs=tool_observation_refs,
                        loop_deadline=loop_deadline,
                    )

                await self._record_tool_state(
                    request=request,
                    proposal=proposal,
                    step_id=step_id,
                    step_index=step_index,
                    causation_event_id=step_started.event_id,
                    state=AgentLoopState.TOOL_VALIDATING,
                )
                _ensure_tool_call_allowed_by_plan(request_plan, proposal)
                observation_ref = await self._proposal_executor.execute(
                    request,
                    proposal,
                    step_id=step_id,
                    causation_event_id=step_started.event_id,
                    used_tool_calls=used_tool_calls,
                    loop_deadline=loop_deadline,
                    step_index=step_index,
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
                    state=AgentLoopState.OBSERVING,
                    step=AgentLoopStep.OBSERVATION,
                )
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                failure_exc = exc
                if isinstance(exc, FinalAnswerStepError):
                    used_model_calls = exc.used_model_calls
                    failure_exc = exc.cause
                failure_decision = self._failure_policy.decide(failure_exc)
                await self._append_event(
                    EventType.AGENT_STEP_FAILED,
                    request,
                    payload={
                        "strategy_name": request.strategy_name.value,
                        "step_id": step_id,
                        "step_index": step_index,
                        "error_code": failure_decision.error_code,
                        "error_type": type(failure_exc).__name__,
                    },
                    causation_id=step_started.event_id,
                    state=AgentLoopState.FAILED,
                    step=AgentLoopStep.FAILED,
                )
                await self._fail(
                    request,
                    failure_exc,
                    decision=failure_decision,
                    causation_id=step_started.event_id,
                    used_model_calls=used_model_calls,
                    used_tool_calls=used_tool_calls,
                    context_manifest_refs=tuple(context_manifest_refs),
                    tool_observation_refs=tuple(tool_observation_refs),
                )
                raise failure_exc
            if consecutive_failures > request.budget.max_consecutive_failures:
                break

        exc = RuntimeError("max_steps_exceeded")
        failure_decision = self._failure_policy.decide(exc)
        await self._fail(
            request,
            exc,
            decision=failure_decision,
            causation_id=loop_started.event_id,
            used_model_calls=used_model_calls,
            used_tool_calls=used_tool_calls,
            context_manifest_refs=tuple(context_manifest_refs),
            tool_observation_refs=tuple(tool_observation_refs),
        )
        raise exc

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

    async def _record_tool_state(
        self,
        *,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        step_id: str,
        step_index: int | None,
        causation_event_id: str,
        state: AgentLoopState | str,
        observation_ref: ToolObservationRef | None = None,
    ) -> None:
        if isinstance(state, str):
            state = AgentLoopState(state)
        payload: dict[str, Any] = {
            "strategy_name": request.strategy_name.value,
            "step_id": step_id,
            "action": "tool_call",
            "tool_name": proposal.tool_name,
        }
        if step_index is not None:
            payload["step_index"] = step_index
        if observation_ref is not None:
            payload["tool_call_id"] = observation_ref.tool_call_id
            payload["observation_status"] = observation_ref.status.value
        await self._append_event(
            EventType.AGENT_STEP_STARTED,
            request,
            payload=payload,
            causation_id=causation_event_id,
            state=state,
            step=AgentLoopStep.TOOL,
        )

    async def _fail(
        self,
        request: LoopExecutionRequest,
        exc: Exception,
        *,
        decision: LoopFailureDecision,
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
                error_code=decision.error_code,
                error_message=decision.error_message,
            ),
        )
        loop_failed = await self._append_event(
            EventType.AGENT_LOOP_FAILED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "status": LoopStatus.FAILED.value,
                "error_code": decision.error_code,
                "error_type": type(exc).__name__,
                "used_model_calls": used_model_calls,
                "used_tool_calls": used_tool_calls,
                "context_manifest_refs": list(context_manifest_refs),
                "tool_observation_refs": [
                    ref.tool_call_id for ref in tool_observation_refs
                ],
            },
            causation_id=causation_id,
            state=AgentLoopState.FAILED,
            step=AgentLoopStep.FAILED,
        )
        await self._append_event(
            EventType.REQUEST_PROCESSING_FAILED,
            request,
            payload={
                "error_type": type(exc).__name__,
                "error_code": decision.error_code,
                "error": {
                    "code": decision.error_code,
                    "message": decision.error_message,
                    "request_id": request.request_id,
                    "details": decision.details,
                },
            },
            causation_id=loop_failed.event_id,
            state=AgentLoopState.FAILED,
            step=AgentLoopStep.FAILED,
        )

    async def _append_event(
        self,
        event_type: EventType,
        request: LoopExecutionRequest,
        *,
        payload: dict[str, Any],
        causation_id: str | None = None,
        sensitivity: Sensitivity = Sensitivity.PROJECT,
        state: AgentLoopState | None = None,
        step: AgentLoopStep | None = None,
    ) -> EventEnvelope:
        return await self._event_recorder.append(
            event_type,
            request,
            payload=payload,
            causation_id=causation_id,
            sensitivity=sensitivity,
            state=state,
            step=step,
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


_KNOWN_TOOL_POLICIES = {"disabled", "available", "required"}


def _tool_request_plan(metadata: dict[str, Any]) -> tuple[str | None, frozenset[str] | None]:
    raw_policy = metadata.get("agent_tool_policy")
    policy = raw_policy if isinstance(raw_policy, str) else None
    raw_allowed = metadata.get("agent_allowed_tool_names")
    if isinstance(raw_allowed, list):
        allowed = frozenset(item for item in raw_allowed if isinstance(item, str) and item)
    elif isinstance(raw_allowed, tuple):
        allowed = frozenset(item for item in raw_allowed if isinstance(item, str) and item)
    elif policy in {"available", "required"}:
        allowed = frozenset()
    else:
        allowed = None
    return policy, allowed


def _tool_proposal_output_contract(
    request_plan: tuple[str | None, frozenset[str] | None],
    *,
    used_tool_calls: int,
) -> str:
    policy, allowed = request_plan
    lines = [
        "Return only a JSON object for the agent loop.",
        'Use {"action":"final_answer"} when ready to answer without another tool.',
        (
            'Use {"action":"tool_call","tool_name":"...","arguments":{...}} '
            "only when an allowed tool is needed."
        ),
        "Do not wrap the JSON in markdown. Do not add extra keys.",
    ]
    if policy == "disabled" or policy not in _KNOWN_TOOL_POLICIES:
        lines.append("Tool calls are disabled for this request; use final_answer only.")
    elif allowed:
        lines.append("Allowed tools: " + ", ".join(sorted(allowed)) + ".")
    else:
        lines.append("No tools are allowed for this request; use final_answer only.")
    if policy == "required" and used_tool_calls <= 0:
        lines.append("A tool call is required before the final answer.")
    return " ".join(lines)


def _should_use_final_chat_without_proposal(
    request_plan: tuple[str | None, frozenset[str] | None],
    *,
    used_tool_calls: int,
    budget: LoopBudget,
) -> bool:
    policy, _allowed = request_plan
    if policy == "disabled":
        return True
    if used_tool_calls < budget.max_tool_calls:
        return False
    return not (policy == "required" and used_tool_calls <= 0)


def _should_fallback_to_final_chat_after_malformed_proposal(
    value: Any,
    request_plan: tuple[str | None, frozenset[str] | None],
    *,
    used_tool_calls: int,
) -> bool:
    policy, _allowed = request_plan
    if policy not in {"available", "required"}:
        return False
    if policy == "required" and used_tool_calls <= 0:
        return False
    if isinstance(value, dict) and value.get("action") == "tool_call":
        return False
    return True


def _ensure_final_answer_allowed(
    request_plan: tuple[str | None, frozenset[str] | None],
    *,
    used_tool_calls: int,
) -> None:
    policy, _allowed = request_plan
    if policy == "required" and used_tool_calls <= 0:
        raise RuntimeError("required_tool_call_missing")


def _ensure_tool_call_allowed_by_plan(
    request_plan: tuple[str | None, frozenset[str] | None],
    proposal: ToolProposal,
) -> None:
    policy, allowed = request_plan
    if policy is None:
        raise RuntimeError("request_plan_missing_tool_policy")
    if policy not in _KNOWN_TOOL_POLICIES:
        raise RuntimeError("request_plan_invalid_tool_policy")
    if policy == "disabled":
        raise RuntimeError("tool_policy_disabled")
    if policy in {"available", "required"}:
        if proposal.tool_name is None:
            raise ToolProposalParseError("tool_call requires tool_name")
        if allowed is not None and proposal.tool_name not in allowed:
            raise RuntimeError("tool_not_allowed_by_request_plan")


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
