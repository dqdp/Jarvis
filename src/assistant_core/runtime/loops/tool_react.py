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
    ToolRequestPlan,
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
from assistant_core.runtime.loops.tool_loop_finalization import (
    should_fallback_to_final_chat_after_malformed_proposal as _should_fallback_to_final_chat_after_malformed_proposal,
    should_fallback_to_final_chat_after_proposal_timeout as _should_fallback_to_final_chat_after_proposal_timeout,
    should_fallback_to_final_chat_after_structured_error as _should_fallback_to_final_chat_after_structured_error,
    should_use_final_chat_without_proposal as _should_use_final_chat_without_proposal,
    tool_call_signature as _tool_call_signature,
)
from assistant_core.runtime.loops.tool_loop_streaming import (
    failed_stream_payload as _failed_stream_payload,
)
from assistant_core.runtime.loops.tool_loop_contracts import (
    ensure_final_answer_allowed as _ensure_final_answer_allowed,
    ensure_tool_call_allowed_by_plan as _ensure_tool_call_allowed_by_plan,
    live_state_unavailable_response as _live_state_unavailable_response,
    malformed_proposal_after_tool_output_contract as _malformed_proposal_after_tool_output_contract,
    repeated_tool_call_output_contract as _repeated_tool_call_output_contract,
    should_complete_live_state_unavailable_deterministically as _should_complete_live_state_unavailable_deterministically,
    tool_observation_recovery_output_contract as _tool_observation_recovery_output_contract,
    tool_proposal_output_contract as _tool_proposal_output_contract,
    tool_proposal_timeout_after_tool_output_contract as _tool_proposal_timeout_after_tool_output_contract,
    unevidenced_tool_proposal_fallback_output_contract as _unevidenced_tool_proposal_fallback_output_contract,
)
from assistant_core.runtime.loops.tool_loop_deterministic import (
    deterministic_datetime_now_response as _deterministic_datetime_now_response,
    recover_malformed_safe_builtin_tool_proposal as _recover_malformed_safe_builtin_tool_proposal,
)
from assistant_core.runtime.loops.tool_loop_evidence import (
    LiveStateEvidencePlan,
    final_answer_deferred_missing_evidence_plan as _final_answer_deferred_missing_evidence_plan,
    final_answer_missing_evidence_plan as _final_answer_missing_evidence_plan,
    failed_observation_exhausts_missing_evidence as _failed_observation_exhausts_missing_evidence,
    request_requires_initial_tool_evidence as _request_requires_initial_tool_evidence,
    TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS as _DEFAULT_TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS,
    tool_proposal_model_call_timeout as _tool_proposal_model_call_timeout,
)
from assistant_core.runtime.loops.observation_recovery import (
    ToolObservationRecoveryAction,
    ToolObservationRecoveryDecision,
    ToolObservationRecoveryError,
    ToolObservationRecoveryPolicy,
)
from assistant_core.runtime.loops.tool_approval import ApprovalWaiter
from assistant_core.runtime.loops.tool_proposal_executor import ToolProposalExecutor
from assistant_core.runtime.request_streaming import public_stream_data


TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS = _DEFAULT_TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS


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
        observation_recovery_policy: Any | None = None,
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
        self._observation_recovery_policy = (
            observation_recovery_policy or ToolObservationRecoveryPolicy()
        )
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
        completed_tool_observations = 0
        consecutive_failures = 0
        request_plan = ToolRequestPlan.from_metadata(request.metadata)
        context_manifest_refs: list[str] = []
        tool_observation_refs: list[ToolObservationRef] = []
        completed_tool_call_signatures: set[tuple[str, str]] = set()
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
            active_step_started = step_started
            active_step_id = step_id
            try:
                if _should_use_final_chat_without_proposal(
                    request_plan,
                    used_tool_calls=used_tool_calls,
                    completed_observations=completed_tool_observations,
                    budget=request.budget,
                ):
                    if await self._defer_final_answer_if_missing_evidence(
                        request,
                        request_plan,
                        step_started=step_started,
                        step_id=step_id,
                        step_index=step_index,
                        sensitivity=request.current_message_sensitivity,
                        tool_observation_refs=tool_observation_refs,
                        used_tool_calls=used_tool_calls,
                    ):
                        continue
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
                                completed_observations=completed_tool_observations,
                                missing_evidence_plan=_final_answer_missing_evidence_plan(
                                    request,
                                    request_plan,
                                    tool_observation_refs=tuple(tool_observation_refs),
                                ),
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
                proposal: ToolProposal | None = None
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
                            _tool_proposal_model_call_timeout(
                                request.budget,
                                completed_observations=completed_tool_observations,
                                request=request,
                                request_plan=request_plan,
                                initial_model_call_cap_seconds=TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS,
                            ),
                        ),
                    )
                except TimeoutError as exc:
                    if _wall_time_expired(loop_deadline):
                        raise RuntimeError("max_wall_time_exceeded") from exc
                    if _should_fallback_to_final_chat_after_proposal_timeout(
                        request,
                        request_plan,
                        completed_observations=completed_tool_observations,
                    ):
                        if await self._defer_final_answer_if_missing_evidence(
                            request,
                            request_plan,
                            step_started=step_started,
                            step_id=step_id,
                            step_index=step_index,
                            sensitivity=context.manifest.max_sensitivity,
                            tool_observation_refs=tool_observation_refs,
                            used_tool_calls=used_tool_calls,
                        ):
                            continue
                        output_contract = (
                            _tool_proposal_timeout_after_tool_output_contract()
                            if completed_tool_observations > 0
                            else _unevidenced_tool_proposal_fallback_output_contract(request_plan)
                        )
                        return await self._final_answer_step.run(
                            request,
                            step_started=step_started,
                            used_model_calls=used_model_calls,
                            used_tool_calls=used_tool_calls,
                            context_manifest_refs=context_manifest_refs,
                            tool_observation_refs=tool_observation_refs,
                            loop_deadline=loop_deadline,
                            output_contract=output_contract,
                        )
                    if proposal is None:
                        raise
                except Exception as exc:
                    if _should_fallback_to_final_chat_after_structured_error(
                        exc,
                        request,
                        request_plan,
                        completed_observations=completed_tool_observations,
                    ):
                        if await self._defer_final_answer_if_missing_evidence(
                            request,
                            request_plan,
                            step_started=step_started,
                            step_id=step_id,
                            step_index=step_index,
                            sensitivity=context.manifest.max_sensitivity,
                            tool_observation_refs=tool_observation_refs,
                            used_tool_calls=used_tool_calls,
                        ):
                            continue
                        output_contract = (
                            _malformed_proposal_after_tool_output_contract()
                            if completed_tool_observations > 0
                            else _unevidenced_tool_proposal_fallback_output_contract(request_plan)
                        )
                        return await self._final_answer_step.run(
                            request,
                            step_started=step_started,
                            used_model_calls=used_model_calls,
                            used_tool_calls=used_tool_calls,
                            context_manifest_refs=context_manifest_refs,
                            tool_observation_refs=tool_observation_refs,
                            loop_deadline=loop_deadline,
                            output_contract=output_contract,
                        )
                    else:
                        raise
                if proposal is None:
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
                        proposal = _recover_malformed_safe_builtin_tool_proposal(
                            model_response.value,
                            request,
                            request_plan,
                            completed_observations=completed_tool_observations,
                        )
                        if proposal is None:
                            if not _should_fallback_to_final_chat_after_malformed_proposal(
                                model_response.value,
                                request,
                                request_plan,
                                completed_observations=completed_tool_observations,
                            ):
                                raise
                            if await self._defer_final_answer_if_missing_evidence(
                                request,
                                request_plan,
                                step_started=step_started,
                                step_id=step_id,
                                step_index=step_index,
                                sensitivity=context.manifest.max_sensitivity,
                                tool_observation_refs=tool_observation_refs,
                                used_tool_calls=used_tool_calls,
                            ):
                                continue
                            output_contract = (
                                _malformed_proposal_after_tool_output_contract()
                                if completed_tool_observations > 0
                                else _unevidenced_tool_proposal_fallback_output_contract(request_plan)
                            )
                            try:
                                return await self._final_answer_step.run(
                                    request,
                                    step_started=step_started,
                                    used_model_calls=used_model_calls,
                                    used_tool_calls=used_tool_calls,
                                    context_manifest_refs=context_manifest_refs,
                                    tool_observation_refs=tool_observation_refs,
                                    loop_deadline=loop_deadline,
                                    output_contract=output_contract,
                                )
                            except RuntimeError as final_exc:
                                if str(final_exc) == "max_model_calls_exceeded":
                                    raise RuntimeError("max_model_calls_exceeded") from exc
                                raise
                if proposal is None:
                    raise RuntimeError("malformed_tool_proposal")
                if proposal.action == "final_answer":
                    if await self._defer_final_answer_if_missing_evidence(
                        request,
                        request_plan,
                        step_started=step_started,
                        step_id=step_id,
                        step_index=step_index,
                        sensitivity=context.manifest.max_sensitivity,
                        tool_observation_refs=tool_observation_refs,
                        used_tool_calls=used_tool_calls,
                    ):
                        continue
                    _ensure_final_answer_allowed(
                        request_plan,
                        completed_observations=completed_tool_observations,
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

                if _tool_call_signature(proposal) in completed_tool_call_signatures:
                    if await self._defer_final_answer_if_missing_evidence(
                        request,
                        request_plan,
                        step_started=step_started,
                        step_id=step_id,
                        step_index=step_index,
                        sensitivity=context.manifest.max_sensitivity,
                        tool_observation_refs=tool_observation_refs,
                        used_tool_calls=used_tool_calls,
                    ):
                        continue
                    return await self._final_answer_step.run(
                        request,
                        step_started=step_started,
                        used_model_calls=used_model_calls,
                        used_tool_calls=used_tool_calls,
                        context_manifest_refs=context_manifest_refs,
                        tool_observation_refs=tool_observation_refs,
                        loop_deadline=loop_deadline,
                        output_contract=_repeated_tool_call_output_contract(proposal),
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
                    tool_requires_live_state = _tool_requires_live_state(
                        request.metadata,
                        observation_ref.tool_name,
                    )
                    recovery_decision = self._observation_recovery_policy.decide(
                        request_plan=request_plan,
                        observation_status=observation_ref.status,
                        observation_error_code=observation_ref.error_code,
                        tool_call_id=observation_ref.tool_call_id,
                        completed_observations=completed_tool_observations,
                        consecutive_failures=consecutive_failures + 1,
                        max_consecutive_failures=request.budget.max_consecutive_failures,
                        tool_requires_live_state=tool_requires_live_state,
                    )
                    if recovery_decision.action == ToolObservationRecoveryAction.FINALIZE:
                        recovery_completed = await self._append_observation_step_recovered(
                            request,
                            proposal=proposal,
                            step_started=step_started,
                            observation_ref=observation_ref,
                            recovery_decision=recovery_decision,
                        )
                        final_step_started = await self._append_recovery_final_step_started(
                            request,
                            step_index=step_index,
                            source_step_id=step_id,
                            causation_id=recovery_completed.event_id,
                        )
                        active_step_started = final_step_started
                        active_step_id = final_step_started.payload["step_id"]
                        if _should_complete_live_state_unavailable_deterministically(
                            observation_ref,
                            tool_requires_live_state=tool_requires_live_state,
                        ) and _failed_observation_exhausts_missing_evidence(
                            request,
                            request_plan,
                            observation_ref,
                            tuple(tool_observation_refs),
                        ):
                            return await self._final_answer_step.complete_deterministic(
                                request,
                                step_started=final_step_started,
                                response_text=_live_state_unavailable_response(observation_ref),
                                used_model_calls=used_model_calls,
                                used_tool_calls=used_tool_calls,
                                context_manifest_refs=context_manifest_refs,
                                tool_observation_refs=tool_observation_refs,
                                source_step_id=step_id,
                            )
                        if await self._defer_final_answer_if_missing_evidence(
                            request,
                            request_plan,
                            step_started=final_step_started,
                            step_id=final_step_started.payload["step_id"],
                            step_index=step_index,
                            sensitivity=observation_ref.sensitivity,
                            tool_observation_refs=tool_observation_refs,
                            used_tool_calls=used_tool_calls,
                            event_state=AgentLoopState.FINALIZING,
                            event_step=AgentLoopStep.FINAL,
                        ):
                            continue
                        return await self._final_answer_step.run(
                            request,
                            step_started=final_step_started,
                            used_model_calls=used_model_calls,
                            used_tool_calls=used_tool_calls,
                            context_manifest_refs=context_manifest_refs,
                            tool_observation_refs=tool_observation_refs,
                            loop_deadline=loop_deadline,
                            source_step_id=step_id,
                            output_contract=_tool_observation_recovery_output_contract(
                                observation_ref,
                                tool_requires_live_state=tool_requires_live_state,
                            ),
                        )
                    raise ToolObservationRecoveryError(recovery_decision)
                await self._append_observation_step_completed(
                    request,
                    proposal=proposal,
                    step_started=step_started,
                    observation_ref=observation_ref,
                )
                completed_tool_observations += 1
                completed_tool_call_signatures.add(_tool_call_signature(proposal))
                consecutive_failures = 0
                deterministic_response = _deterministic_datetime_now_response(
                    request,
                    observation_ref,
                )
                if deterministic_response is not None:
                    final_step_started = await self._append_final_step_started_after_observation(
                        request,
                        step_index=step_index,
                        source_step_id=step_id,
                        causation_id=step_started.event_id,
                    )
                    active_step_started = final_step_started
                    active_step_id = final_step_started.payload["step_id"]
                    return await self._final_answer_step.complete_deterministic(
                        request,
                        step_started=final_step_started,
                        response_text=deterministic_response,
                        used_model_calls=used_model_calls,
                        used_tool_calls=used_tool_calls,
                        context_manifest_refs=context_manifest_refs,
                        tool_observation_refs=tool_observation_refs,
                        source_step_id=step_id,
                        source="deterministic_tool_answer",
                        degraded=False,
                    )
                if _should_use_final_chat_without_proposal(
                    request_plan,
                    used_tool_calls=used_tool_calls,
                    completed_observations=completed_tool_observations,
                    budget=request.budget,
                ):
                    if await self._defer_final_answer_if_missing_evidence(
                        request,
                        request_plan,
                        step_started=step_started,
                        step_id=step_id,
                        step_index=step_index,
                        sensitivity=observation_ref.sensitivity,
                        tool_observation_refs=tool_observation_refs,
                        used_tool_calls=used_tool_calls,
                    ):
                        continue
                    final_step_started = await self._append_final_step_started_after_observation(
                        request,
                        step_index=step_index,
                        source_step_id=step_id,
                        causation_id=step_started.event_id,
                    )
                    active_step_started = final_step_started
                    active_step_id = final_step_started.payload["step_id"]
                    return await self._final_answer_step.run(
                        request,
                        step_started=final_step_started,
                        used_model_calls=used_model_calls,
                        used_tool_calls=used_tool_calls,
                        context_manifest_refs=context_manifest_refs,
                        tool_observation_refs=tool_observation_refs,
                        loop_deadline=loop_deadline,
                        source_step_id=step_id,
                    )
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
                        "step_id": active_step_id,
                        "step_index": step_index,
                        "error_code": failure_decision.error_code,
                        "error_type": type(failure_exc).__name__,
                    },
                    causation_id=active_step_started.event_id,
                    state=AgentLoopState.FAILED,
                    step=AgentLoopStep.FAILED,
                )
                await self._fail(
                    request,
                    failure_exc,
                    decision=failure_decision,
                    causation_id=active_step_started.event_id,
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
            async for event in self._public_stream_events(request, seen_stream_events):
                yield event
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

    async def _append_observation_step_completed(
        self,
        request: LoopExecutionRequest,
        *,
        proposal: ToolProposal,
        step_started: EventEnvelope,
        observation_ref: ToolObservationRef,
    ) -> None:
        await self._append_event(
            EventType.AGENT_STEP_COMPLETED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": step_started.payload["step_id"],
                "step_index": step_started.payload["step_index"],
                "action": "tool_call",
                "tool_name": proposal.tool_name,
                "tool_call_id": observation_ref.tool_call_id,
                "observation_status": observation_ref.status.value,
            },
            causation_id=step_started.event_id,
            state=AgentLoopState.OBSERVING,
            step=AgentLoopStep.OBSERVATION,
        )

    async def _append_observation_step_recovered(
        self,
        request: LoopExecutionRequest,
        *,
        proposal: ToolProposal,
        step_started: EventEnvelope,
        observation_ref: ToolObservationRef,
        recovery_decision: ToolObservationRecoveryDecision,
    ) -> EventEnvelope:
        payload = {
            "strategy_name": request.strategy_name.value,
            "step_id": step_started.payload["step_id"],
            "step_index": step_started.payload["step_index"],
            "action": "tool_observation_recovered",
            "tool_name": proposal.tool_name,
            "tool_call_id": observation_ref.tool_call_id,
            "observation_status": observation_ref.status.value,
            "recovery_action": recovery_decision.action.value,
        }
        if recovery_decision.error_code is not None:
            payload["error_code"] = recovery_decision.error_code
        return await self._append_event(
            EventType.AGENT_STEP_COMPLETED,
            request,
            payload=payload,
            causation_id=step_started.event_id,
            state=AgentLoopState.OBSERVING,
            step=AgentLoopStep.OBSERVATION,
        )

    async def _append_recovery_final_step_started(
        self,
        request: LoopExecutionRequest,
        *,
        step_index: int,
        source_step_id: str,
        causation_id: str,
    ) -> EventEnvelope:
        return await self._append_event(
            EventType.AGENT_STEP_STARTED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": str(uuid4()),
                "step_index": step_index,
                "action": "final_answer",
                "source_step_id": source_step_id,
                "source": "tool_observation_recovery",
            },
            causation_id=causation_id,
            state=AgentLoopState.FINALIZING,
            step=AgentLoopStep.FINAL,
        )

    async def _append_final_step_started_after_observation(
        self,
        request: LoopExecutionRequest,
        *,
        step_index: int,
        source_step_id: str,
        causation_id: str,
    ) -> EventEnvelope:
        return await self._append_event(
            EventType.AGENT_STEP_STARTED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": str(uuid4()),
                "step_index": step_index,
                "action": "final_answer",
                "source_step_id": source_step_id,
                "source": "tool_observation_completed",
            },
            causation_id=causation_id,
            state=AgentLoopState.FINALIZING,
            step=AgentLoopStep.FINAL,
        )

    async def _defer_final_answer_if_missing_evidence(
        self,
        request: LoopExecutionRequest,
        request_plan: ToolRequestPlan,
        *,
        step_started: EventEnvelope,
        step_id: str,
        step_index: int,
        sensitivity: Sensitivity,
        tool_observation_refs: list[ToolObservationRef],
        used_tool_calls: int,
        event_state: AgentLoopState = AgentLoopState.PROPOSING,
        event_step: AgentLoopStep = AgentLoopStep.PROPOSAL,
    ) -> bool:
        evidence_plan = _final_answer_deferred_missing_evidence_plan(
            request,
            request_plan,
            tool_observation_refs=tool_observation_refs,
            used_tool_calls=used_tool_calls,
        )
        if evidence_plan is None:
            return False
        await self._append_deferred_final_answer_step(
            request,
            step_started=step_started,
            step_id=step_id,
            step_index=step_index,
            sensitivity=sensitivity,
            evidence_plan=evidence_plan,
            event_state=event_state,
            event_step=event_step,
        )
        return True

    async def _append_deferred_final_answer_step(
        self,
        request: LoopExecutionRequest,
        *,
        step_started: EventEnvelope,
        step_id: str,
        step_index: int,
        sensitivity: Sensitivity,
        evidence_plan: LiveStateEvidencePlan,
        event_state: AgentLoopState,
        event_step: AgentLoopStep,
    ) -> EventEnvelope:
        missing_families = sorted(family.value for family in evidence_plan.missing_families)
        family = (
            missing_families[0]
            if missing_families
            else evidence_plan.family.value
            if evidence_plan.family is not None
            else None
        )
        return await self._append_event(
            EventType.AGENT_STEP_COMPLETED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": step_id,
                "step_index": step_index,
                "action": "final_answer_deferred_missing_evidence",
                "missing_evidence_family": family,
                "missing_evidence_families": missing_families,
                "candidate_tool_names": sorted(evidence_plan.candidate_tool_names),
                "missing_tool_names": sorted(evidence_plan.missing_tool_names),
            },
            causation_id=step_started.event_id,
            sensitivity=sensitivity,
            state=event_state,
            step=event_step,
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
    EventType.MODEL_REQUEST_CREATED,
    EventType.MODEL_RESPONSE_RECEIVED,
    EventType.ASSISTANT_MESSAGE_CREATED,
    EventType.APPROVAL_REQUIRED,
    EventType.APPROVAL_GRANTED,
    EventType.APPROVAL_DENIED,
    EventType.APPROVAL_EXPIRED,
    EventType.APPROVAL_CANCELLED,
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_DENIED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_TIMEOUT,
    EventType.TOOL_CALL_CANCELLED,
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


def _tool_requires_live_state(metadata: dict[str, Any], tool_name: str) -> bool:
    raw_live_state_names = metadata.get("agent_live_state_tool_names")
    if isinstance(raw_live_state_names, list):
        return tool_name in {item for item in raw_live_state_names if isinstance(item, str)}
    if isinstance(raw_live_state_names, tuple):
        return tool_name in {item for item in raw_live_state_names if isinstance(item, str)}
    return False
