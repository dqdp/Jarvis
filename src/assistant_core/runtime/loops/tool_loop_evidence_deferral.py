from __future__ import annotations

from typing import Awaitable, Callable

from assistant_core.domain.events import EventEnvelope, EventType
from assistant_core.domain.loops import (
    AgentLoopState,
    AgentLoopStep,
    LoopExecutionRequest,
    ToolObservationRef,
    ToolRequestPlan,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.loops.tool_loop_evidence import (
    LiveStateEvidencePlan,
    final_answer_deferred_missing_evidence_plan,
)


AppendEvent = Callable[..., Awaitable[EventEnvelope]]


async def defer_final_answer_if_missing_evidence(
    append_event: AppendEvent,
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
    evidence_plan = final_answer_deferred_missing_evidence_plan(
        request,
        request_plan,
        tool_observation_refs=tool_observation_refs,
        used_tool_calls=used_tool_calls,
    )
    if evidence_plan is None:
        return False
    await _append_deferred_evidence_step(
        append_event,
        request,
        step_started=step_started,
        step_id=step_id,
        step_index=step_index,
        sensitivity=sensitivity,
        evidence_plan=evidence_plan,
        action="final_answer_deferred_missing_evidence",
        event_state=event_state,
        event_step=event_step,
    )
    return True


async def _append_deferred_evidence_step(
    append_event: AppendEvent,
    request: LoopExecutionRequest,
    *,
    step_started: EventEnvelope,
    step_id: str,
    step_index: int,
    sensitivity: Sensitivity,
    evidence_plan: LiveStateEvidencePlan,
    action: str,
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
    payload = {
        "strategy_name": request.strategy_name.value,
        "step_id": step_id,
        "step_index": step_index,
        "action": action,
        "missing_evidence_family": family,
        "missing_evidence_families": missing_families,
        "candidate_tool_names": sorted(evidence_plan.candidate_tool_names),
        "missing_tool_names": sorted(evidence_plan.missing_tool_names),
    }
    return await append_event(
        EventType.AGENT_STEP_COMPLETED,
        request,
        payload=payload,
        causation_id=step_started.event_id,
        sensitivity=sensitivity,
        state=event_state,
        step=event_step,
    )
