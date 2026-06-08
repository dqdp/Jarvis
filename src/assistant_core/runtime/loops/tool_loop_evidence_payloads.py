from __future__ import annotations

from typing import Any

from assistant_core.domain.loops import LoopExecutionRequest, ToolObservationRef, ToolRequestPlan
from assistant_core.runtime.loops.tool_loop_evidence import (
    final_answer_missing_evidence_plan,
)


def failure_missing_evidence_payload(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    exc: Exception,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> dict[str, Any]:
    if str(exc) != "required_tool_evidence_missing":
        return {}
    try:
        evidence_plan = final_answer_missing_evidence_plan(
            request,
            request_plan,
            tool_observation_refs=tool_observation_refs,
        )
    except RuntimeError:
        return {}
    if evidence_plan is None:
        return {}
    missing_families = sorted(family.value for family in evidence_plan.missing_families)
    family = (
        missing_families[0]
        if missing_families
        else evidence_plan.family.value
        if evidence_plan.family is not None
        else None
    )
    return {
        "missing_evidence_family": family,
        "missing_evidence_families": missing_families,
        "candidate_evidence_kinds": sorted(
            kind.value for kind in evidence_plan.candidate_evidence_kinds
        ),
        "missing_evidence_kinds": sorted(
            kind.value for kind in evidence_plan.missing_evidence_kinds
        ),
        "candidate_tool_names": sorted(evidence_plan.candidate_tool_names),
        "missing_tool_names": sorted(evidence_plan.missing_tool_names),
    }
