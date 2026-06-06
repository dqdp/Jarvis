from __future__ import annotations

import json
from typing import Any

from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    ToolProposal,
    ToolRequestPlan,
)
from assistant_core.runtime.loops.tool_loop_evidence import (
    request_requires_initial_tool_evidence,
)


def should_use_final_chat_without_proposal(
    request_plan: ToolRequestPlan,
    *,
    used_tool_calls: int,
    completed_observations: int,
    budget: LoopBudget,
) -> bool:
    policy = request_plan.policy
    if policy == "disabled":
        return True
    if used_tool_calls < budget.max_tool_calls:
        return False
    return not (
        request_plan.final_answer_requires_observation() and completed_observations <= 0
    )


def should_fallback_to_final_chat_after_malformed_proposal(
    value: Any,
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    completed_observations: int,
) -> bool:
    policy = request_plan.policy
    if policy not in {"available", "required"}:
        return False
    if policy == "required" and completed_observations <= 0:
        return False
    if (
        completed_observations <= 0
        and request_requires_initial_tool_evidence(request, request_plan)
    ):
        return False
    if isinstance(value, dict) and value.get("action") == "tool_call":
        return False
    return True


def should_fallback_to_final_chat_after_structured_error(
    exc: Exception,
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    completed_observations: int,
) -> bool:
    if not is_structured_output_validation_error(exc):
        return False
    if (
        completed_observations <= 0
        and request_requires_initial_tool_evidence(request, request_plan)
    ):
        return False
    policy = request_plan.policy
    if policy == "available":
        return True
    if policy == "required" and completed_observations > 0:
        return True
    return False


def should_fallback_to_final_chat_after_proposal_timeout(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    completed_observations: int,
) -> bool:
    if (
        completed_observations <= 0
        and request_requires_initial_tool_evidence(request, request_plan)
    ):
        return False
    policy = request_plan.policy
    if policy == "available":
        return True
    if policy == "required" and completed_observations > 0:
        return True
    return False


def tool_call_signature(proposal: ToolProposal) -> tuple[str, str]:
    return (
        proposal.tool_name or "",
        json.dumps(proposal.arguments, sort_keys=True, separators=(",", ":"), default=str),
    )


def is_structured_output_validation_error(exc: Exception) -> bool:
    return type(exc).__name__ == "StructuredOutputValidationError"
