from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from assistant_core.domain.loops import ToolProposalParseError
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.runtime.loops.observation_recovery import ToolObservationRecoveryError


@dataclass(frozen=True)
class LoopFailureDecision:
    error_code: str
    error_message: str = "tool loop failed"
    details: dict[str, Any] = field(default_factory=dict)


class LoopFailurePolicy:
    def decide(self, exc: Exception) -> LoopFailureDecision:
        if isinstance(exc, ToolObservationRecoveryError):
            return LoopFailureDecision(
                error_code=exc.error_code,
                error_message=exc.error_message,
                details=exc.details,
            )
        return LoopFailureDecision(error_code=loop_error_code(exc))


_SAFE_RUNTIME_ERROR_CODES = {
    "max_model_calls_exceeded",
    "max_steps_exceeded",
    "max_tool_calls_exceeded",
    "max_wall_time_exceeded",
    "request_plan_invalid_tool_policy",
    "request_plan_missing_tool_policy",
    "required_tool_evidence_missing",
    "required_tool_call_missing",
    "tool_not_allowed_by_request_plan",
    "tool_policy_disabled",
    "approval_cancelled",
    "approval_denied",
    "approval_expired",
}
_SAFE_TOOL_OBSERVATION_ERROR_CODES = {
    f"tool_observation_{status.value}"
    for status in ToolObservationStatus
    if status is not ToolObservationStatus.COMPLETED
}


def loop_error_code(exc: Exception) -> str:
    if isinstance(exc, ToolProposalParseError):
        return "malformed_tool_proposal"
    if isinstance(exc, RuntimeError):
        code = str(exc)
        if code in _SAFE_RUNTIME_ERROR_CODES or code in _SAFE_TOOL_OBSERVATION_ERROR_CODES:
            return code
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "runtime_error"
