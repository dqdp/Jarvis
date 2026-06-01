from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from assistant_core.domain.tools import (
    SAFE_TOOL_OBSERVATION_ERROR_CODES,
    ToolObservationStatus,
    safe_tool_observation_error_code,
)


class ToolObservationRecoveryAction(StrEnum):
    CONTINUE = "continue"
    FINALIZE = "finalize"
    FAIL = "fail"


@dataclass(frozen=True)
class ToolObservationRecoveryDecision:
    action: ToolObservationRecoveryAction
    error_code: str | None = None
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ToolObservationRecoveryError(RuntimeError):
    def __init__(self, decision: ToolObservationRecoveryDecision) -> None:
        if decision.error_code is None:
            raise ValueError("recovery failure requires error_code")
        super().__init__(decision.error_code)
        self.error_code = decision.error_code
        self.error_message = decision.error_message or "tool observation failed"
        self.details = dict(decision.details)


class ToolObservationRecoveryPolicy:
    def decide(
        self,
        *,
        request_plan: tuple[str | None, frozenset[str] | None],
        observation_status: ToolObservationStatus,
        observation_error_code: str | None = None,
        tool_call_id: str | None = None,
        completed_observations: int,
        consecutive_failures: int,
        max_consecutive_failures: int,
        tool_requires_live_state: bool = False,
    ) -> ToolObservationRecoveryDecision:
        if observation_status == ToolObservationStatus.COMPLETED:
            return ToolObservationRecoveryDecision(action=ToolObservationRecoveryAction.CONTINUE)

        observation_error_code = safe_tool_observation_error_code(observation_error_code)
        details = {
            "observation_status": observation_status.value,
            "completed_observations": completed_observations,
            "tool_requires_live_state": tool_requires_live_state,
        }
        if observation_error_code is not None:
            details["observation_error_code"] = observation_error_code
        if tool_call_id is not None:
            details["tool_call_id"] = tool_call_id
        policy, _allowed = request_plan
        if (
            observation_status in _RECOVERABLE_OBSERVATION_STATUSES
            and observation_error_code not in _NON_RECOVERABLE_ERROR_CODES
            and _final_answer_allowed_after_observation(policy, completed_observations)
            and consecutive_failures <= max_consecutive_failures
        ):
            return ToolObservationRecoveryDecision(
                action=ToolObservationRecoveryAction.FINALIZE,
                error_code=_error_code(observation_status),
                error_message=_error_message(observation_status),
                details=details,
            )
        return ToolObservationRecoveryDecision(
            action=ToolObservationRecoveryAction.FAIL,
            error_code=_terminal_error_code(observation_status, observation_error_code),
            error_message=_error_message(observation_status),
            details=details,
        )


_RECOVERABLE_OBSERVATION_STATUSES = {
    ToolObservationStatus.FAILED,
    ToolObservationStatus.TIMEOUT,
}
_NON_RECOVERABLE_ERROR_CODES = {
    "invalid_arguments",
    "unknown_tool",
    "tool_disabled",
}
_DENIED_PASSTHROUGH_ERROR_CODES = set(SAFE_TOOL_OBSERVATION_ERROR_CODES) - {"tool_error"}


def _final_answer_allowed_after_observation(policy: str | None, completed_observations: int) -> bool:
    if policy == "available":
        return True
    if policy == "required" and completed_observations > 0:
        return True
    return False


def _error_code(status: ToolObservationStatus) -> str:
    return f"tool_observation_{status.value}"


def _terminal_error_code(status: ToolObservationStatus, observation_error_code: str | None) -> str:
    if observation_error_code in _NON_RECOVERABLE_ERROR_CODES:
        return observation_error_code
    if (
        status == ToolObservationStatus.DENIED
        and observation_error_code in _DENIED_PASSTHROUGH_ERROR_CODES
    ):
        return observation_error_code
    return _error_code(status)


def _error_message(status: ToolObservationStatus) -> str:
    return {
        ToolObservationStatus.APPROVAL_REQUIRED: "tool approval required",
        ToolObservationStatus.CANCELLED: "tool cancelled",
        ToolObservationStatus.DENIED: "tool denied",
        ToolObservationStatus.FAILED: "tool observation failed",
        ToolObservationStatus.TIMEOUT: "tool timeout",
    }.get(status, "tool observation failed")
