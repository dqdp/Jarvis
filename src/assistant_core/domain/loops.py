from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from assistant_core.domain.conversations import ConversationMessage
from assistant_core.domain.policy import PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolObservation, ToolObservationStatus


class LoopStrategyName(StrEnum):
    MEMORY_AUGMENTED_ANSWER = "memory_augmented_answer"
    TOOL_REACT_LOOP = "tool_react_loop"


class LoopStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoopStepStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class UnknownLoopStrategy(ValueError):
    """Raised when a requested loop strategy is not registered."""


class ToolProposalParseError(ValueError):
    """Raised when model output cannot be converted into a safe tool proposal."""


@dataclass(frozen=True)
class LoopBudget:
    max_steps: int
    max_model_calls: int
    max_tool_calls: int
    max_wall_time_seconds: int
    max_context_assembly_seconds: int
    max_model_call_seconds: int
    max_consecutive_failures: int

    @classmethod
    def from_runtime_budget(cls, budget: Any) -> LoopBudget:
        return cls(
            max_steps=getattr(budget, "max_steps", 1),
            max_model_calls=budget.max_model_calls,
            max_tool_calls=budget.max_tool_calls,
            max_wall_time_seconds=budget.max_wall_time_seconds,
            max_context_assembly_seconds=budget.max_context_assembly_seconds,
            max_model_call_seconds=budget.max_model_call_seconds,
            max_consecutive_failures=getattr(budget, "max_consecutive_failures", 1),
        )

    def __post_init__(self) -> None:
        _require_positive("max_steps", self.max_steps)
        _require_positive("max_model_calls", self.max_model_calls)
        _require_non_negative("max_tool_calls", self.max_tool_calls)
        _require_positive("max_wall_time_seconds", self.max_wall_time_seconds)
        _require_non_negative("max_context_assembly_seconds", self.max_context_assembly_seconds)
        _require_non_negative("max_model_call_seconds", self.max_model_call_seconds)
        _require_positive("max_consecutive_failures", self.max_consecutive_failures)


@dataclass(frozen=True)
class LoopExecutionRequest:
    request_id: str
    conversation_id: str
    user_message_id: str
    user_id: str
    user_input: str
    active_project_namespace: str | None
    current_message_sensitivity: Sensitivity
    model_profile: str
    strategy_name: LoopStrategyName | str
    budget: LoopBudget
    correlation_id: str | None = None
    permission_mode: PermissionMode | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "conversation_id",
            "user_message_id",
            "user_id",
            "user_input",
            "model_profile",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")
        try:
            strategy_name = LoopStrategyName(self.strategy_name)
        except ValueError as exc:
            raise ValueError("strategy_name is required") from exc
        object.__setattr__(self, "strategy_name", strategy_name)
        if not isinstance(self.budget, LoopBudget):
            raise ValueError("budget is required")
        if not isinstance(self.current_message_sensitivity, Sensitivity):
            raise ValueError("current_message_sensitivity is required")
        if self.permission_mode is not None and not isinstance(
            self.permission_mode,
            PermissionMode,
        ):
            object.__setattr__(self, "permission_mode", PermissionMode(self.permission_mode))


@dataclass(frozen=True)
class LoopExecutionResult:
    status: LoopStatus
    response_text: str
    assistant_message: ConversationMessage | None
    used_model_calls: int
    used_tool_calls: int
    context_manifest_refs: tuple[str, ...]
    degraded: bool
    tool_observation_refs: tuple[ToolObservationRef, ...] = ()
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, LoopStatus):
            object.__setattr__(self, "status", LoopStatus(self.status))
        _require_non_negative("used_model_calls", self.used_model_calls)
        _require_non_negative("used_tool_calls", self.used_tool_calls)


@dataclass(frozen=True)
class LoopStreamEvent:
    event_type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ToolProposal:
    action: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"tool_call", "final_answer"}:
            raise ToolProposalParseError("unsupported tool proposal action")
        if self.action == "tool_call":
            if not self.tool_name:
                raise ToolProposalParseError("tool_call requires tool_name")
            if not isinstance(self.arguments, dict):
                raise ToolProposalParseError("tool_call arguments must be a mapping")
        if self.action == "final_answer" and not self.final_answer:
            raise ToolProposalParseError("final_answer requires final_answer")


@dataclass(frozen=True)
class ToolObservationRef:
    tool_call_id: str
    tool_name: str
    status: ToolObservationStatus | str
    content: str
    content_type: str
    sensitivity: Sensitivity
    truncated: bool = False
    error_code: str | None = None

    @classmethod
    def from_observation(cls, observation: ToolObservation) -> ToolObservationRef:
        return cls(
            tool_call_id=observation.tool_call_id,
            tool_name=observation.tool_name,
            status=observation.status,
            content=observation.content,
            content_type=observation.content_type,
            sensitivity=observation.sensitivity,
            truncated=observation.truncated,
            error_code=observation.error["code"] if observation.error else None,
        )

    def __post_init__(self) -> None:
        if not self.tool_call_id:
            raise ValueError("tool_call_id is required")
        if not self.tool_name:
            raise ValueError("tool_name is required")
        if not isinstance(self.status, ToolObservationStatus):
            object.__setattr__(self, "status", ToolObservationStatus(self.status))
        if not isinstance(self.sensitivity, Sensitivity):
            object.__setattr__(self, "sensitivity", Sensitivity(self.sensitivity))


@dataclass(frozen=True)
class LoopStep:
    step_id: str
    step_index: int
    status: LoopStepStatus
    action: str
    tool_call_id: str | None = None
    error_code: str | None = None


def parse_tool_proposal(value: dict[str, Any]) -> ToolProposal:
    if not isinstance(value, dict):
        raise ToolProposalParseError("tool proposal must be a mapping")
    action = value.get("action")
    if action == "final_answer":
        return ToolProposal(action="final_answer", final_answer=value.get("final_answer"))
    if action == "tool_call":
        return ToolProposal(
            action="tool_call",
            tool_name=value.get("tool_name"),
            arguments=value.get("arguments", {}),
        )
    raise ToolProposalParseError("unsupported tool proposal action")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
