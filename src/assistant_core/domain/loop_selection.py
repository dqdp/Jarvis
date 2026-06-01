from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any

from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import (
    Capability,
    PermissionMode,
    PolicyDecisionOutcome,
)
from assistant_core.domain.sensitivity import Sensitivity


_STABLE_LABEL = re.compile(r"^[a-z0-9_.:-]+$")


class LoopSelectionMode(StrEnum):
    AUTO = "auto"
    CHAT = "chat"
    TOOLS = "tools"
    INVALID_OVERRIDE = "invalid_override"


class SelectionDecisionStatus(StrEnum):
    SELECTED = "selected"
    FALLBACK_CHAT = "fallback_chat"
    CLARIFICATION_REQUIRED = "clarification_required"
    REJECTED_BY_POLICY = "rejected_by_policy"
    TOOLS_UNAVAILABLE = "tools_unavailable"
    INVALID_OVERRIDE = "invalid_override"


class SelectionFallbackPreference(StrEnum):
    CHAT = "chat"
    FAIL_UNAVAILABLE = "fail_unavailable"
    ASK_CLARIFICATION = "ask_clarification"


@dataclass(frozen=True)
class LoopSelectionRequest:
    request_id: str
    conversation_id: str
    user_id: str
    requested_mode: LoopSelectionMode | str
    user_input: str
    current_message_sensitivity: Sensitivity | str
    active_project_namespace: str | None
    working_directory: str | None
    permission_mode: PermissionMode | str
    available_capabilities: frozenset[Capability | str] = field(default_factory=frozenset)
    available_tools_summary: tuple[Any, ...] = ()
    runtime_budget_summary: dict[str, Any] = field(default_factory=dict)
    model_profile_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("request_id", "conversation_id", "user_id", "user_input"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")
        object.__setattr__(self, "requested_mode", LoopSelectionMode(self.requested_mode))
        object.__setattr__(
            self,
            "current_message_sensitivity",
            Sensitivity(self.current_message_sensitivity),
        )
        object.__setattr__(self, "permission_mode", PermissionMode(self.permission_mode))
        object.__setattr__(
            self,
            "available_capabilities",
            frozenset(_capability_or_raw(value) for value in self.available_capabilities),
        )
        object.__setattr__(self, "available_tools_summary", tuple(self.available_tools_summary))
        if not isinstance(self.runtime_budget_summary, dict):
            raise ValueError("runtime_budget_summary must be a mapping")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")


@dataclass(frozen=True)
class LoopSelectionDecision:
    requested_mode: LoopSelectionMode | str
    selected_loop_strategy: LoopStrategyName | str | None
    selected_model_profile: str | None
    reason_code: str
    requires_tools: bool
    policy_outcome: PolicyDecisionOutcome | str | None
    approval_possible: bool
    fallback_behavior: SelectionFallbackPreference | str
    decision_status: SelectionDecisionStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_mode", LoopSelectionMode(self.requested_mode))
        if self.selected_loop_strategy is not None:
            object.__setattr__(
                self,
                "selected_loop_strategy",
                LoopStrategyName(self.selected_loop_strategy),
            )
        if not _is_stable_label(self.reason_code):
            raise ValueError("reason_code must be a stable label")
        if self.policy_outcome is not None:
            object.__setattr__(self, "policy_outcome", PolicyDecisionOutcome(self.policy_outcome))
        object.__setattr__(
            self,
            "fallback_behavior",
            SelectionFallbackPreference(self.fallback_behavior),
        )
        object.__setattr__(self, "decision_status", SelectionDecisionStatus(self.decision_status))

    def redacted_event_payload(self, *, request_id: str, conversation_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "conversation_id": conversation_id,
            "requested_mode": self.requested_mode.value,
            "selected_loop_strategy": _enum_value(self.selected_loop_strategy),
            "selected_model_profile": self.selected_model_profile,
            "reason_code": self.reason_code,
            "requires_tools": self.requires_tools,
            "policy_outcome": _enum_value(self.policy_outcome),
            "approval_possible": self.approval_possible,
            "fallback_behavior": self.fallback_behavior.value,
            "decision_status": self.decision_status.value,
        }


def _is_stable_label(value: str) -> bool:
    return isinstance(value, str) and bool(_STABLE_LABEL.fullmatch(value))


def _capability_or_raw(value: Capability | str) -> Capability | str:
    if isinstance(value, Capability):
        return value
    try:
        return Capability(value)
    except ValueError:
        return value


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
