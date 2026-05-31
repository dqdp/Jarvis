from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
import re
from typing import Any

from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import (
    Capability,
    PermissionMode,
    PolicyDecisionOutcome,
    RiskClass,
)
from assistant_core.domain.sensitivity import Sensitivity


_STABLE_LABEL = re.compile(r"^[a-z0-9_.:-]+$")


class LoopSelectionMode(StrEnum):
    AUTO = "auto"
    CHAT = "chat"
    TOOLS = "tools"
    INVALID_OVERRIDE = "invalid_override"


class IntentFamily(StrEnum):
    ORDINARY_CHAT = "ordinary_chat"
    PROJECT_DOCS_QUESTION = "project_docs_question"
    PROJECT_INSPECTION = "project_inspection"
    SYSTEM_DIAGNOSTICS = "system_diagnostics"
    SAFE_BUILTIN_TOOL = "safe_builtin_tool"
    CODE_EXECUTION = "code_execution"
    EXTERNAL_INTEGRATION = "external_integration"
    PLANNER_TASK = "planner_task"
    BACKGROUND_WORKFLOW = "background_workflow"
    UNKNOWN = "unknown"


class SelectionDecisionStatus(StrEnum):
    SELECTED = "selected"
    FALLBACK_CHAT = "fallback_chat"
    REJECTED_BY_POLICY = "rejected_by_policy"
    TOOLS_UNAVAILABLE = "tools_unavailable"
    INVALID_OVERRIDE = "invalid_override"
    CLASSIFIER_UNAVAILABLE = "classifier_unavailable"


class SelectionFallbackPreference(StrEnum):
    CHAT = "chat"
    FAIL_UNAVAILABLE = "fail_unavailable"
    ASK_CLARIFICATION = "ask_clarification"


@dataclass(frozen=True)
class CapabilityCandidate:
    capability: Capability | str
    intent_family: IntentFamily | str
    confidence: float
    requires_live_state: bool
    requires_execution: bool
    requires_write: bool
    tool_names: tuple[str, ...] = ()
    risk_classes: frozenset[RiskClass] = field(default_factory=frozenset)
    scope_hint: str | None = None
    evidence_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", _capability(self.capability))
        object.__setattr__(self, "intent_family", IntentFamily(self.intent_family))
        _require_confidence(self.confidence)
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        if not all(_is_stable_label(value) for value in self.tool_names):
            raise ValueError("tool_names must be stable references")
        object.__setattr__(
            self,
            "risk_classes",
            frozenset(
                risk if isinstance(risk, RiskClass) else RiskClass(risk)
                for risk in self.risk_classes
            ),
        )
        object.__setattr__(self, "evidence_codes", tuple(self.evidence_codes))
        if not all(_is_stable_label(value) for value in self.evidence_codes):
            raise ValueError("evidence_codes must be stable labels")
        if self.scope_hint is not None and not _is_stable_label(self.scope_hint):
            raise ValueError("scope_hint must be a stable label")

    def redacted_payload(self) -> dict[str, Any]:
        return {
            "capability": _enum_value(self.capability),
            "intent_family": self.intent_family.value,
            "confidence": self.confidence,
            "requires_live_state": self.requires_live_state,
            "requires_execution": self.requires_execution,
            "requires_write": self.requires_write,
            "tool_names": list(self.tool_names),
            "risk_classes": sorted(risk.value for risk in self.risk_classes),
            "evidence_codes": list(self.evidence_codes),
        }


@dataclass(frozen=True)
class IntentClassification:
    intent_family: IntentFamily | str
    confidence: float
    candidate_capabilities: tuple[CapabilityCandidate, ...] = ()
    requires_live_state: bool = False
    requires_execution: bool = False
    answer_without_tools_would_be_misleading: bool = False
    reason_code: str = "unknown"
    fallback_preference: SelectionFallbackPreference | str = SelectionFallbackPreference.CHAT
    classification_source: str = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_family", IntentFamily(self.intent_family))
        _require_confidence(self.confidence)
        object.__setattr__(self, "candidate_capabilities", tuple(self.candidate_capabilities))
        object.__setattr__(
            self,
            "fallback_preference",
            SelectionFallbackPreference(self.fallback_preference),
        )
        if not _is_stable_label(self.reason_code):
            raise ValueError("reason_code must be a stable label")
        if not _is_stable_label(self.classification_source):
            raise ValueError("classification_source must be a stable label")


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
    intent_family: IntentFamily | str
    reason_code: str
    confidence: float
    candidate_capabilities: tuple[CapabilityCandidate, ...]
    requires_tools: bool
    requires_live_state: bool
    policy_outcome: PolicyDecisionOutcome | str | None
    approval_possible: bool
    fallback_behavior: SelectionFallbackPreference | str
    decision_status: SelectionDecisionStatus | str
    classification_source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_mode", LoopSelectionMode(self.requested_mode))
        if self.selected_loop_strategy is not None:
            object.__setattr__(
                self,
                "selected_loop_strategy",
                LoopStrategyName(self.selected_loop_strategy),
            )
        object.__setattr__(self, "intent_family", IntentFamily(self.intent_family))
        if not _is_stable_label(self.reason_code):
            raise ValueError("reason_code must be a stable label")
        _require_confidence(self.confidence)
        object.__setattr__(self, "candidate_capabilities", tuple(self.candidate_capabilities))
        if self.policy_outcome is not None:
            object.__setattr__(self, "policy_outcome", PolicyDecisionOutcome(self.policy_outcome))
        object.__setattr__(
            self,
            "fallback_behavior",
            SelectionFallbackPreference(self.fallback_behavior),
        )
        object.__setattr__(self, "decision_status", SelectionDecisionStatus(self.decision_status))
        if not _is_stable_label(self.classification_source):
            raise ValueError("classification_source must be a stable label")

    def redacted_event_payload(self, *, request_id: str, conversation_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "conversation_id": conversation_id,
            "requested_mode": self.requested_mode.value,
            "selected_loop_strategy": _enum_value(self.selected_loop_strategy),
            "selected_model_profile": self.selected_model_profile,
            "intent_family": self.intent_family.value,
            "reason_code": self.reason_code,
            "classification_source": self.classification_source,
            "confidence": self.confidence,
            "candidate_capabilities": [
                candidate.redacted_payload() for candidate in self.candidate_capabilities
            ],
            "requires_tools": self.requires_tools,
            "requires_live_state": self.requires_live_state,
            "policy_outcome": _enum_value(self.policy_outcome),
            "approval_possible": self.approval_possible,
            "fallback_behavior": self.fallback_behavior.value,
            "decision_status": self.decision_status.value,
        }


def _require_confidence(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("confidence must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _is_stable_label(value: str) -> bool:
    return isinstance(value, str) and bool(_STABLE_LABEL.fullmatch(value))


def _capability_or_raw(value: Capability | str) -> Capability | str:
    if isinstance(value, Capability):
        return value
    try:
        return Capability(value)
    except ValueError:
        return value


def _capability(value: Capability | str) -> Capability:
    if isinstance(value, Capability):
        return value
    try:
        return Capability(value)
    except ValueError as exc:
        raise ValueError("capability must be a known capability") from exc


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value
