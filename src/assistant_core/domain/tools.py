from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity


SENSITIVITY_ORDER = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.PROJECT: 1,
    Sensitivity.PERSONAL: 2,
    Sensitivity.INFRA: 3,
    Sensitivity.SECRET: 4,
}


class ToolObservationStatus(StrEnum):
    COMPLETED = "completed"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    display_name: str
    description: str
    capability: Capability
    risk_classes: frozenset[RiskClass]
    input_schema: dict[str, Any]
    adapter_name: str
    output_schema: dict[str, Any] | None = None
    default_timeout_seconds: float = 5.0
    max_output_bytes: int = 20_000
    sensitivity_ceiling: Sensitivity = Sensitivity.PROJECT
    requires_approval_by_default: bool = False
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name is required")
        if not self.display_name:
            raise ValueError("tool display_name is required")
        if not self.description:
            raise ValueError("tool description is required")
        if not isinstance(self.capability, Capability):
            object.__setattr__(self, "capability", Capability(self.capability))
        risk_classes = frozenset(
            risk if isinstance(risk, RiskClass) else RiskClass(risk)
            for risk in self.risk_classes
        )
        if not risk_classes:
            raise ValueError("tool risk_classes are required")
        object.__setattr__(self, "risk_classes", risk_classes)
        if not isinstance(self.input_schema, dict) or not self.input_schema:
            raise ValueError("tool input_schema is required")
        if not self.adapter_name:
            raise ValueError("tool adapter_name is required")
        if self.default_timeout_seconds <= 0:
            raise ValueError("tool default_timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("tool max_output_bytes must be positive")


@dataclass(frozen=True)
class ToolCallRequest:
    tool_name: str
    arguments: dict[str, Any]
    request_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None
    step_id: str | None = None
    user_id: str | None = None
    project_namespace: str | None = None
    working_directory: str | None = None
    sensitivity: Sensitivity = Sensitivity.PROJECT
    permission_mode: str | None = None
    approval_id: str | None = None
    idempotency_key: str | None = None
    timeout_seconds: float | None = None
    max_output_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name is required")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a mapping")
        if not all(isinstance(key, str) for key in self.arguments):
            raise ValueError("argument keys must be strings")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_bytes is not None and self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


@dataclass(frozen=True)
class ToolObservation:
    tool_call_id: str
    tool_name: str
    status: ToolObservationStatus
    content: str
    content_type: str
    sensitivity: Sensitivity
    truncated: bool
    output_bytes: int
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    error: dict[str, Any] | None = None
    artifact_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        *,
        tool_name: str,
        status: ToolObservationStatus,
        sensitivity: Sensitivity,
        started_at: datetime,
        completed_at: datetime,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolObservation:
        return cls(
            tool_call_id=str(uuid4()),
            tool_name=tool_name,
            status=status,
            content="",
            content_type="text/plain",
            sensitivity=sensitivity,
            truncated=False,
            output_bytes=0,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
            error=error,
            metadata=metadata or {},
        )
