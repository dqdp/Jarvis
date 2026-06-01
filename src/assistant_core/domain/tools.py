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

DEFAULT_TOOL_OBSERVATION_ERROR_CODE = "tool_error"

SAFE_TOOL_OBSERVATION_ERROR_CODES = frozenset(
    {
        DEFAULT_TOOL_OBSERVATION_ERROR_CODE,
        "approval_cancelled",
        "approval_conflict",
        "approval_denied",
        "approval_error",
        "approval_expired",
        "approval_granted",
        "approval_not_expired",
        "approval_not_found",
        "approval_pending",
        "approval_required",
        "approval_required_for_external_side_effect",
        "approval_required_for_shell_read",
        "approval_required_for_system_diagnostics",
        "approval_required_for_write",
        "approval_required_for_write_risk",
        "approval_scope_mismatch",
        "approval_store_unavailable",
        "approval_used",
        "backend_not_found",
        "command_family_denied",
        "command_path_denied",
        "empty_command",
        "git_subcommand_denied",
        "interactive_command_denied",
        "invalid_path",
        "invalid_working_directory",
        "invalid_arguments",
        "diagnostics_family_mismatch",
        "line_count_exceeds_limit",
        "line_range_exceeds_limit",
        "mutating_command_denied",
        "network_client_denied",
        "path_argument_required",
        "path_argument_must_be_file",
        "path_outside_workspace",
        "secret_path_denied",
        "sensitivity_ceiling_exceeded",
        "sensor_mutation_denied",
        "sensor_polling_denied",
        "shell_syntax_denied",
        "tool_disabled",
        "tool_failed",
        "tool_timeout",
        "unknown_tool",
        "unsupported_arguments",
        "unsupported_command",
        "unsupported_platform_command",
        "working_directory_required",
    }
)


def safe_tool_observation_error_code(error_code: Any) -> str | None:
    if error_code is None:
        return None
    if isinstance(error_code, str) and error_code in SAFE_TOOL_OBSERVATION_ERROR_CODES:
        return error_code
    return DEFAULT_TOOL_OBSERVATION_ERROR_CODE


class ToolObservationStatus(StrEnum):
    COMPLETED = "completed"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolParseStatus(StrEnum):
    PARSED = "parsed"
    PARTIAL = "partial"
    UNPARSED = "unparsed"
    NOT_APPLICABLE = "not_applicable"


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
    causation_event_id: str | None = None
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
    structured_content: Any | None = None
    structured_schema: str | None = None
    structured_schema_version: int | None = None
    parse_status: ToolParseStatus | str = ToolParseStatus.NOT_APPLICABLE
    parse_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parse_status, ToolParseStatus):
            object.__setattr__(self, "parse_status", ToolParseStatus(self.parse_status))
        object.__setattr__(self, "parse_warnings", tuple(self.parse_warnings))

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


@dataclass(frozen=True)
class ToolInvocationResult:
    content: str
    content_type: str = "text/plain"
    truncated: bool = False
    output_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    structured_content: Any | None = None
    structured_schema: str | None = None
    structured_schema_version: int | None = None
    parse_status: ToolParseStatus | str = ToolParseStatus.NOT_APPLICABLE
    parse_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parse_status, ToolParseStatus):
            object.__setattr__(self, "parse_status", ToolParseStatus(self.parse_status))
        object.__setattr__(self, "parse_warnings", tuple(self.parse_warnings))
