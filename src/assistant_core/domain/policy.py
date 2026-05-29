from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from assistant_core.domain.sensitivity import Sensitivity


class Capability(StrEnum):
    MODEL_LOCAL = "model.local"
    MODEL_CLOUD = "model.cloud"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    MEMORY_LIFECYCLE = "memory.lifecycle"
    CONTEXT_INSPECT = "context.inspect"
    CONTENT_RETRIEVE = "content.retrieve"
    CONTENT_INGEST = "content.ingest"
    CONTENT_INDEX = "content.index"
    TOOL_SAFE = "tool.safe"
    TOOL_SHELL_READ = "tool.shell.read"
    TOOL_SHELL_WRITE = "tool.shell.write"
    TOOL_SHELL_NETWORK = "tool.shell.network"
    TOOL_SHELL_DESTRUCTIVE = "tool.shell.destructive"
    TOOL_FILESYSTEM_READ = "tool.filesystem.read"
    TOOL_FILESYSTEM_WRITE = "tool.filesystem.write"
    INTEGRATION_MCP = "integration.mcp"
    INTEGRATION_SEARCH = "integration.search"
    INTEGRATION_TELEGRAM = "integration.telegram"
    INTEGRATION_SPOTIFY = "integration.spotify"
    INTEGRATION_GITHUB = "integration.github"
    INTEGRATION_CALENDAR = "integration.calendar"
    INTEGRATION_MAIL = "integration.mail"
    TASK_SCHEDULE = "task.schedule"
    TASK_BACKGROUND = "task.background"
    TASK_SLEEP_REFLECTION = "task.sleep_reflection"
    VOICE_INPUT = "voice.input"
    VOICE_OUTPUT = "voice.output"
    VOICE_REALTIME = "voice.realtime"
    APPROVAL_REQUEST = "approval.request"
    APPROVAL_GRANT = "approval.grant"
    APPROVAL_DENY = "approval.deny"


class RiskClass(StrEnum):
    SAFE = "safe"
    READ_ONLY = "read_only"
    WRITES_LOCAL = "writes_local"
    NETWORK = "network"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    SECRETS = "secrets"
    DESTRUCTIVE = "destructive"
    AUTONOMOUS = "autonomous"
    CLOUD = "cloud"


class PermissionMode(StrEnum):
    LOCKED_DOWN = "locked_down"
    DEVELOPER_LOCAL = "developer_local"
    AUTOMATION = "automation"


class PolicyDecisionOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    reason: str
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    outcome: PolicyDecisionOutcome | str | None = None
    capability: Capability | str | None = None
    risk_classes: frozenset[RiskClass] = field(default_factory=frozenset)
    sensitivity: Sensitivity | None = None
    permission_mode: PermissionMode | str | None = None
    subject: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    redacted_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome is None:
            outcome = PolicyDecisionOutcome.ALLOW if self.allowed else PolicyDecisionOutcome.DENY
        else:
            outcome = (
                self.outcome
                if isinstance(self.outcome, PolicyDecisionOutcome)
                else PolicyDecisionOutcome(self.outcome)
            )
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "permission_mode",
            _permission_mode_or_none(self.permission_mode),
        )
        object.__setattr__(
            self,
            "capability",
            _capability_or_raw(self.capability),
        )
        object.__setattr__(
            self,
            "risk_classes",
            frozenset(_risk_class(value) for value in self.risk_classes),
        )
        if self.subject is None:
            object.__setattr__(self, "subject", _capability_value(self.capability))


@dataclass(frozen=True)
class ModelPolicyRequest:
    profile: str
    sensitivity: Sensitivity
    provider: str | None = None
    cloud: bool | None = None
    purpose: str | None = None
    request_id: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class MemoryWritePolicyRequest:
    namespace: str
    sensitivity: Sensitivity


@dataclass(frozen=True)
class ContextPolicyRequest:
    source_ref: str
    sensitivity: Sensitivity


@dataclass(frozen=True)
class CapabilityPolicyRequest:
    capability: Capability | str
    risk_classes: frozenset[RiskClass] = field(default_factory=frozenset)
    sensitivity: Sensitivity = Sensitivity.PROJECT
    permission_mode: PermissionMode | str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    request_id: str | None = None
    task_id: str | None = None
    project_namespace: str | None = None
    working_directory: str | None = None
    integration_id: str | None = None
    tool_name: str | None = None
    sensitivity_ceiling: Sensitivity | None = None
    autonomous: bool = False
    scope: dict[str, Any] = field(default_factory=dict)
    redacted_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", _capability_or_raw(self.capability))
        object.__setattr__(
            self,
            "risk_classes",
            frozenset(_risk_class(value) for value in self.risk_classes),
        )
        object.__setattr__(
            self,
            "permission_mode",
            _permission_mode_or_none(self.permission_mode),
        )


def _capability_or_raw(value: Capability | str | None) -> Capability | str | None:
    if value is None or isinstance(value, Capability):
        return value
    try:
        return Capability(value)
    except ValueError:
        return value


def _risk_class(value: RiskClass | str) -> RiskClass:
    return value if isinstance(value, RiskClass) else RiskClass(value)


def _permission_mode_or_none(
    value: PermissionMode | str | None,
) -> PermissionMode | None:
    if value is None or isinstance(value, PermissionMode):
        return value
    return PermissionMode(value)


def _capability_value(capability: Capability | str | None) -> str | None:
    if isinstance(capability, Capability):
        return capability.value
    return capability
