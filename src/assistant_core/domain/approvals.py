from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity


DEFAULT_APPROVAL_TTL = timedelta(minutes=5)


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalDecision(StrEnum):
    GRANT = "grant"
    DENY = "deny"
    CANCEL = "cancel"


class ApprovalError(Exception):
    code = "approval_error"


class ApprovalNotFound(ApprovalError):
    code = "approval_not_found"


class ApprovalConflict(ApprovalError):
    def __init__(self, message: str, *, code: str = "approval_conflict") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ApprovalScope:
    capability: Capability | str
    risk_classes: frozenset[RiskClass | str]
    tool_name: str
    user_id: str | None
    request_id: str | None
    conversation_id: str | None
    step_id: str | None
    project_namespace: str | None
    working_directory: str | None
    sensitivity: Sensitivity | str
    permission_mode: str | None
    argument_keys: tuple[str, ...]
    arguments_hash: str

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("approval scope capability is required")
        if not self.risk_classes:
            raise ValueError("approval scope risk_classes are required")
        if not self.tool_name:
            raise ValueError("approval scope tool_name is required")
        if not self.arguments_hash:
            raise ValueError("approval scope arguments_hash is required")
        object.__setattr__(self, "capability", _capability(self.capability))
        object.__setattr__(
            self,
            "risk_classes",
            frozenset(_risk_class(risk) for risk in self.risk_classes),
        )
        if not isinstance(self.sensitivity, Sensitivity):
            object.__setattr__(self, "sensitivity", Sensitivity(self.sensitivity))
        object.__setattr__(self, "argument_keys", tuple(sorted(self.argument_keys)))

    def matches(self, other: ApprovalScope) -> bool:
        return self == other

    def payload(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "risk_classes": sorted(risk.value for risk in self.risk_classes),
            "tool_name": self.tool_name,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "step_id": self.step_id,
            "project_namespace": self.project_namespace,
            "working_directory": self.working_directory,
            "sensitivity": self.sensitivity.value,
            "permission_mode": self.permission_mode,
            "argument_keys": list(self.argument_keys),
            "arguments_hash": self.arguments_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ApprovalScope:
        return cls(
            capability=payload["capability"],
            risk_classes=frozenset(payload["risk_classes"]),
            tool_name=payload["tool_name"],
            user_id=payload.get("user_id"),
            request_id=payload.get("request_id"),
            conversation_id=payload.get("conversation_id"),
            step_id=payload.get("step_id"),
            project_namespace=payload.get("project_namespace"),
            working_directory=payload.get("working_directory"),
            sensitivity=payload.get("sensitivity", Sensitivity.PROJECT.value),
            permission_mode=payload.get("permission_mode"),
            argument_keys=tuple(payload.get("argument_keys", [])),
            arguments_hash=payload["arguments_hash"],
        )


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    status: ApprovalStatus | str
    capability: Capability | str
    risk_classes: frozenset[RiskClass | str]
    scope: ApprovalScope
    redacted_payload: dict[str, Any]
    requested_by: str | None
    created_at: datetime
    expires_at: datetime
    decision_actor_id: str | None = None
    decision_reason: str | None = None
    granted_at: datetime | None = None
    denied_at: datetime | None = None
    cancelled_at: datetime | None = None
    used_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.approval_id:
            raise ValueError("approval_id is required")
        if not self.capability:
            raise ValueError("approval capability is required")
        if not self.risk_classes:
            raise ValueError("approval risk_classes are required")
        if self.expires_at <= self.created_at:
            raise ValueError("approval expires_at must be after created_at")
        if not isinstance(self.scope, ApprovalScope):
            raise ValueError("approval scope is required")
        object.__setattr__(self, "status", ApprovalStatus(self.status))
        object.__setattr__(self, "capability", _capability(self.capability))
        object.__setattr__(
            self,
            "risk_classes",
            frozenset(_risk_class(risk) for risk in self.risk_classes),
        )
        object.__setattr__(self, "redacted_payload", _redact_mapping(self.redacted_payload))
        if self.scope.capability != self.capability:
            raise ValueError("approval scope capability must match approval capability")

    def grant(
        self,
        *,
        actor_id: str | None,
        now: datetime | None = None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        current_time = now or _now()
        self._ensure_pending(current_time)
        return replace(
            self,
            status=ApprovalStatus.GRANTED,
            decision_actor_id=actor_id,
            decision_reason=reason,
            granted_at=current_time,
        )

    def deny(
        self,
        *,
        actor_id: str | None,
        now: datetime | None = None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        current_time = now or _now()
        self._ensure_pending(current_time)
        return replace(
            self,
            status=ApprovalStatus.DENIED,
            decision_actor_id=actor_id,
            decision_reason=reason,
            denied_at=current_time,
        )

    def cancel(
        self,
        *,
        actor_id: str | None,
        now: datetime | None = None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        current_time = now or _now()
        self._ensure_pending(current_time)
        return replace(
            self,
            status=ApprovalStatus.CANCELLED,
            decision_actor_id=actor_id,
            decision_reason=reason,
            cancelled_at=current_time,
        )

    def expire(self, *, now: datetime | None = None) -> ApprovalRequest:
        current_time = now or _now()
        if self.status not in {ApprovalStatus.PENDING, ApprovalStatus.GRANTED}:
            raise ApprovalConflict(
                f"approval is {self.status.value}",
                code=f"approval_{self.status.value}",
            )
        if self.status == ApprovalStatus.GRANTED and self.used_at is not None:
            raise ApprovalConflict("approval was already used", code="approval_used")
        if current_time < self.expires_at:
            raise ApprovalConflict("approval is not expired", code="approval_not_expired")
        return replace(self, status=ApprovalStatus.EXPIRED)

    def consume(self, *, scope: ApprovalScope, now: datetime | None = None) -> ApprovalRequest:
        current_time = now or _now()
        if self.status != ApprovalStatus.GRANTED:
            raise ApprovalConflict(
                f"approval is {self.status.value}",
                code=f"approval_{self.status.value}",
            )
        if self.used_at is not None:
            raise ApprovalConflict("approval was already used", code="approval_used")
        if current_time >= self.expires_at:
            raise ApprovalConflict("approval expired", code="approval_expired")
        if not self.scope.matches(scope):
            raise ApprovalConflict("approval scope mismatch", code="approval_scope_mismatch")
        return replace(self, used_at=current_time)

    def _ensure_pending(self, now: datetime) -> None:
        if self.status != ApprovalStatus.PENDING:
            raise ApprovalConflict(
                f"approval is {self.status.value}",
                code=f"approval_{self.status.value}",
            )
        if now >= self.expires_at:
            raise ApprovalConflict("approval expired", code="approval_expired")


@dataclass(frozen=True)
class CreateApprovalCommand:
    scope: ApprovalScope
    redacted_payload: dict[str, Any]
    requested_by: str | None
    approval_id: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_request(self) -> ApprovalRequest:
        created_at = self.created_at or _now()
        return ApprovalRequest(
            approval_id=self.approval_id or str(uuid4()),
            status=ApprovalStatus.PENDING,
            capability=self.scope.capability,
            risk_classes=self.scope.risk_classes,
            scope=self.scope,
            redacted_payload=self.redacted_payload,
            requested_by=self.requested_by,
            created_at=created_at,
            expires_at=self.expires_at or created_at + DEFAULT_APPROVAL_TTL,
            metadata=self.metadata,
        )


def _capability(value: Capability | str) -> Capability:
    return value if isinstance(value, Capability) else Capability(value)


def _risk_class(value: RiskClass | str) -> RiskClass:
    return value if isinstance(value, RiskClass) else RiskClass(value)


def _now() -> datetime:
    return datetime.now(UTC)


_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_SENSITIVE_VALUE_MARKERS = (
    "-----begin",
    "akia",
    "github_pat_",
    "ghp_",
    "private key",
    "sk-",
    "sk_",
)


def _redact_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _redact_value(key, value) for key, value in mapping.items()}


def _redact_value(key: str, value: Any) -> Any:
    key_lower = key.lower()
    if any(marker in key_lower for marker in _SENSITIVE_KEY_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return _redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, item) for item in value)
    if isinstance(value, str) and _looks_sensitive(value):
        return "<redacted>"
    return value


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SENSITIVE_VALUE_MARKERS)
