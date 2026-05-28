from __future__ import annotations

from dataclasses import dataclass

from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    reason: str


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
