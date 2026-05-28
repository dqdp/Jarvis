from __future__ import annotations

from assistant_core.config.settings import Settings
from assistant_core.domain.policy import (
    ContextPolicyRequest,
    MemoryWritePolicyRequest,
    ModelPolicyRequest,
    PolicyDecision,
)


class ConfigPolicyEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def evaluate_model_request(
        self,
        request: ModelPolicyRequest,
    ) -> PolicyDecision:
        profile = self._settings.model_profiles.get(request.profile)
        if profile is None:
            return _deny("unknown_model_profile", "model profile is not configured")

        if profile.cloud and not self._settings.policy.cloud_models_enabled:
            return _deny("cloud_models_disabled", "cloud model profiles are disabled")
        if not profile.enabled:
            return _deny("model_profile_disabled", "model profile is disabled")

        access_key = "cloud" if profile.cloud else "local"
        access = self._settings.policy.model_access.get(access_key, {})
        if request.sensitivity.value in access.get("deny_sensitivity", []):
            return _deny("sensitivity_denied", "sensitivity is denied for model access")

        allowed = access.get("allow_sensitivity", [])
        if allowed and request.sensitivity.value not in allowed:
            return _deny("sensitivity_not_allowed", "sensitivity is not allowed for model access")

        return _allow("allowed", "model request is allowed")

    async def evaluate_memory_write(
        self,
        request: MemoryWritePolicyRequest,
    ) -> PolicyDecision:
        if request.sensitivity.value in self._settings.policy.memory_write.deny_sensitivity:
            return _deny("sensitivity_denied", "sensitivity is denied for memory writes")

        return _allow("allowed", "memory write is allowed")

    async def evaluate_context_inclusion(
        self,
        request: ContextPolicyRequest,
    ) -> PolicyDecision:
        if request.sensitivity.value in self._settings.policy.context_inclusion.deny_sensitivity:
            return _deny("sensitivity_denied", "sensitivity is denied for context inclusion")

        return _allow("allowed", "context source is allowed")


def _allow(code: str, reason: str) -> PolicyDecision:
    return PolicyDecision(allowed=True, code=code, reason=reason)


def _deny(code: str, reason: str) -> PolicyDecision:
    return PolicyDecision(allowed=False, code=code, reason=reason)
