from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from assistant_core.config.settings import Settings
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.policy import (
    Capability,
    CapabilityPolicyRequest,
    ContextPolicyRequest,
    MemoryWritePolicyRequest,
    ModelPolicyRequest,
    PermissionMode,
    PolicyDecision,
    PolicyDecisionOutcome,
    RiskClass,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventLogPort


class ConfigPolicyEngine:
    def __init__(self, settings: Settings, event_log: EventLogPort | None = None) -> None:
        self._settings = settings
        self._event_log = event_log

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

    async def evaluate_capability_request(
        self,
        request: CapabilityPolicyRequest,
    ) -> PolicyDecision:
        decision = self._evaluate_capability_request(request)
        if self._event_log is not None:
            await self._record_capability_decision(request, decision)
        return decision

    def _evaluate_capability_request(
        self,
        request: CapabilityPolicyRequest,
    ) -> PolicyDecision:
        scope = _decision_scope(request)
        mode = request.permission_mode or self._settings.permissions.mode
        if not isinstance(request.capability, Capability):
            return _capability_deny(
                "unknown_capability",
                "capability is not configured",
                request,
                mode,
                scope,
            )

        capability = request.capability
        action = self._configured_action(mode, capability)

        if _is_tool_capability(capability) and not self._settings.policy.tools_enabled:
            return _capability_deny(
                "tools_disabled",
                "tool capabilities are disabled by policy",
                request,
                mode,
                scope,
            )

        if request.sensitivity == Sensitivity.SECRET or RiskClass.SECRETS in request.risk_classes:
            return _capability_deny(
                "secret_access_denied",
                "secret access is denied",
                request,
                mode,
                scope,
            )

        if capability == Capability.MODEL_CLOUD or RiskClass.CLOUD in request.risk_classes:
            return _capability_deny(
                "cloud_disabled",
                "cloud capability is disabled",
                request,
                mode,
                scope,
            )

        if capability == Capability.TOOL_SHELL_DESTRUCTIVE or RiskClass.DESTRUCTIVE in request.risk_classes:
            return _capability_deny(
                "destructive_action_denied",
                "destructive actions are denied",
                request,
                mode,
                scope,
            )

        if capability == Capability.TOOL_SHELL_NETWORK or RiskClass.NETWORK in request.risk_classes:
            return _capability_deny(
                "network_denied_by_default",
                "network actions are denied by default",
                request,
                mode,
                scope,
            )

        if (
            mode == PermissionMode.AUTOMATION
            and capability == Capability.MEMORY_WRITE
            and (request.autonomous or RiskClass.AUTONOMOUS in request.risk_classes)
        ):
            return _capability_deny(
                "autonomous_memory_write_denied",
                "direct autonomous memory writes are denied",
                request,
                mode,
                scope,
            )

        if action == PolicyDecisionOutcome.DENY:
            return _capability_deny(
                _deny_code(capability),
                "capability is denied by policy",
                request,
                mode,
                scope,
            )

        if capability == Capability.TOOL_SHELL_READ:
            root_decision = self._shell_read_scope_decision(request, mode, scope)
            if root_decision is not None:
                return root_decision
        if capability in _SYSTEM_DIAGNOSTICS_CAPABILITIES:
            root_decision = self._system_diagnostics_scope_decision(request, mode, scope)
            if root_decision is not None:
                return root_decision

        if action is None:
            return _capability_deny(
                "capability_not_configured",
                "capability is not configured for this permission mode",
                request,
                mode,
                scope,
            )

        if _locked_down_denies_risk(mode, request.risk_classes):
            return _capability_deny(
                "locked_down_risk_denied",
                "locked down mode denies write and external side effect risks",
                request,
                mode,
                scope,
            )

        if RiskClass.WRITES_LOCAL in request.risk_classes and action == PolicyDecisionOutcome.ALLOW:
            return _capability_approval(
                "approval_required_for_write_risk",
                "local write risk requires approval",
                request,
                mode,
                scope,
            )

        if RiskClass.EXTERNAL_SIDE_EFFECT in request.risk_classes:
            return _capability_approval(
                "approval_required_for_external_side_effect",
                "external side effects require approval",
                request,
                mode,
                scope,
            )

        if action == PolicyDecisionOutcome.ALLOW:
            code = _allow_code(capability)
            return _capability_allow(code, "capability is allowed", request, mode, scope)

        if action == PolicyDecisionOutcome.APPROVAL_REQUIRED:
            code = _approval_code(capability)
            return _capability_approval(code, "capability requires approval", request, mode, scope)

        return _capability_deny(
            "capability_not_configured",
            "capability is not configured for this permission mode",
            request,
            mode,
            scope,
        )

    def _configured_action(
        self,
        mode: PermissionMode,
        capability: Capability,
    ) -> PolicyDecisionOutcome | None:
        raw = self._settings.permissions.modes.get(mode.value, {}).get(capability.value)
        if raw is None:
            return None
        return PolicyDecisionOutcome(raw)

    def _shell_read_scope_decision(
        self,
        request: CapabilityPolicyRequest,
        mode: PermissionMode,
        scope: dict[str, object],
    ) -> PolicyDecision | None:
        allowed_roots = _allowed_shell_roots(self._settings)
        scope["allowed_roots"] = [str(root) for root in allowed_roots]
        if request.working_directory is None:
            return _capability_deny(
                "working_directory_required",
                "working directory is required for shell read",
                request,
                mode,
                scope,
            )
        try:
            working_directory = Path(request.working_directory).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            scope["working_directory"] = "<redacted>"
            return _capability_deny(
                "invalid_working_directory",
                "working directory is invalid",
                request,
                mode,
                scope,
            )
        if not working_directory.exists() or not working_directory.is_dir():
            scope["working_directory"] = "<redacted>"
            return _capability_deny(
                "working_directory_not_found",
                "working directory must be an existing directory",
                request,
                mode,
                scope,
            )
        scope["working_directory"] = _stable_metadata_or_redacted(str(working_directory))
        if not _is_inside_any_root(working_directory, allowed_roots):
            return _capability_deny(
                "outside_allowed_workspace",
                "working directory is outside allowed workspace roots",
                request,
                mode,
                scope,
            )
        if _locked_down_denies_risk(mode, request.risk_classes):
            return _capability_deny(
                "locked_down_risk_denied",
                "locked down mode denies write and external side effect risks",
                request,
                mode,
                scope,
            )
        action = self._configured_action(mode, Capability.TOOL_SHELL_READ)
        if action == PolicyDecisionOutcome.APPROVAL_REQUIRED:
            return _capability_approval(
                "approval_required_for_shell_read",
                "shell read requires approval in this permission mode",
                request,
                mode,
                scope,
            )
        return None

    def _system_diagnostics_scope_decision(
        self,
        request: CapabilityPolicyRequest,
        mode: PermissionMode,
        scope: dict[str, object],
    ) -> PolicyDecision | None:
        allowed_roots = _allowed_system_diagnostics_roots(self._settings)
        scope["allowed_roots"] = [str(root) for root in allowed_roots]
        if request.working_directory is None:
            return _capability_deny(
                "working_directory_required",
                "working directory is required for system diagnostics",
                request,
                mode,
                scope,
            )
        try:
            working_directory = Path(request.working_directory).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            scope["working_directory"] = "<redacted>"
            return _capability_deny(
                "invalid_working_directory",
                "working directory is invalid",
                request,
                mode,
                scope,
            )
        if not working_directory.exists() or not working_directory.is_dir():
            scope["working_directory"] = "<redacted>"
            return _capability_deny(
                "working_directory_not_found",
                "working directory must be an existing directory",
                request,
                mode,
                scope,
            )
        scope["working_directory"] = _stable_metadata_or_redacted(str(working_directory))
        if not _is_inside_any_root(working_directory, allowed_roots):
            return _capability_deny(
                "outside_allowed_workspace",
                "working directory is outside allowed workspace roots",
                request,
                mode,
                scope,
            )
        action = self._configured_action(mode, request.capability)
        if action == PolicyDecisionOutcome.APPROVAL_REQUIRED:
            return _capability_approval(
                "approval_required_for_system_diagnostics",
                "system diagnostics requires approval in this permission mode",
                request,
                mode,
                scope,
            )
        return None

    async def _record_capability_decision(
        self,
        request: CapabilityPolicyRequest,
        decision: PolicyDecision,
    ) -> None:
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=EventType.POLICY_CAPABILITY_DECISION_RECORDED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                correlation_id=request.request_id,
                causation_id=None,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=request.user_id,
                source_component="policy_engine",
                source_node=None,
                sensitivity=request.sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "decision_id": decision.decision_id,
                    "outcome": decision.outcome.value,
                    "capability": _capability_value(decision.capability),
                    "risk_classes": sorted(risk.value for risk in decision.risk_classes),
                    "subject": decision.subject,
                    "created_at": decision.created_at.isoformat(),
                    "expires_at": (
                        decision.expires_at.isoformat()
                        if decision.expires_at is not None
                        else None
                    ),
                    "permission_mode": (
                        decision.permission_mode.value
                        if isinstance(decision.permission_mode, PermissionMode)
                        else decision.permission_mode
                    ),
                    "code": decision.code,
                    "reason": decision.reason,
                    "scope": _redact_mapping(decision.scope, redact_strings=True),
                    "redacted_payload": _redact_mapping(
                        decision.redacted_payload,
                        redact_strings=True,
                    ),
                },
                metadata={},
            ),
        )


def _allow(code: str, reason: str) -> PolicyDecision:
    return PolicyDecision(allowed=True, code=code, reason=reason)


def _deny(code: str, reason: str) -> PolicyDecision:
    return PolicyDecision(allowed=False, code=code, reason=reason)


def _capability_allow(
    code: str,
    reason: str,
    request: CapabilityPolicyRequest,
    mode: PermissionMode,
    scope: dict[str, object],
) -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        code=code,
        reason=reason,
        outcome=PolicyDecisionOutcome.ALLOW,
        capability=request.capability,
        risk_classes=request.risk_classes,
        sensitivity=request.sensitivity,
        permission_mode=mode,
        scope=scope,
        redacted_payload=_redact_mapping(request.redacted_payload, redact_strings=True),
    )


def _capability_deny(
    code: str,
    reason: str,
    request: CapabilityPolicyRequest,
    mode: PermissionMode,
    scope: dict[str, object],
) -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        code=code,
        reason=reason,
        outcome=PolicyDecisionOutcome.DENY,
        capability=request.capability,
        risk_classes=request.risk_classes,
        sensitivity=request.sensitivity,
        permission_mode=mode,
        scope=scope,
        redacted_payload=_redact_mapping(request.redacted_payload, redact_strings=True),
    )


def _capability_approval(
    code: str,
    reason: str,
    request: CapabilityPolicyRequest,
    mode: PermissionMode,
    scope: dict[str, object],
) -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        code=code,
        reason=reason,
        outcome=PolicyDecisionOutcome.APPROVAL_REQUIRED,
        capability=request.capability,
        risk_classes=request.risk_classes,
        sensitivity=request.sensitivity,
        permission_mode=mode,
        scope=scope,
        redacted_payload=_redact_mapping(request.redacted_payload, redact_strings=True),
    )


def _decision_scope(request: CapabilityPolicyRequest) -> dict[str, object]:
    scope: dict[str, object] = {}
    scope["capability"] = _capability_value(request.capability)
    if request.working_directory is not None:
        scope["working_directory"] = _stable_metadata_or_redacted(request.working_directory)
    if request.tool_name is not None:
        scope["tool_name"] = _stable_metadata_or_redacted(request.tool_name)
    if request.project_namespace is not None:
        scope["project_namespace"] = _stable_metadata_or_redacted(request.project_namespace)
    if request.integration_id is not None:
        scope["integration_id"] = _stable_metadata_or_redacted(request.integration_id)
    if request.task_id is not None:
        scope["task_id"] = _stable_metadata_or_redacted(request.task_id)
    return scope


def _stable_metadata_or_redacted(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/@")
    if (
        value
        and len(value) <= 256
        and all(character in allowed for character in value)
        and not _looks_sensitive(value)
    ):
        return value
    return "<redacted>"


def _locked_down_denies_risk(
    mode: PermissionMode,
    risk_classes: frozenset[RiskClass],
) -> bool:
    return (
        mode == PermissionMode.LOCKED_DOWN
        and bool({RiskClass.WRITES_LOCAL, RiskClass.EXTERNAL_SIDE_EFFECT} & risk_classes)
    )


def _capability_value(capability: Capability | str | None) -> str | None:
    if isinstance(capability, Capability):
        return capability.value
    return capability


def _is_tool_capability(capability: Capability) -> bool:
    return capability.value.startswith("tool.")


def _allowed_shell_roots(settings: Settings) -> list[Path]:
    config = settings.capabilities["tool.shell.read"]
    roots = config["allowed_roots"]
    return [Path(root).expanduser().resolve() for root in roots]


def _allowed_system_diagnostics_roots(settings: Settings) -> list[Path]:
    config = settings.capabilities["tool.system.read"]
    roots = config["allowed_roots"]
    return [Path(root).expanduser().resolve() for root in roots]


def _is_inside_any_root(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _allow_code(capability: Capability) -> str:
    if capability == Capability.TOOL_SAFE:
        return "allowed_safe_tool"
    if capability == Capability.TOOL_SHELL_READ:
        return "allowed_shell_read"
    if capability in _SYSTEM_DIAGNOSTICS_CAPABILITIES:
        return "allowed_system_diagnostics"
    if capability == Capability.CONTENT_RETRIEVE:
        return "allowed_content_retrieve"
    if capability == Capability.CONTEXT_INSPECT:
        return "allowed_context_inspect"
    return "allowed"


def _approval_code(capability: Capability) -> str:
    if capability in {Capability.TOOL_SHELL_WRITE, Capability.TOOL_FILESYSTEM_WRITE}:
        return "approval_required_for_write"
    if capability == Capability.TOOL_SHELL_READ:
        return "approval_required_for_shell_read"
    if capability in _SYSTEM_DIAGNOSTICS_CAPABILITIES:
        return "approval_required_for_system_diagnostics"
    return "approval_required"


def _deny_code(capability: Capability) -> str:
    if capability == Capability.MODEL_CLOUD:
        return "cloud_disabled"
    if capability == Capability.TOOL_SHELL_DESTRUCTIVE:
        return "destructive_action_denied"
    if capability == Capability.TOOL_SHELL_NETWORK:
        return "network_denied_by_default"
    return "capability_denied"


_SYSTEM_DIAGNOSTICS_CAPABILITIES = frozenset(
    {
        Capability.TOOL_SYSTEM_READ_PROCESS,
        Capability.TOOL_SYSTEM_READ_RESOURCES,
        Capability.TOOL_SYSTEM_READ_HARDWARE,
        Capability.TOOL_SYSTEM_READ_NETWORK,
        Capability.TOOL_SYSTEM_READ_SENSORS,
    },
)


_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "prompt",
    "secret",
    "token",
)
_SENSITIVE_VALUE_MARKERS = (
    ".aws",
    ".azure",
    ".gcp",
    ".kube",
    ".ssh",
    "-----begin",
    "id_ed25519",
    "id_rsa",
    "prompt",
    "sk-",
)


def _redact_mapping(
    mapping: dict[str, object],
    *,
    redact_strings: bool = False,
) -> dict[str, object]:
    return {
        key: _redact_value(key, value, redact_strings=redact_strings)
        for key, value in mapping.items()
    }


def _redact_value(key: str, value: object, *, redact_strings: bool = False) -> object:
    key_lower = key.lower()
    if any(marker in key_lower for marker in _SENSITIVE_KEY_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return _redact_mapping(value, redact_strings=redact_strings)
    if isinstance(value, list):
        return [_redact_value(key, item, redact_strings=redact_strings) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(key, item, redact_strings=redact_strings) for item in value)
    if isinstance(value, str) and (redact_strings or _looks_sensitive(value)):
        return "<redacted>"
    return value


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SENSITIVE_VALUE_MARKERS)
