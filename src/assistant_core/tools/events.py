from __future__ import annotations

from typing import Any

from assistant_core.domain.events import EventType
from assistant_core.domain.policy import Capability
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import SENSITIVITY_ORDER, ToolCallRequest, ToolObservation, ToolSpec


SYSTEM_DIAGNOSTICS_CAPABILITIES = frozenset(
    {
        Capability.TOOL_SYSTEM_READ_PROCESS,
        Capability.TOOL_SYSTEM_READ_RESOURCES,
        Capability.TOOL_SYSTEM_READ_HARDWARE,
        Capability.TOOL_SYSTEM_READ_NETWORK,
        Capability.TOOL_SYSTEM_READ_SENSORS,
    },
)


def tool_event_payload(
    spec: ToolSpec,
    *,
    policy_decision_id: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    payload = {
        "tool_name": spec.name,
        "capability": spec.capability.value,
        "risk_classes": sorted(risk.value for risk in spec.risk_classes),
        "policy_decision_id": policy_decision_id,
    }
    if error_code is not None:
        payload["error_code"] = error_code
    return payload


def audited_tool_event_payload(
    spec: ToolSpec,
    tool_metadata: dict[str, Any],
    *,
    policy_decision_id: str | None = None,
    error_code: str | None = None,
    policy_outcome: str | None = None,
    duration_ms: int | None = None,
    observation: ToolObservation | None = None,
) -> dict[str, Any]:
    safe_metadata = {
        key: value
        for key, value in tool_metadata.items()
        if key
        in {
            "argv",
            "cwd",
            "exit_code",
            "family",
            "platform",
            "raw_stderr_bytes",
            "raw_stdout_bytes",
            "source",
            "stderr_truncated",
            "stdout_truncated",
            "unavailable",
        }
    }
    payload: dict[str, Any] = {
        **tool_event_payload(
            spec,
            policy_decision_id=policy_decision_id,
            error_code=error_code,
        ),
        **safe_metadata,
    }
    if observation is not None:
        payload["truncated"] = observation.truncated
        payload["output_bytes"] = observation.output_bytes
        if observation.structured_schema is not None:
            payload["structured_schema"] = observation.structured_schema
        if observation.structured_schema_version is not None:
            payload["structured_schema_version"] = observation.structured_schema_version
        payload["parse_status"] = observation.parse_status.value
        if observation.parse_warnings:
            payload["parse_warnings"] = list(observation.parse_warnings)
    if policy_outcome is not None:
        payload["policy_outcome"] = policy_outcome
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return payload


def tool_output_sensitivity(request: ToolCallRequest, spec: ToolSpec) -> Sensitivity:
    return max_sensitivity(request.sensitivity, tool_output_sensitivity_floor(spec))


def tool_output_sensitivity_floor(spec: ToolSpec) -> Sensitivity:
    if spec.capability == Capability.TOOL_SHELL_READ:
        return Sensitivity.PROJECT
    if spec.capability in SYSTEM_DIAGNOSTICS_CAPABILITIES:
        return Sensitivity.INFRA
    return Sensitivity.PUBLIC


def max_sensitivity(first: Sensitivity, second: Sensitivity) -> Sensitivity:
    return first if SENSITIVITY_ORDER[first] >= SENSITIVITY_ORDER[second] else second


def is_shell_spec(spec: ToolSpec) -> bool:
    return spec.capability == Capability.TOOL_SHELL_READ


def is_system_diagnostics_spec(spec: ToolSpec) -> bool:
    return spec.capability in SYSTEM_DIAGNOSTICS_CAPABILITIES


def classified_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_CLASSIFIED
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_CLASSIFIED
    return None


def denied_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_DENIED
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED
    return None


def started_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_STARTED
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_STARTED
    return None


def completed_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_COMPLETED
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED
    return None


def failed_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_FAILED
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_FAILED
    return None


def timeout_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_TIMEOUT
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_TIMEOUT
    return None


def output_truncated_event_type(spec: ToolSpec) -> EventType | None:
    if is_shell_spec(spec):
        return EventType.TOOL_SHELL_OUTPUT_TRUNCATED
    if is_system_diagnostics_spec(spec):
        return EventType.TOOL_SYSTEM_DIAGNOSTICS_OUTPUT_TRUNCATED
    return None


def looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "api_key",
            "apikey",
            "authorization",
            "credential",
            ".env",
            ".ssh",
            "ghp_",
            "github_pat_",
            "akia",
            "id_ed25519",
            "id_rsa",
            "known_hosts",
            "password",
            "pat_",
            ".crt",
            ".key",
            ".pem",
            "-----begin",
            "openssh",
            "private key",
            "private_key",
            "prompt",
            "secret",
            "sk-",
            "sk_",
            "token",
        )
    )
