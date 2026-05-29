from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.events import EventType
from assistant_core.domain.policy import (
    Capability,
    CapabilityPolicyRequest,
    ContextPolicyRequest,
    MemoryWritePolicyRequest,
    ModelPolicyRequest,
    PermissionMode,
    PolicyDecisionOutcome,
    RiskClass,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.policy.engine import ConfigPolicyEngine


pytestmark = pytest.mark.unit


def _engine() -> ConfigPolicyEngine:
    settings = ConfigLoader(Path("config")).load("test")
    return ConfigPolicyEngine(settings)


def test_local_model_project_allowed() -> None:
    decision = asyncio.run(
        _engine().evaluate_model_request(
            ModelPolicyRequest(profile="local_main", sensitivity=Sensitivity.PROJECT),
        ),
    )

    assert decision.allowed is True


def test_local_model_secret_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_model_request(
            ModelPolicyRequest(profile="local_main", sensitivity=Sensitivity.SECRET),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "sensitivity_denied"


def test_cloud_model_denied_by_default() -> None:
    decision = asyncio.run(
        _engine().evaluate_model_request(
            ModelPolicyRequest(profile="cloud_reasoning", sensitivity=Sensitivity.PUBLIC),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "cloud_models_disabled"


def test_secret_memory_write_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_memory_write(
            MemoryWritePolicyRequest(
                namespace="project.personal_assistant",
                sensitivity=Sensitivity.SECRET,
            ),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "sensitivity_denied"


def test_secret_context_inclusion_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_context_inclusion(
            ContextPolicyRequest(source_ref="memory:secret", sensitivity=Sensitivity.SECRET),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "sensitivity_denied"


def test_permission_mode_developer_local_is_default() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    assert settings.permissions.mode == PermissionMode.DEVELOPER_LOCAL


def test_permission_modes_validate_known_values() -> None:
    assert {mode.value for mode in PermissionMode} == {
        "locked_down",
        "developer_local",
        "automation",
    }


def test_unknown_capability_is_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability="unknown.capability",
                risk_classes=frozenset({RiskClass.SAFE}),
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.allowed is False
    assert decision.code == "unknown_capability"


def test_safe_tool_capability_is_allowed() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE}),
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.ALLOW
    assert decision.allowed is True
    assert decision.code == "allowed_safe_tool"


def test_network_risk_overrides_allowed_capability() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE, RiskClass.NETWORK}),
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "network_denied_by_default"


def test_external_side_effect_risk_overrides_allowed_capability() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE, RiskClass.EXTERNAL_SIDE_EFFECT}),
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED
    assert decision.code == "approval_required_for_external_side_effect"


def test_external_side_effect_does_not_weaken_explicit_deny() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_WRITE,
                risk_classes=frozenset({RiskClass.WRITES_LOCAL, RiskClass.EXTERNAL_SIDE_EFFECT}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.LOCKED_DOWN,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "capability_denied"


def test_external_side_effect_does_not_enable_unconfigured_capability() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.INTEGRATION_GITHUB,
                risk_classes=frozenset({RiskClass.EXTERNAL_SIDE_EFFECT}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.DEVELOPER_LOCAL,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "capability_not_configured"


def test_writes_local_risk_cannot_pass_as_safe_tool() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE, RiskClass.WRITES_LOCAL}),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED
    assert decision.code == "approval_required_for_write_risk"


@pytest.mark.parametrize(
    "risk_class",
    [RiskClass.WRITES_LOCAL, RiskClass.EXTERNAL_SIDE_EFFECT],
)
def test_locked_down_hard_denies_write_and_external_risks(risk_class: RiskClass) -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE, risk_class}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.LOCKED_DOWN,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "locked_down_risk_denied"


def test_model_cloud_is_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.MODEL_CLOUD,
                risk_classes=frozenset({RiskClass.CLOUD}),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.allowed is False
    assert decision.code == "cloud_disabled"


def test_secret_access_is_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_FILESYSTEM_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY, RiskClass.SECRETS}),
                sensitivity=Sensitivity.SECRET,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.allowed is False
    assert decision.code == "secret_access_denied"


def test_shell_write_requires_approval() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_WRITE,
                risk_classes=frozenset({RiskClass.WRITES_LOCAL}),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED
    assert decision.allowed is False
    assert decision.code == "approval_required_for_write"


def test_shell_destructive_is_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_DESTRUCTIVE,
                risk_classes=frozenset({RiskClass.DESTRUCTIVE}),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "destructive_action_denied"


def test_locked_down_requires_approval_for_shell_read() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.LOCKED_DOWN,
                working_directory=str(Path.cwd()),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED
    assert decision.code == "approval_required_for_shell_read"


def test_developer_local_allows_shell_read_inside_allowed_workspace() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=str(Path.cwd()),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.ALLOW
    assert decision.code == "allowed_shell_read"


def test_shell_read_allowed_roots_are_not_process_cwd_relative(monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd().parent)
    engine = ConfigPolicyEngine(ConfigLoader(Path.cwd() / "Jarvis" / "config").load("test"))

    decision = asyncio.run(
        engine.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=str(Path.cwd()),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "outside_allowed_workspace"
    assert str(Path.cwd() / "Jarvis") in decision.scope["allowed_roots"]


def test_developer_local_denies_shell_read_outside_allowed_workspace() -> None:
    outside = Path.cwd().parent

    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=str(outside),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "outside_allowed_workspace"
    assert decision.scope["working_directory"] == str(outside)


def test_shell_read_scope_deny_overrides_approval_risks() -> None:
    outside = Path.cwd().parent

    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY, RiskClass.EXTERNAL_SIDE_EFFECT}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=str(outside),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "outside_allowed_workspace"


@pytest.mark.parametrize(
    "risk_class",
    [RiskClass.WRITES_LOCAL, RiskClass.EXTERNAL_SIDE_EFFECT],
)
def test_locked_down_shell_read_scope_allows_root_check_before_risk_deny(
    risk_class: RiskClass,
) -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY, risk_class}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.LOCKED_DOWN,
                working_directory=str(Path.cwd()),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "locked_down_risk_denied"


def test_automation_denies_direct_memory_write() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.MEMORY_WRITE,
                risk_classes=frozenset({RiskClass.WRITES_LOCAL, RiskClass.AUTONOMOUS}),
                sensitivity=Sensitivity.PERSONAL,
                permission_mode=PermissionMode.AUTOMATION,
                autonomous=True,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "autonomous_memory_write_denied"


def test_policy_decision_contains_stable_reason_and_scope() -> None:
    working_directory = str(Path.cwd().parent)

    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=working_directory,
            ),
        ),
    )

    assert decision.code == "outside_allowed_workspace"
    assert decision.reason
    assert decision.capability == Capability.TOOL_SHELL_READ
    assert decision.scope["working_directory"] == working_directory
    assert "allowed_roots" in decision.scope


def test_approval_required_outcome_contains_scoped_metadata() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_WRITE,
                risk_classes=frozenset({RiskClass.WRITES_LOCAL}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=str(Path.cwd()),
                tool_name="shell.write",
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED
    assert decision.scope["capability"] == Capability.TOOL_SHELL_WRITE.value
    assert decision.scope["tool_name"] == "shell.write"
    assert decision.scope["working_directory"] == str(Path.cwd())
    assert decision.decision_id
    assert decision.subject == Capability.TOOL_SHELL_WRITE.value
    assert decision.created_at is not None


def test_policy_scope_metadata_redacts_non_identifier_text() -> None:
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE}),
                sensitivity=Sensitivity.PROJECT,
                tool_name="please include this raw prompt in audit",
                project_namespace="project.personal_assistant",
                integration_id="integration.telegram",
                task_id="task:123",
            ),
        ),
    )

    assert decision.scope["tool_name"] == "<redacted>"
    assert decision.scope["project_namespace"] == "project.personal_assistant"
    assert decision.scope["integration_id"] == "integration.telegram"
    assert decision.scope["task_id"] == "task:123"
    assert "please include this raw prompt in audit" not in str(decision.scope).lower()


def test_working_directory_scope_does_not_store_unresolved_input() -> None:
    raw_text = "please include this raw prompt in audit"
    decision = asyncio.run(
        _engine().evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=raw_text,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.scope["working_directory"] != raw_text
    assert raw_text not in str(decision.scope)


def test_denied_capability_decision_emits_policy_capability_event() -> None:
    event_log = InMemoryEventLog()
    engine = ConfigPolicyEngine(ConfigLoader(Path("config")).load("test"), event_log=event_log)

    decision = asyncio.run(
        engine.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.MODEL_CLOUD,
                risk_classes=frozenset({RiskClass.CLOUD}),
                sensitivity=Sensitivity.PROJECT,
                request_id="request-1",
                conversation_id="conversation-1",
            ),
        ),
    )

    events = asyncio.run(event_log.query(EventFilter(request_id="request-1")))
    event = next(
        event
        for event in events
        if event.event_type == EventType.POLICY_CAPABILITY_DECISION_RECORDED
    )

    assert decision.code == "cloud_disabled"
    assert event.payload["outcome"] == "deny"
    assert event.payload["capability"] == "model.cloud"
    assert event.payload["code"] == "cloud_disabled"
    assert event.payload["decision_id"] == decision.decision_id


def test_approval_required_capability_decision_emits_policy_capability_event() -> None:
    event_log = InMemoryEventLog()
    engine = ConfigPolicyEngine(ConfigLoader(Path("config")).load("test"), event_log=event_log)

    asyncio.run(
        engine.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_WRITE,
                risk_classes=frozenset({RiskClass.WRITES_LOCAL}),
                sensitivity=Sensitivity.PROJECT,
                request_id="request-2",
            ),
        ),
    )

    events = asyncio.run(event_log.query(EventFilter(request_id="request-2")))
    event = next(
        event
        for event in events
        if event.event_type == EventType.POLICY_CAPABILITY_DECISION_RECORDED
    )

    assert event.payload["outcome"] == "approval_required"
    assert event.payload["code"] == "approval_required_for_write"


def test_policy_capability_event_payload_is_redacted() -> None:
    event_log = InMemoryEventLog()
    engine = ConfigPolicyEngine(ConfigLoader(Path("config")).load("test"), event_log=event_log)

    decision = asyncio.run(
        engine.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_FILESYSTEM_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY, RiskClass.SECRETS}),
                sensitivity=Sensitivity.SECRET,
                request_id="request-3",
                scope={
                    "payload": "FULL RAW PROMPT TEXT WITHOUT MARKER",
                    "path": "/Users/alex/.ssh/id_rsa",
                    "token": "sk-test-secret",
                    "raw_prompt": "full hidden prompt",
                },
                redacted_payload={
                    "details": "FULL RAW RESPONSE TEXT WITHOUT MARKER",
                    "path": "<redacted>",
                    "prompt": "full user or system prompt",
                },
            ),
        ),
    )

    assert "payload" not in decision.scope
    assert decision.redacted_payload == {
        "details": "<redacted>",
        "path": "<redacted>",
        "prompt": "<redacted>",
    }

    events = asyncio.run(event_log.query(EventFilter(request_id="request-3")))
    event = next(
        event
        for event in events
        if event.event_type == EventType.POLICY_CAPABILITY_DECISION_RECORDED
    )

    assert "raw_payload" not in event.payload
    assert "payload" not in event.payload["scope"]
    assert event.payload["redacted_payload"] == {
        "details": "<redacted>",
        "path": "<redacted>",
        "prompt": "<redacted>",
    }
    serialized_payload = str(event.payload).lower()
    assert "sk-test-secret" not in serialized_payload
    assert "id_rsa" not in serialized_payload
    assert "full hidden prompt" not in serialized_payload
    assert "full user or system prompt" not in serialized_payload
    assert "full raw prompt text without marker" not in serialized_payload
    assert "full raw response text without marker" not in serialized_payload
