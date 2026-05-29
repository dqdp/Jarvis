from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.policy import (
    Capability,
    CapabilityPolicyRequest,
    PermissionMode,
    PolicyDecisionOutcome,
    RiskClass,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.policy import PolicyPort


pytestmark = pytest.mark.contract


def _engine() -> ConfigPolicyEngine:
    return ConfigPolicyEngine(ConfigLoader(Path("config")).load("test"))


def test_policy_port_evaluates_capability_request() -> None:
    policy: PolicyPort = _engine()

    decision = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE}),
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.ALLOW


def test_config_policy_engine_returns_allow_deny_and_approval_required() -> None:
    policy = _engine()

    allowed = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SAFE,
                risk_classes=frozenset({RiskClass.SAFE}),
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )
    denied = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.MODEL_CLOUD,
                risk_classes=frozenset({RiskClass.CLOUD}),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )
    approval = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_WRITE,
                risk_classes=frozenset({RiskClass.WRITES_LOCAL}),
                sensitivity=Sensitivity.PROJECT,
            ),
        ),
    )

    assert allowed.outcome == PolicyDecisionOutcome.ALLOW
    assert denied.outcome == PolicyDecisionOutcome.DENY
    assert approval.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED


def test_capability_policy_uses_permission_mode() -> None:
    policy = _engine()

    developer_decision = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.DEVELOPER_LOCAL,
                working_directory=str(Path.cwd()),
            ),
        ),
    )
    locked_down_decision = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                permission_mode=PermissionMode.LOCKED_DOWN,
                working_directory=str(Path.cwd()),
            ),
        ),
    )

    assert developer_decision.outcome == PolicyDecisionOutcome.ALLOW
    assert locked_down_decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED


def test_capability_policy_uses_sensitivity() -> None:
    policy = _engine()

    decision = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.CONTEXT_INSPECT,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.SECRET,
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "secret_access_denied"


def test_capability_policy_uses_working_directory_scope() -> None:
    policy = _engine()

    decision = asyncio.run(
        policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.TOOL_SHELL_READ,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=Sensitivity.PROJECT,
                working_directory=str(Path.cwd().parent),
            ),
        ),
    )

    assert decision.outcome == PolicyDecisionOutcome.DENY
    assert decision.code == "outside_allowed_workspace"
