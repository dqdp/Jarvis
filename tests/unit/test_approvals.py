from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from assistant_core.domain.approvals import (
    ApprovalConflict,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStatus,
)
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity


pytestmark = pytest.mark.unit


def _scope(
    *,
    tool_name: str = "fake.echo",
    request_id: str = "request-1",
    arguments_hash: str = "sha256:args",
) -> ApprovalScope:
    return ApprovalScope(
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        tool_name=tool_name,
        user_id="user-1",
        request_id=request_id,
        conversation_id="conversation-1",
        step_id="step-1",
        project_namespace="project.personal_assistant",
        working_directory=None,
        sensitivity=Sensitivity.PROJECT,
        permission_mode="developer_local",
        argument_keys=("message",),
        arguments_hash=arguments_hash,
    )


def _approval(*, expires_at: datetime | None = None) -> ApprovalRequest:
    now = datetime(2026, 5, 29, 10, 0, tzinfo=UTC)
    return ApprovalRequest(
        approval_id="approval-1",
        status=ApprovalStatus.PENDING,
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        scope=_scope(),
        redacted_payload={"tool_name": "fake.echo", "argument_keys": ["message"]},
        requested_by="user-1",
        created_at=now,
        expires_at=expires_at or now + timedelta(minutes=5),
    )


def test_approval_request_requires_capability_risk_scope_and_expiry() -> None:
    now = datetime(2026, 5, 29, 10, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="capability"):
        ApprovalRequest(
            approval_id="approval-1",
            status=ApprovalStatus.PENDING,
            capability="",
            risk_classes=frozenset({RiskClass.SAFE}),
            scope=_scope(),
            redacted_payload={},
            requested_by="user-1",
            created_at=now,
            expires_at=now + timedelta(minutes=5),
        )

    with pytest.raises(ValueError, match="risk_classes"):
        ApprovalRequest(
            approval_id="approval-1",
            status=ApprovalStatus.PENDING,
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset(),
            scope=_scope(),
            redacted_payload={},
            requested_by="user-1",
            created_at=now,
            expires_at=now + timedelta(minutes=5),
        )

    with pytest.raises(ValueError, match="expires_at"):
        ApprovalRequest(
            approval_id="approval-1",
            status=ApprovalStatus.PENDING,
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            scope=_scope(),
            redacted_payload={},
            requested_by="user-1",
            created_at=now,
            expires_at=now,
        )


def test_approval_grant_changes_pending_to_granted() -> None:
    now = datetime(2026, 5, 29, 10, 1, tzinfo=UTC)
    granted = _approval().grant(actor_id="user-1", now=now)

    assert granted.status == ApprovalStatus.GRANTED
    assert granted.granted_at == now
    assert granted.decision_actor_id == "user-1"


def test_approval_deny_changes_pending_to_denied() -> None:
    now = datetime(2026, 5, 29, 10, 1, tzinfo=UTC)
    denied = _approval().deny(actor_id="user-1", now=now, reason="user denied")

    assert denied.status == ApprovalStatus.DENIED
    assert denied.denied_at == now
    assert denied.decision_reason == "user denied"


def test_approval_cancel_changes_pending_to_cancelled() -> None:
    now = datetime(2026, 5, 29, 10, 1, tzinfo=UTC)
    cancelled = _approval().cancel(actor_id="user-1", now=now, reason="ctrl-c")

    assert cancelled.status == ApprovalStatus.CANCELLED
    assert cancelled.cancelled_at == now
    assert cancelled.decision_reason == "ctrl-c"


def test_expired_approval_cannot_be_granted() -> None:
    now = datetime(2026, 5, 29, 10, 0, tzinfo=UTC)
    approval = _approval(expires_at=now + timedelta(seconds=1))

    with pytest.raises(ApprovalConflict, match="expired"):
        approval.grant(actor_id="user-1", now=now + timedelta(seconds=2))


def test_denied_approval_cannot_be_reused() -> None:
    denied = _approval().deny(
        actor_id="user-1",
        now=datetime(2026, 5, 29, 10, 1, tzinfo=UTC),
        reason="no",
    )

    with pytest.raises(ApprovalConflict, match="denied"):
        denied.consume(scope=_scope(), now=datetime(2026, 5, 29, 10, 2, tzinfo=UTC))


def test_granted_approval_cannot_be_reused_for_different_tool_call() -> None:
    granted = _approval().grant(
        actor_id="user-1",
        now=datetime(2026, 5, 29, 10, 1, tzinfo=UTC),
    )

    with pytest.raises(ApprovalConflict, match="scope"):
        granted.consume(
            scope=_scope(arguments_hash="sha256:different"),
            now=datetime(2026, 5, 29, 10, 2, tzinfo=UTC),
        )


def test_granted_unused_approval_can_expire_after_ttl() -> None:
    granted = _approval(
        expires_at=datetime(2026, 5, 29, 10, 2, tzinfo=UTC),
    ).grant(
        actor_id="user-1",
        now=datetime(2026, 5, 29, 10, 1, tzinfo=UTC),
    )

    expired = granted.expire(now=datetime(2026, 5, 29, 10, 3, tzinfo=UTC))

    assert expired.status == ApprovalStatus.EXPIRED


def test_approval_scope_must_match_tool_call_and_capability() -> None:
    scope = _scope()

    assert scope.matches(_scope())
    assert not scope.matches(replace(scope, tool_name="fake.other"))
    assert not scope.matches(replace(scope, capability=Capability.TOOL_SHELL_READ))
    assert not scope.matches(replace(scope, user_id="user-2"))
    assert not scope.matches(replace(scope, sensitivity=Sensitivity.SECRET))


def test_approval_request_redacts_secret_shaped_payload() -> None:
    approval = ApprovalRequest(
        approval_id="approval-secret",
        status=ApprovalStatus.PENDING,
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        scope=_scope(),
        redacted_payload={
            "tool_name": "fake.echo",
            "api_key": "sk-live-secret",
            "nested": {"token": "github_pat_secret"},
        },
        requested_by="user-1",
        created_at=datetime(2026, 5, 29, 10, 0, tzinfo=UTC),
        expires_at=datetime(2026, 5, 29, 10, 5, tzinfo=UTC),
    )

    assert approval.redacted_payload == {
        "tool_name": "fake.echo",
        "api_key": "<redacted>",
        "nested": {"token": "<redacted>"},
    }
