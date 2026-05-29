from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from assistant_core.approvals.in_memory import InMemoryApprovalStore
from assistant_core.domain.events import EventType
from assistant_core.domain.policy import PolicyDecision, PolicyDecisionOutcome
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.tools.fake import FakeToolAdapter, fake_echo_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry


pytestmark = pytest.mark.e2e


class ApprovalRequiredPolicy:
    async def evaluate_capability_request(self, request):
        return PolicyDecision(
            allowed=False,
            code="approval_required",
            reason="approval required by e2e policy",
            outcome=PolicyDecisionOutcome.APPROVAL_REQUIRED,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


def _request(*, approval_id: str | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name="fake.echo",
        arguments={"message": "hello"},
        request_id="request-approval-e2e",
        conversation_id="conversation-approval-e2e",
        user_id="user-1",
        sensitivity=Sensitivity.PUBLIC,
        approval_id=approval_id,
    )


def _parts():
    event_log = InMemoryEventLog()
    approval_store = InMemoryApprovalStore(event_log=event_log)
    adapter = FakeToolAdapter(fake_echo_tool().spec, response="executed")
    gateway = ToolGateway(
        registry=ToolRegistry([adapter]),
        policy=ApprovalRequiredPolicy(),
        event_log=event_log,
        approval_store=approval_store,
    )
    return gateway, approval_store, adapter, event_log


def test_approval_required_tool_call_does_not_execute() -> None:
    gateway, _approval_store, adapter, _event_log = _parts()

    observation = asyncio.run(gateway.invoke(_request()))

    assert observation.status == ToolObservationStatus.APPROVAL_REQUIRED
    assert observation.metadata["approval_id"]
    assert adapter.call_count == 0


def test_granted_approval_allows_retry_execution() -> None:
    gateway, approval_store, adapter, _event_log = _parts()
    first = asyncio.run(gateway.invoke(_request()))

    asyncio.run(approval_store.grant_approval(first.metadata["approval_id"], actor_id="user-1"))
    second = asyncio.run(gateway.invoke(_request(approval_id=first.metadata["approval_id"])))

    assert second.status == ToolObservationStatus.COMPLETED
    assert second.content == "executed"
    assert adapter.call_count == 1


def test_denied_approval_prevents_execution() -> None:
    gateway, approval_store, adapter, _event_log = _parts()
    first = asyncio.run(gateway.invoke(_request()))

    asyncio.run(approval_store.deny_approval(first.metadata["approval_id"], actor_id="user-1"))
    second = asyncio.run(gateway.invoke(_request(approval_id=first.metadata["approval_id"])))

    assert second.status == ToolObservationStatus.DENIED
    assert second.error["code"] == "approval_denied"
    assert adapter.call_count == 0


def test_expired_approval_prevents_execution() -> None:
    gateway, approval_store, adapter, _event_log = _parts()
    first = asyncio.run(gateway.invoke(_request()))

    approval = asyncio.run(approval_store.get_approval(first.metadata["approval_id"]))
    assert approval is not None
    asyncio.run(
        approval_store.expire_stale(
            now=(approval.expires_at + timedelta(seconds=1)).astimezone(UTC),
        ),
    )
    second = asyncio.run(gateway.invoke(_request(approval_id=first.metadata["approval_id"])))

    assert second.status == ToolObservationStatus.DENIED
    assert second.error["code"] == "approval_expired"
    assert adapter.call_count == 0


def test_approval_events_are_emitted() -> None:
    gateway, approval_store, _adapter, event_log = _parts()
    first = asyncio.run(gateway.invoke(_request()))
    asyncio.run(approval_store.grant_approval(first.metadata["approval_id"], actor_id="user-1"))
    asyncio.run(gateway.invoke(_request(approval_id=first.metadata["approval_id"])))

    events = asyncio.run(event_log.query(EventFilter(request_id="request-approval-e2e")))

    assert EventType.APPROVAL_REQUIRED in [event.event_type for event in events]
    assert EventType.APPROVAL_GRANTED in [event.event_type for event in events]
    assert EventType.TOOL_CALL_APPROVED in [event.event_type for event in events]
