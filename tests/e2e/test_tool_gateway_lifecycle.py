from __future__ import annotations

import asyncio

import pytest

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


class FixedPolicy:
    def __init__(self, outcome: PolicyDecisionOutcome) -> None:
        self.outcome = outcome

    async def evaluate_capability_request(self, request):
        return PolicyDecision(
            allowed=self.outcome == PolicyDecisionOutcome.ALLOW,
            code=self.outcome.value,
            reason="e2e policy decision",
            outcome=self.outcome,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


def _request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_name="fake.echo",
        arguments={"message": "hello"},
        request_id="req-e2e-tool",
        conversation_id="conv-e2e-tool",
        user_id="user-e2e-tool",
        sensitivity=Sensitivity.PROJECT,
    )


def _gateway(policy: FixedPolicy, adapter: FakeToolAdapter):
    event_log = InMemoryEventLog()
    return (
        ToolGateway(
            registry=ToolRegistry([adapter]),
            policy=policy,
            event_log=event_log,
        ),
        event_log,
    )


def test_safe_tool_call_emits_policy_and_tool_lifecycle_events() -> None:
    gateway, event_log = _gateway(FixedPolicy(PolicyDecisionOutcome.ALLOW), fake_echo_tool())

    observation = asyncio.run(gateway.invoke(_request()))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-e2e-tool")))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_OBSERVATION_RECORDED,
    ]


def test_denied_tool_call_emits_denied_event_and_skips_adapter_execution() -> None:
    adapter = fake_echo_tool()
    gateway, event_log = _gateway(FixedPolicy(PolicyDecisionOutcome.DENY), adapter)

    observation = asyncio.run(gateway.invoke(_request()))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-e2e-tool")))

    assert observation.status == ToolObservationStatus.DENIED
    assert adapter.call_count == 0
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_DENIED,
        EventType.TOOL_OBSERVATION_RECORDED,
    ]


def test_approval_required_tool_call_does_not_execute_adapter() -> None:
    adapter = fake_echo_tool()
    gateway, event_log = _gateway(FixedPolicy(PolicyDecisionOutcome.APPROVAL_REQUIRED), adapter)

    observation = asyncio.run(gateway.invoke(_request()))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-e2e-tool")))

    assert observation.status == ToolObservationStatus.APPROVAL_REQUIRED
    assert adapter.call_count == 0
    assert EventType.TOOL_CALL_STARTED not in [event.event_type for event in events]
