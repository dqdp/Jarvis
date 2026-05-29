from __future__ import annotations

import asyncio

import pytest

from assistant_core.domain.events import EventType
from assistant_core.domain.policy import PolicyDecision, PolicyDecisionOutcome
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.ports.tools import ToolGatewayPort
from assistant_core.tools.builtin import (
    calculator_tool,
    daemon_status_tool,
    datetime_now_tool,
)
from assistant_core.tools.fake import fake_echo_tool, fake_fail_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry


pytestmark = pytest.mark.contract


class AllowPolicy:
    def __init__(self, outcome: PolicyDecisionOutcome = PolicyDecisionOutcome.ALLOW) -> None:
        self.outcome = outcome
        self.requests = []

    async def evaluate_capability_request(self, request):
        self.requests.append(request)
        return PolicyDecision(
            allowed=self.outcome == PolicyDecisionOutcome.ALLOW,
            code=self.outcome.value,
            reason="contract policy decision",
            outcome=self.outcome,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


def _gateway(
    *,
    policy: AllowPolicy | None = None,
    event_log: InMemoryEventLog | None = None,
) -> tuple[ToolGatewayPort, AllowPolicy, InMemoryEventLog]:
    effective_policy = policy or AllowPolicy()
    effective_event_log = event_log or InMemoryEventLog()
    gateway = ToolGateway(
        registry=ToolRegistry(
            [
                fake_echo_tool(),
                fake_fail_tool(),
                datetime_now_tool(),
                calculator_tool(),
                daemon_status_tool(),
            ],
        ),
        policy=effective_policy,
        event_log=effective_event_log,
    )
    return gateway, effective_policy, effective_event_log


def _request(
    tool_name: str,
    arguments: dict[str, object] | None = None,
    *,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        arguments=arguments or {},
        request_id="req-contract-tool",
        conversation_id="conv-contract-tool",
        user_id="user-contract-tool",
        sensitivity=sensitivity,
    )


def test_tool_gateway_lists_enabled_tools() -> None:
    gateway, _policy, _event_log = _gateway()

    specs = asyncio.run(gateway.list_tools())

    assert [spec.name for spec in specs] == [
        "calculator.evaluate",
        "daemon.status",
        "datetime.now",
        "fake.echo",
        "fake.fail",
    ]


def test_tool_gateway_gets_tool_by_name() -> None:
    gateway, _policy, _event_log = _gateway()

    spec = asyncio.run(gateway.get_tool("fake.echo"))

    assert spec is not None
    assert spec.name == "fake.echo"


def test_tool_gateway_invokes_fake_echo() -> None:
    gateway, policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(_request("fake.echo", {"message": "hello"})),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content == "hello"
    assert policy.requests[0].tool_name == "fake.echo"


def test_tool_gateway_invokes_datetime_now() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(gateway.invoke(_request("datetime.now", sensitivity=Sensitivity.PUBLIC)))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert "iso" in observation.content


def test_tool_gateway_invokes_calculator_evaluate() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(
        gateway.invoke(
            _request(
                "calculator.evaluate",
                {"expression": "1 + 2 * 3"},
                sensitivity=Sensitivity.PUBLIC,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content == "7"


def test_tool_gateway_invokes_daemon_status() -> None:
    gateway, _policy, _event_log = _gateway()

    observation = asyncio.run(gateway.invoke(_request("daemon.status", sensitivity=Sensitivity.PUBLIC)))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.content_type == "application/json"
    assert '"status": "ok"' in observation.content


def test_tool_gateway_records_completed_lifecycle_events() -> None:
    event_log = InMemoryEventLog()
    gateway, _policy, _event_log = _gateway(event_log=event_log)

    observation = asyncio.run(
        gateway.invoke(_request("fake.echo", {"message": "hello"})),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-contract-tool")))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_OBSERVATION_RECORDED,
    ]


def test_tool_gateway_records_failed_lifecycle_events() -> None:
    event_log = InMemoryEventLog()
    gateway, _policy, _event_log = _gateway(event_log=event_log)

    observation = asyncio.run(gateway.invoke(_request("fake.fail")))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-contract-tool")))

    assert observation.status == ToolObservationStatus.FAILED
    assert EventType.TOOL_CALL_FAILED in [event.event_type for event in events]
    assert events[-1].event_type == EventType.TOOL_OBSERVATION_RECORDED


def test_tool_gateway_records_denied_lifecycle_events() -> None:
    event_log = InMemoryEventLog()
    gateway, _policy, _event_log = _gateway(
        policy=AllowPolicy(PolicyDecisionOutcome.DENY),
        event_log=event_log,
    )

    observation = asyncio.run(
        gateway.invoke(_request("fake.echo", {"message": "hello"})),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-contract-tool")))

    assert observation.status == ToolObservationStatus.DENIED
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_DENIED,
        EventType.TOOL_OBSERVATION_RECORDED,
    ]
