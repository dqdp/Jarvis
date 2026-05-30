from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from assistant_core.approvals.in_memory import InMemoryApprovalStore
from assistant_core.domain.events import EventType
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.tools.approval_coordination import ToolApprovalCoordinator
from assistant_core.tools.audit import ToolInvocationAuditRecorder
from assistant_core.tools.fake import fake_echo_tool


pytestmark = pytest.mark.unit


def _request(**overrides) -> ToolCallRequest:
    values = {
        "tool_name": "fake.echo",
        "arguments": {"message": "hello"},
        "request_id": "request-tool",
        "conversation_id": "conversation-tool",
        "user_id": "user-1",
        "sensitivity": Sensitivity.PROJECT,
    }
    values.update(overrides)
    return ToolCallRequest(**values)


def test_tool_approval_coordinator_creates_redacted_approval_metadata() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        approval_store = InMemoryApprovalStore(event_log=event_log)
        coordinator = ToolApprovalCoordinator(approval_store)

        metadata = await coordinator.create_metadata(
            _request(arguments={"message": "raw value"}),
            fake_echo_tool().spec,
            started_at=datetime.now(UTC),
            policy_decision_id="decision-1",
        )
        events = await event_log.query(EventFilter(request_id="request-tool"))
        return metadata, events

    metadata, events = asyncio.run(scenario())

    assert metadata["approval_id"]
    serialized_events = str([event.payload for event in events]).lower()
    assert "raw value" not in serialized_events
    assert "argument_keys" in serialized_events


def test_tool_approval_coordinator_returns_denied_observation_for_missing_store() -> None:
    async def scenario():
        coordinator = ToolApprovalCoordinator(None)
        return await coordinator.validate_approval(
            _request(approval_id="approval-missing"),
            fake_echo_tool().spec,
            tool_call_id="tool-call-1",
        )

    observation = asyncio.run(scenario())

    assert observation.status == ToolObservationStatus.DENIED
    assert observation.error["code"] == "approval_store_unavailable"


def test_tool_audit_recorder_writes_observation_without_raw_content() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        recorder = ToolInvocationAuditRecorder(event_log)
        request = _request(causation_event_id="cause-1", step_id="step-1")
        await recorder.record_event(
            EventType.TOOL_CALL_STARTED,
            request,
            tool_call_id="tool-call-1",
            payload={
                "tool_name": "fake.echo",
                "capability": Capability.TOOL_SAFE.value,
                "risk_classes": [RiskClass.SAFE.value],
            },
        )
        events = await event_log.query(EventFilter(request_id="request-tool"))
        return events

    events = asyncio.run(scenario())

    assert events[0].event_type == EventType.TOOL_CALL_STARTED
    assert events[0].causation_id == "cause-1"
    assert events[0].payload["step_id"] == "step-1"
