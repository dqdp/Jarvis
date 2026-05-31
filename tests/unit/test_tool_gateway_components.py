from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from assistant_core.approvals.in_memory import InMemoryApprovalStore
from assistant_core.domain.events import EventType
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import (
    ToolCallRequest,
    ToolInvocationResult,
    ToolObservationStatus,
    ToolParseStatus,
)
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.tools.approval_coordination import ToolApprovalCoordinator
from assistant_core.tools.audit import ToolInvocationAuditRecorder
from assistant_core.tools.fake import fake_echo_tool
from assistant_core.tools.results import completed_observation


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


def test_completed_observation_preserves_typed_tool_payload() -> None:
    result = ToolInvocationResult(
        content='{"stdout": "raw fallback"}',
        content_type="application/json",
        structured_content={
            "schema": "system.memory_overview",
            "available": "18000 MiB",
        },
        structured_schema="system.memory_overview",
        structured_schema_version=1,
        parse_status=ToolParseStatus.PARSED,
        parse_warnings=("estimated_available",),
    )

    observation = completed_observation(
        request=_request(),
        adapter=fake_echo_tool(),
        tool_call_id="tool-call-1",
        started_at=datetime.now(UTC),
        result=result,
        max_output_bytes=20_000,
    )

    assert observation.structured_content == {
        "schema": "system.memory_overview",
        "available": "18000 MiB",
    }
    assert observation.structured_schema == "system.memory_overview"
    assert observation.structured_schema_version == 1
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.parse_warnings == ("estimated_available",)


def test_completed_observation_redacts_sensitive_structured_payload() -> None:
    result = ToolInvocationResult(
        content='{"api_key": "sk-live-secret"}',
        content_type="application/json",
        structured_content={
            "api_key": "sk-live-secret",
            "nested": {
                "token": "plain-token-value",
                "safe": "visible",
            },
        },
        structured_schema="fake.secret_payload",
        structured_schema_version=1,
        parse_status=ToolParseStatus.PARSED,
    )

    observation = completed_observation(
        request=_request(),
        adapter=fake_echo_tool(),
        tool_call_id="tool-call-1",
        started_at=datetime.now(UTC),
        result=result,
        max_output_bytes=20_000,
    )

    assert observation.content == '{"redacted": true}'
    assert observation.structured_content == {
        "api_key": "<redacted>",
        "nested": {
            "token": "<redacted>",
            "safe": "visible",
        },
    }


def test_tool_audit_recorder_writes_typed_observation_metadata_without_payload() -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        recorder = ToolInvocationAuditRecorder(event_log)
        request = _request(causation_event_id="cause-typed", step_id="step-typed")
        observation = completed_observation(
            request=request,
            adapter=fake_echo_tool(),
            tool_call_id="tool-call-typed",
            started_at=datetime.now(UTC),
            result=ToolInvocationResult(
                content='{"stdout": "raw"}',
                content_type="application/json",
                structured_content={"free": "1024 MiB"},
                structured_schema="system.memory_overview",
                structured_schema_version=1,
                parse_status=ToolParseStatus.PARTIAL,
                parse_warnings=("total_memory_unavailable",),
            ),
            max_output_bytes=20_000,
        )
        await recorder.record_observation(
            request,
            observation,
            policy_decision_id="decision-typed",
        )
        return await event_log.query(EventFilter(request_id="request-tool"))

    events = asyncio.run(scenario())
    payload = events[0].payload

    assert payload["structured_schema"] == "system.memory_overview"
    assert payload["structured_schema_version"] == 1
    assert payload["parse_status"] == "partial"
    assert payload["parse_warnings"] == ["total_memory_unavailable"]
    assert "structured_content" not in payload
