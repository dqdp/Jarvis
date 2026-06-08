from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from assistant_core.domain.approvals import ApprovalRequest, ApprovalScope, ApprovalStatus
from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    LoopStrategyName,
    ToolObservationRef,
    ToolProposal,
    ToolProposalParseError,
)
from assistant_core.domain.policy import Capability, PermissionMode, RiskClass
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolObservation, ToolObservationStatus, ToolParseStatus
from assistant_core.runtime.loops.tool_approval import ApprovalWaiter
from assistant_core.runtime.loops.tool_loop_deterministic import (
    deterministic_datetime_now_response,
)
from assistant_core.runtime.loops.tool_proposal_executor import ToolProposalExecutor


pytestmark = pytest.mark.unit


def _budget() -> LoopBudget:
    return LoopBudget(
        max_steps=4,
        max_model_calls=4,
        max_tool_calls=2,
        max_wall_time_seconds=60,
        max_context_assembly_seconds=10,
        max_model_call_seconds=60,
        max_consecutive_failures=1,
    )


def _loop_request() -> LoopExecutionRequest:
    return LoopExecutionRequest(
        request_id="request-tool",
        conversation_id="conversation-tool",
        user_message_id="message-user",
        user_id="user-1",
        user_input="use tool",
        active_project_namespace="project.personal_assistant",
        current_message_sensitivity=Sensitivity.PROJECT,
        model_profile="local_main",
        strategy_name=LoopStrategyName.TOOL_REACT_LOOP,
        budget=_budget(),
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
    )


def _approval(status: ApprovalStatus) -> ApprovalRequest:
    now = datetime.now(UTC)
    scope = ApprovalScope(
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        tool_name="fake.echo",
        user_id="user-1",
        request_id="request-tool",
        conversation_id="conversation-tool",
        step_id="step-1",
        project_namespace="project.personal_assistant",
        working_directory=None,
        sensitivity=Sensitivity.PROJECT,
        permission_mode=PermissionMode.DEVELOPER_LOCAL.value,
        argument_keys=("message",),
        arguments_hash="hash",
    )
    return ApprovalRequest(
        approval_id="approval-1",
        status=status,
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        scope=scope,
        requested_by="user-1",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
        redacted_payload={},
    )


class FakeApprovalStore:
    def __init__(self, approval: ApprovalRequest) -> None:
        self.approval = approval
        self.expire_calls = 0
        self.cancel_calls = []

    async def expire_stale(self, *, now):
        self.expire_calls += 1
        return []

    async def get_approval(self, approval_id: str):
        assert approval_id == self.approval.approval_id
        return self.approval

    async def cancel_approval(self, approval_id: str, *, actor_id: str | None, reason: str):
        self.cancel_calls.append((approval_id, actor_id, reason))
        self.approval = replace(self.approval, status=ApprovalStatus.CANCELLED)
        return self.approval

class FakeConversationStore:
    def __init__(self) -> None:
        self.statuses = []

    async def update_assistant_request_status(self, command):
        self.statuses.append(command.status)


class FakeGateway:
    def __init__(self) -> None:
        self.calls = []

    async def invoke(self, request):
        self.calls.append(request)
        now = datetime.now(UTC)
        if request.approval_id is None:
            return ToolObservation.empty(
                tool_name=request.tool_name,
                status=ToolObservationStatus.APPROVAL_REQUIRED,
                sensitivity=request.sensitivity,
                started_at=now,
                completed_at=now,
                metadata={"approval_id": "approval-1"},
                error={"code": "approval_required", "message": "approval required"},
            )
        return ToolObservation.empty(
            tool_name=request.tool_name,
            status=ToolObservationStatus.COMPLETED,
            sensitivity=request.sensitivity,
            started_at=now,
            completed_at=now,
        )


def test_deterministic_datetime_now_response_requires_typed_observation_schema() -> None:
    request = replace(_loop_request(), user_input="который час?")
    raw_ref = ToolObservationRef(
        tool_call_id="tool-call-datetime",
        tool_name="datetime.now",
        status=ToolObservationStatus.COMPLETED,
        content='{"iso": "2026-06-05T20:59:07+03:00"}',
        content_type="application/json",
        sensitivity=Sensitivity.PROJECT,
    )

    assert deterministic_datetime_now_response(request, raw_ref) is None


def test_deterministic_datetime_now_response_uses_typed_observation_schema() -> None:
    request = replace(_loop_request(), user_input="который час?")
    typed_ref = ToolObservationRef(
        tool_call_id="tool-call-datetime",
        tool_name="datetime.now",
        status=ToolObservationStatus.COMPLETED,
        content='{"iso": "2026-06-05T20:59:07+03:00"}',
        content_type="application/json",
        sensitivity=Sensitivity.PROJECT,
        structured_schema="datetime.now",
        structured_content={"iso": "2026-06-05T20:59:07+03:00"},
        structured_schema_version=1,
        parse_status=ToolParseStatus.PARSED,
    )

    assert deterministic_datetime_now_response(request, typed_ref) == "Сейчас 20:59."


def test_approval_waiter_returns_after_granted_approval() -> None:
    async def scenario():
        store = FakeApprovalStore(_approval(ApprovalStatus.GRANTED))
        await ApprovalWaiter(store).wait(
            "approval-1",
            loop_deadline=asyncio.get_running_loop().time() + 1.0,
            actor_id="user-1",
        )
        return store.expire_calls

    assert asyncio.run(scenario()) == 1


def test_approval_waiter_cancels_pending_approval_when_wall_time_expires() -> None:
    async def scenario():
        store = FakeApprovalStore(_approval(ApprovalStatus.PENDING))
        with pytest.raises(RuntimeError, match="max_wall_time_exceeded"):
            await ApprovalWaiter(store).wait(
                "approval-1",
                loop_deadline=asyncio.get_running_loop().time(),
                actor_id="user-1",
            )
        return store.approval.status, store.cancel_calls

    status, cancel_calls = asyncio.run(scenario())

    assert status == ApprovalStatus.CANCELLED
    assert cancel_calls == [("approval-1", "user-1", "request timed out")]


def test_tool_proposal_executor_waits_for_approval_and_reinvokes_with_approval_id() -> None:
    async def scenario():
        gateway = FakeGateway()
        conversation_store = FakeConversationStore()
        approval_store = FakeApprovalStore(_approval(ApprovalStatus.GRANTED))
        executor = ToolProposalExecutor(
            tool_gateway=gateway,
            conversation_store=conversation_store,
            approval_waiter=ApprovalWaiter(approval_store),
        )

        observation = await executor.execute(
            _loop_request(),
            ToolProposal(action="tool_call", tool_name="fake.echo", arguments={"message": "hi"}),
            step_id="step-1",
            causation_event_id="event-step",
            used_tool_calls=0,
            loop_deadline=asyncio.get_running_loop().time() + 1.0,
        )
        return observation, gateway.calls, conversation_store.statuses

    observation, calls, statuses = asyncio.run(scenario())

    assert observation.status == ToolObservationStatus.COMPLETED
    assert [call.approval_id for call in calls] == [None, "approval-1"]
    assert statuses == [RequestStatus.WAITING_APPROVAL, RequestStatus.RUNNING]


def test_tool_proposal_executor_passes_request_working_directory_to_gateway() -> None:
    async def scenario():
        gateway = FakeGateway()
        executor = ToolProposalExecutor(
            tool_gateway=gateway,
            conversation_store=FakeConversationStore(),
        )

        await executor.execute(
            replace(_loop_request(), working_directory="/tmp/jarvis-project"),
            ToolProposal(action="tool_call", tool_name="fake.echo", arguments={"message": "hi"}),
            step_id="step-1",
            causation_event_id="event-step",
            used_tool_calls=0,
            loop_deadline=asyncio.get_running_loop().time() + 1.0,
        )
        return gateway.calls

    calls = asyncio.run(scenario())

    assert calls[0].working_directory == "/tmp/jarvis-project"


@pytest.mark.parametrize(
    ("unit_argument", "expected_unit"),
    [
        ("hours", "hours"),
        ("дней", "days"),
        ("день", "days"),
    ],
)
def test_tool_proposal_executor_canonicalizes_datetime_diff_arguments_for_gateway(
    unit_argument: str,
    expected_unit: str,
) -> None:
    async def scenario():
        gateway = FakeGateway()
        executor = ToolProposalExecutor(
            tool_gateway=gateway,
            conversation_store=FakeConversationStore(),
        )

        observation = await executor.execute(
            _loop_request(),
            ToolProposal(
                action="tool_call",
                tool_name="datetime.diff",
                arguments={
                    "from_iso": "1945-09-02",
                    "to_iso": "2026-06-08T00:12:00+03:00",
                    "unit": unit_argument,
                },
            ),
            step_id="step-1",
            causation_event_id="event-step",
            used_tool_calls=0,
            loop_deadline=asyncio.get_running_loop().time() + 1.0,
        )
        return observation, gateway.calls

    observation, calls = asyncio.run(scenario())

    assert calls[0].arguments == {
        "from_iso": "1945-09-02T00:00:00+03:00",
        "to_iso": "2026-06-08T00:12:00+03:00",
        "unit": expected_unit,
    }
    assert observation.arguments == calls[0].arguments


def test_tool_proposal_executor_canonicalizes_datetime_until_unit_for_gateway() -> None:
    async def scenario():
        gateway = FakeGateway()
        executor = ToolProposalExecutor(
            tool_gateway=gateway,
            conversation_store=FakeConversationStore(),
        )

        observation = await executor.execute(
            _loop_request(),
            ToolProposal(
                action="tool_call",
                tool_name="datetime.until",
                arguments={"target": "next_new_year", "unit": "дней"},
            ),
            step_id="step-1",
            causation_event_id="event-step",
            used_tool_calls=0,
            loop_deadline=asyncio.get_running_loop().time() + 1.0,
        )
        return observation, gateway.calls

    observation, calls = asyncio.run(scenario())

    assert calls[0].arguments == {"target": "next_new_year", "unit": "days"}
    assert observation.arguments == calls[0].arguments


def test_tool_proposal_executor_rejects_invalid_datetime_diff_arguments_before_gateway() -> None:
    async def scenario():
        gateway = FakeGateway()
        executor = ToolProposalExecutor(
            tool_gateway=gateway,
            conversation_store=FakeConversationStore(),
        )

        with pytest.raises(ToolProposalParseError):
            await executor.execute(
                _loop_request(),
                ToolProposal(
                    action="tool_call",
                    tool_name="datetime.diff",
                    arguments={
                        "from_iso": "917-11-07",
                        "to_iso": "2026-06-08T00:12:00+03:00",
                        "unit": "дней",
                    },
                ),
                step_id="step-1",
                causation_event_id="event-step",
                used_tool_calls=0,
                loop_deadline=asyncio.get_running_loop().time() + 1.0,
            )
        return gateway.calls

    calls = asyncio.run(scenario())

    assert calls == []
