from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from assistant_core.domain.approvals import ApprovalRequest, ApprovalScope, ApprovalStatus
from assistant_core.domain.loops import LoopBudget, LoopExecutionRequest, LoopStrategyName, ToolProposal
from assistant_core.domain.policy import Capability, PermissionMode, RiskClass
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolObservation, ToolObservationStatus
from assistant_core.runtime.loops.tool_approval import ApprovalWaiter
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
