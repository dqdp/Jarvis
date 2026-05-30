from __future__ import annotations

import asyncio

from assistant_core.domain.conversations import UpdateAssistantRequestStatusCommand
from assistant_core.domain.loops import LoopExecutionRequest, ToolObservationRef, ToolProposal, ToolProposalParseError
from assistant_core.domain.requests import RequestStatus
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus


class ToolProposalExecutor:
    def __init__(self, *, tool_gateway, conversation_store, approval_waiter=None) -> None:
        self._tool_gateway = tool_gateway
        self._conversation_store = conversation_store
        self._approval_waiter = approval_waiter

    async def execute(
        self,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        *,
        step_id: str,
        causation_event_id: str,
        used_tool_calls: int,
        loop_deadline: float,
    ) -> ToolObservationRef:
        if proposal.tool_name is None:
            raise ToolProposalParseError("tool_call requires tool_name")
        _ensure_tool_budget(used_tool_calls=used_tool_calls, request=request)
        observation = await self._invoke_gateway(
            request,
            proposal,
            step_id=step_id,
            causation_event_id=causation_event_id,
            approval_id=None,
            loop_deadline=loop_deadline,
        )
        if observation.status != ToolObservationStatus.APPROVAL_REQUIRED:
            return ToolObservationRef.from_observation(observation)

        approval_id = observation.metadata.get("approval_id")
        if approval_id is None or self._approval_waiter is None:
            return ToolObservationRef.from_observation(observation)

        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.WAITING_APPROVAL,
            ),
        )
        await self._approval_waiter.wait(
            approval_id,
            loop_deadline=loop_deadline,
            actor_id=request.user_id,
        )
        await self._conversation_store.update_assistant_request_status(
            UpdateAssistantRequestStatusCommand(
                request_id=request.request_id,
                status=RequestStatus.RUNNING,
            ),
        )
        observation = await self._invoke_gateway(
            request,
            proposal,
            step_id=step_id,
            causation_event_id=causation_event_id,
            approval_id=approval_id,
            loop_deadline=loop_deadline,
        )
        return ToolObservationRef.from_observation(observation)

    async def _invoke_gateway(
        self,
        request: LoopExecutionRequest,
        proposal: ToolProposal,
        *,
        step_id: str,
        causation_event_id: str,
        approval_id: str | None,
        loop_deadline: float,
    ):
        assert proposal.tool_name is not None
        return await self._tool_gateway.invoke(
            ToolCallRequest(
                tool_name=proposal.tool_name,
                arguments=proposal.arguments,
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                correlation_id=request.correlation_id or request.request_id,
                causation_event_id=causation_event_id,
                step_id=step_id,
                user_id=request.user_id,
                project_namespace=request.active_project_namespace,
                working_directory=request.working_directory,
                sensitivity=request.current_message_sensitivity,
                permission_mode=request.permission_mode,
                approval_id=approval_id,
                timeout_seconds=_remaining_timeout(
                    loop_deadline,
                    request.budget.max_model_call_seconds,
                ),
                metadata={"loop_strategy": request.strategy_name.value},
            ),
        )


def _ensure_tool_budget(*, used_tool_calls: int, request: LoopExecutionRequest) -> None:
    if used_tool_calls >= request.budget.max_tool_calls:
        raise RuntimeError("max_tool_calls_exceeded")


def _remaining_timeout(deadline: float, operation_timeout: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise RuntimeError("max_wall_time_exceeded")
    return min(float(operation_timeout), remaining)
