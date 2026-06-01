from __future__ import annotations

import asyncio

from assistant_core.domain.context import ContextAssemblyRequest
from assistant_core.domain.conversations import CompleteAssistantResponseCommand
from assistant_core.domain.events import EventEnvelope, EventType
from assistant_core.domain.loops import (
    AgentLoopState,
    AgentLoopStep,
    LoopExecutionRequest,
    LoopExecutionResult,
    LoopStatus,
    ToolObservationRef,
)
from assistant_core.domain.models import ChatModelRequest
from assistant_core.ports.context_assembler import ContextAssemblerPort
from assistant_core.ports.conversation_store import ConversationStorePort
from assistant_core.ports.model_router import ModelRouterPort
from assistant_core.runtime.loops.event_recorder import LoopEventRecorder


class FinalAnswerStepError(Exception):
    def __init__(self, cause: Exception, *, used_model_calls: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.used_model_calls = used_model_calls


class FinalAnswerStep:
    def __init__(
        self,
        *,
        conversation_store: ConversationStorePort,
        context_assembler: ContextAssemblerPort,
        model_router: ModelRouterPort,
        event_recorder: LoopEventRecorder,
    ) -> None:
        self._conversation_store = conversation_store
        self._context_assembler = context_assembler
        self._model_router = model_router
        self._event_recorder = event_recorder

    async def run(
        self,
        request: LoopExecutionRequest,
        *,
        step_started: EventEnvelope,
        used_model_calls: int,
        used_tool_calls: int,
        context_manifest_refs: list[str],
        tool_observation_refs: list[ToolObservationRef],
        loop_deadline: float,
        output_contract: str | None = None,
    ) -> LoopExecutionResult:
        if used_model_calls >= request.budget.max_model_calls:
            raise RuntimeError("max_model_calls_exceeded")
        step_id = step_started.payload["step_id"]
        step_index = step_started.payload["step_index"]
        final_context_started = await self._event_recorder.append(
            EventType.CONTEXT_ASSEMBLY_STARTED,
            request,
            payload={
                "step_id": step_id,
                "step_index": step_index,
                "purpose": "final_answer",
            },
            causation_id=step_started.event_id,
            state=AgentLoopState.CONTEXT_ASSEMBLING,
            step=AgentLoopStep.FINAL,
        )
        final_context = await asyncio.wait_for(
            self._context_assembler.assemble(
                ContextAssemblyRequest(
                    request_id=request.request_id,
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    current_user_message=request.user_input,
                    active_project_namespace=request.active_project_namespace,
                    loop_strategy=request.strategy_name.value,
                    model_profile=request.model_profile,
                    current_message_sensitivity=request.current_message_sensitivity,
                    current_user_message_id=request.user_message_id,
                    causation_event_id=final_context_started.event_id,
                    purpose="final_answer",
                    permission_mode=request.permission_mode,
                    tool_observation_refs=tuple(tool_observation_refs),
                    output_contract=output_contract,
                ),
            ),
            timeout=_remaining_timeout(
                loop_deadline,
                request.budget.max_context_assembly_seconds,
            ),
        )
        context_manifest_refs.append(final_context.manifest.context_manifest_id)
        model_started = await self._event_recorder.append(
            EventType.MODEL_REQUEST_CREATED,
            request,
            payload={"context_manifest_id": final_context.manifest.context_manifest_id},
            causation_id=final_context_started.event_id,
            sensitivity=final_context.manifest.max_sensitivity,
            state=AgentLoopState.FINALIZING,
            step=AgentLoopStep.FINAL,
        )
        used_model_calls += 1
        try:
            chat_response = await asyncio.wait_for(
                self._model_router.chat(
                    ChatModelRequest(
                        profile=request.model_profile,
                        messages=final_context.messages,
                        sensitivity=final_context.manifest.max_sensitivity,
                        request_id=request.request_id,
                        conversation_id=request.conversation_id,
                        context_manifest_id=final_context.manifest.context_manifest_id,
                    ),
                ),
                timeout=_remaining_timeout(
                    loop_deadline,
                    request.budget.max_model_call_seconds,
                ),
            )
            model_received = await self._event_recorder.append(
                EventType.MODEL_RESPONSE_RECEIVED,
                request,
                payload={"context_manifest_id": final_context.manifest.context_manifest_id},
                causation_id=model_started.event_id,
                sensitivity=final_context.manifest.max_sensitivity,
                state=AgentLoopState.FINALIZING,
                step=AgentLoopStep.FINAL,
            )
            completion = await self._conversation_store.complete_assistant_response(
                CompleteAssistantResponseCommand(
                    conversation_id=request.conversation_id,
                    request_id=request.request_id,
                    content=chat_response.text,
                    sensitivity=final_context.manifest.max_sensitivity,
                ),
            )
            assistant_event = await self._event_recorder.append(
                EventType.ASSISTANT_MESSAGE_CREATED,
                request,
                payload={
                    "message_id": completion.message.message_id,
                    "content_hash": completion.message.content_hash,
                },
                causation_id=model_received.event_id,
                sensitivity=final_context.manifest.max_sensitivity,
                state=AgentLoopState.FINALIZING,
                step=AgentLoopStep.FINAL,
            )
            await self._event_recorder.append(
                EventType.AGENT_STEP_COMPLETED,
                request,
                payload={
                    "strategy_name": request.strategy_name.value,
                    "step_id": step_id,
                    "step_index": step_index,
                    "action": "final_answer",
                },
                causation_id=step_started.event_id,
                sensitivity=final_context.manifest.max_sensitivity,
                state=AgentLoopState.FINALIZING,
                step=AgentLoopStep.FINAL,
            )
            loop_completed = await self._event_recorder.append(
                EventType.AGENT_LOOP_COMPLETED,
                request,
                payload={
                    "strategy_name": request.strategy_name.value,
                    "status": LoopStatus.COMPLETED.value,
                    "used_model_calls": used_model_calls,
                    "used_tool_calls": used_tool_calls,
                    "context_manifest_refs": list(context_manifest_refs),
                    "tool_observation_refs": [
                        ref.tool_call_id for ref in tool_observation_refs
                    ],
                },
                causation_id=assistant_event.event_id,
                sensitivity=final_context.manifest.max_sensitivity,
                state=AgentLoopState.COMPLETED,
                step=AgentLoopStep.COMPLETED,
            )
            await self._event_recorder.append(
                EventType.REQUEST_PROCESSING_COMPLETED,
                request,
                payload={"assistant_message_id": completion.message.message_id},
                causation_id=loop_completed.event_id,
                sensitivity=final_context.manifest.max_sensitivity,
                state=AgentLoopState.COMPLETED,
                step=AgentLoopStep.COMPLETED,
            )
        except TimeoutError as exc:
            if _wall_time_expired(loop_deadline):
                raise FinalAnswerStepError(
                    RuntimeError("max_wall_time_exceeded"),
                    used_model_calls=used_model_calls,
                ) from exc
            raise FinalAnswerStepError(exc, used_model_calls=used_model_calls) from exc
        except Exception as exc:
            raise FinalAnswerStepError(exc, used_model_calls=used_model_calls) from exc
        return LoopExecutionResult(
            status=LoopStatus.COMPLETED,
            response_text=chat_response.text,
            assistant_message=completion.message,
            used_model_calls=used_model_calls,
            used_tool_calls=used_tool_calls,
            context_manifest_refs=tuple(context_manifest_refs),
            tool_observation_refs=tuple(tool_observation_refs),
            degraded=False,
        )

    async def complete_deterministic(
        self,
        request: LoopExecutionRequest,
        *,
        step_started: EventEnvelope,
        response_text: str,
        used_model_calls: int,
        used_tool_calls: int,
        context_manifest_refs: list[str],
        tool_observation_refs: list[ToolObservationRef],
    ) -> LoopExecutionResult:
        completion = await self._conversation_store.complete_assistant_response(
            CompleteAssistantResponseCommand(
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                content=response_text,
                sensitivity=request.current_message_sensitivity,
            ),
        )
        assistant_event = await self._event_recorder.append(
            EventType.ASSISTANT_MESSAGE_CREATED,
            request,
            payload={
                "message_id": completion.message.message_id,
                "content_hash": completion.message.content_hash,
            },
            causation_id=step_started.event_id,
            sensitivity=request.current_message_sensitivity,
            state=AgentLoopState.FINALIZING,
            step=AgentLoopStep.FINAL,
        )
        await self._event_recorder.append(
            EventType.AGENT_STEP_COMPLETED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "step_id": step_started.payload["step_id"],
                "step_index": step_started.payload["step_index"],
                "action": "final_answer",
                "source": "deterministic_recovery",
            },
            causation_id=step_started.event_id,
            sensitivity=request.current_message_sensitivity,
            state=AgentLoopState.FINALIZING,
            step=AgentLoopStep.FINAL,
        )
        loop_completed = await self._event_recorder.append(
            EventType.AGENT_LOOP_COMPLETED,
            request,
            payload={
                "strategy_name": request.strategy_name.value,
                "status": LoopStatus.COMPLETED.value,
                "used_model_calls": used_model_calls,
                "used_tool_calls": used_tool_calls,
                "context_manifest_refs": list(context_manifest_refs),
                "tool_observation_refs": [ref.tool_call_id for ref in tool_observation_refs],
                "source": "deterministic_recovery",
            },
            causation_id=assistant_event.event_id,
            sensitivity=request.current_message_sensitivity,
            state=AgentLoopState.COMPLETED,
            step=AgentLoopStep.COMPLETED,
        )
        await self._event_recorder.append(
            EventType.REQUEST_PROCESSING_COMPLETED,
            request,
            payload={"assistant_message_id": completion.message.message_id},
            causation_id=loop_completed.event_id,
            sensitivity=request.current_message_sensitivity,
            state=AgentLoopState.COMPLETED,
            step=AgentLoopStep.COMPLETED,
        )
        return LoopExecutionResult(
            status=LoopStatus.COMPLETED,
            response_text=response_text,
            assistant_message=completion.message,
            used_model_calls=used_model_calls,
            used_tool_calls=used_tool_calls,
            context_manifest_refs=tuple(context_manifest_refs),
            tool_observation_refs=tuple(tool_observation_refs),
            degraded=True,
        )


def _remaining_timeout(deadline: float, operation_timeout: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise RuntimeError("max_wall_time_exceeded")
    return min(float(operation_timeout), remaining)


def _wall_time_expired(deadline: float) -> bool:
    return asyncio.get_running_loop().time() >= deadline
