from __future__ import annotations

from dataclasses import dataclass

from assistant_core.config.settings import Settings
from assistant_core.domain.conversations import ConversationMessage
from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    LoopStrategyName,
    UnknownLoopStrategy,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.context_assembler import ContextAssemblerPort
from assistant_core.ports.conversation_store import ConversationStorePort
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.model_router import ModelRouterPort
from assistant_core.runtime.loops import LoopStrategyRegistry, MemoryAugmentedAnswerLoop


@dataclass(frozen=True)
class RuntimeTurnCommand:
    request_id: str
    conversation_id: str
    user_message_id: str
    user_id: str
    user_input: str
    active_project_namespace: str | None
    current_message_sensitivity: Sensitivity = Sensitivity.PROJECT
    model_profile: str = "local_main"
    loop_strategy: str = LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value


@dataclass(frozen=True)
class RuntimeTurnResult:
    request_id: str
    response_text: str
    assistant_message: ConversationMessage | None
    model_calls: int
    degraded: bool


@dataclass(frozen=True)
class RuntimeStreamEvent:
    event_type: str
    data: dict


class AgentRuntime:
    def __init__(
        self,
        *,
        conversation_store: ConversationStorePort,
        context_assembler: ContextAssemblerPort,
        model_router: ModelRouterPort,
        event_log: EventLogPort,
        settings: Settings,
        loop_strategy_registry=None,
    ) -> None:
        self._settings = settings
        self._loop_strategy_registry = loop_strategy_registry or LoopStrategyRegistry(
            [
                MemoryAugmentedAnswerLoop(
                    conversation_store=conversation_store,
                    context_assembler=context_assembler,
                    model_router=model_router,
                    event_log=event_log,
                ),
            ],
        )

    async def run_turn(self, command: RuntimeTurnCommand) -> RuntimeTurnResult:
        request = self._loop_execution_request(command)
        strategy = self._loop_strategy_registry.get(request.strategy_name)
        result = await strategy.run_turn(request)
        return RuntimeTurnResult(
            request_id=command.request_id,
            response_text=result.response_text,
            assistant_message=result.assistant_message,
            model_calls=result.used_model_calls,
            degraded=result.degraded,
        )

    async def stream_turn(self, command: RuntimeTurnCommand):
        request = self._loop_execution_request(command)
        strategy = self._loop_strategy_registry.get(request.strategy_name)
        async for event in strategy.stream_turn(request):
            yield RuntimeStreamEvent(event.event_type, event.data)

    def _loop_execution_request(self, command: RuntimeTurnCommand) -> LoopExecutionRequest:
        try:
            strategy_name = LoopStrategyName(command.loop_strategy)
        except ValueError as exc:
            raise UnknownLoopStrategy(f"unknown loop strategy: {command.loop_strategy}") from exc
        budget_config = self._settings.runtime_budgets.get(strategy_name.value)
        if budget_config is None:
            raise UnknownLoopStrategy(f"missing runtime budget: {strategy_name.value}")
        return LoopExecutionRequest(
            request_id=command.request_id,
            conversation_id=command.conversation_id,
            user_message_id=command.user_message_id,
            user_id=command.user_id,
            user_input=command.user_input,
            active_project_namespace=command.active_project_namespace,
            current_message_sensitivity=command.current_message_sensitivity,
            model_profile=command.model_profile,
            strategy_name=strategy_name,
            budget=LoopBudget.from_runtime_budget(budget_config),
            correlation_id=command.request_id,
        )
