from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    LoopExecutionResult,
    LoopStatus,
    LoopStrategyName,
    UnknownLoopStrategy,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.loops import LoopStrategyRegistry, MemoryAugmentedAnswerLoop


pytestmark = pytest.mark.unit


class StubStrategy:
    strategy_name = LoopStrategyName.MEMORY_AUGMENTED_ANSWER


def _budget() -> LoopBudget:
    return LoopBudget(
        max_steps=1,
        max_model_calls=1,
        max_tool_calls=0,
        max_wall_time_seconds=180,
        max_context_assembly_seconds=10,
        max_model_call_seconds=120,
        max_consecutive_failures=1,
    )


def test_strategy_registry_selects_memory_augmented_answer_by_default() -> None:
    strategy = StubStrategy()
    registry = LoopStrategyRegistry([strategy])

    assert registry.default_strategy() is strategy
    assert registry.get(LoopStrategyName.MEMORY_AUGMENTED_ANSWER) is strategy


def test_unknown_strategy_is_rejected() -> None:
    registry = LoopStrategyRegistry([StubStrategy()])

    with pytest.raises(UnknownLoopStrategy):
        registry.get("missing_strategy")


def test_memory_augmented_answer_budget_keeps_max_tool_calls_zero() -> None:
    settings = ConfigLoader("config").load("test")
    budget = LoopBudget.from_runtime_budget(
        settings.runtime_budgets[LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value],
    )

    assert budget.max_model_calls == 1
    assert budget.max_tool_calls == 0


def test_memory_augmented_answer_does_not_require_tool_gateway() -> None:
    parameters = inspect.signature(MemoryAugmentedAnswerLoop).parameters

    assert "tool_gateway" not in parameters


def test_loop_execution_request_requires_strategy_name_and_budget() -> None:
    base = dict(
        request_id="request-1",
        conversation_id="conversation-1",
        user_message_id="message-1",
        user_id="user-1",
        user_input="hello",
        active_project_namespace="project.personal_assistant",
        current_message_sensitivity=Sensitivity.PROJECT,
        model_profile="local_main",
        strategy_name=LoopStrategyName.MEMORY_AUGMENTED_ANSWER,
        budget=_budget(),
    )

    assert LoopExecutionRequest(**base).strategy_name == LoopStrategyName.MEMORY_AUGMENTED_ANSWER

    with pytest.raises(ValueError):
        LoopExecutionRequest(**{**base, "strategy_name": ""})
    with pytest.raises(ValueError):
        LoopExecutionRequest(**{**base, "budget": None})


def test_loop_execution_result_reports_model_and_tool_call_counts() -> None:
    result = LoopExecutionResult(
        status=LoopStatus.COMPLETED,
        response_text="answer",
        assistant_message=None,
        used_model_calls=1,
        used_tool_calls=0,
        context_manifest_refs=("manifest-1",),
        degraded=False,
    )

    assert result.used_model_calls == 1
    assert result.used_tool_calls == 0
    assert result.context_manifest_refs == ("manifest-1",)

    with pytest.raises(ValueError):
        replace(result, used_model_calls=-1)
    with pytest.raises(ValueError):
        replace(result, used_tool_calls=-1)
