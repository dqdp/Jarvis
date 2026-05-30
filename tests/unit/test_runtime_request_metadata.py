from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import SelectionDecisionStatus
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.runtime.request_metadata import LoopSelectionError, runtime_request_metadata


pytestmark = pytest.mark.unit


def test_request_metadata_rejects_tool_loop_when_budget_disallows_tool_calls() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    budget = replace(
        settings.runtime_budgets[LoopStrategyName.TOOL_REACT_LOOP.value],
        max_tool_calls=0,
    )
    settings = replace(
        settings,
        runtime_budgets={
            **settings.runtime_budgets,
            LoopStrategyName.TOOL_REACT_LOOP.value: budget,
        },
    )

    with pytest.raises(LoopSelectionError) as exc_info:
        asyncio.run(_resolve_tool_metadata(settings))

    assert str(exc_info.value) == "tool loop is not executable by runtime budget"
    assert exc_info.value.decision is not None
    assert exc_info.value.decision.selected_loop_strategy is None
    assert exc_info.value.decision.reason_code == "selected_tool_loop_budget_unavailable"
    assert exc_info.value.decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE


def test_request_metadata_rejects_tool_loop_when_budget_is_missing() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    settings = replace(
        settings,
        runtime_budgets={
            name: budget
            for name, budget in settings.runtime_budgets.items()
            if name != LoopStrategyName.TOOL_REACT_LOOP.value
        },
    )

    with pytest.raises(LoopSelectionError) as exc_info:
        asyncio.run(_resolve_tool_metadata(settings))

    assert str(exc_info.value) == "loop strategy is not configured"
    assert exc_info.value.decision is not None
    assert exc_info.value.decision.selected_loop_strategy is None
    assert exc_info.value.decision.reason_code == "selected_loop_budget_unavailable"
    assert exc_info.value.decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE


async def _resolve_tool_metadata(settings):
    return await runtime_request_metadata(
        SimpleNamespace(
            content="show cpu usage",
            sensitivity=Sensitivity.PROJECT,
            loop_strategy="tools",
            model_profile=None,
            working_directory=str(Path.cwd()),
        ),
        settings,
        request_id="request-1",
        conversation_id="conversation-1",
        user_id="user-1",
        active_project_namespace="project.personal_assistant",
        working_directory=str(Path.cwd()),
        policy=ConfigPolicyEngine(settings),
    )
