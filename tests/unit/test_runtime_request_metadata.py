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
from assistant_core.runtime.request_metadata import (
    AgentToolPolicy,
    LoopSelectionError,
    emit_loop_selection_failure,
    runtime_request_metadata,
)


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


def test_agent_request_plan_records_chat_mode_with_tools_disabled() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Расскажи, как решаются кубические уравнения.",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy="chat",
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
    )

    plan = resolution.agent_request_plan
    assert plan.requested_mode == "chat"
    assert plan.selected_loop_strategy == "tool_react_loop"
    assert plan.tool_policy is AgentToolPolicy.DISABLED
    assert plan.allowed_tool_names == ()
    assert plan.redacted_metadata()["agent_tool_policy"] == "disabled"
    assert resolution.metadata["agent_tool_policy"] == "disabled"
    assert resolution.metadata["agent_allowed_tool_count"] == 0


def test_agent_request_plan_records_tools_mode_with_required_tools() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(_resolve_tool_metadata(settings))

    plan = resolution.agent_request_plan
    assert plan.requested_mode == "tools"
    assert plan.selected_loop_strategy == "tool_react_loop"
    assert plan.tool_policy is AgentToolPolicy.REQUIRED
    assert plan.allowed_tool_names
    assert resolution.metadata["agent_tool_policy"] == "required"
    assert resolution.metadata["agent_allowed_tool_count"] == len(plan.allowed_tool_names)


def test_agent_request_plan_disables_tools_when_policy_disables_tools() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    settings = replace(settings, policy=replace(settings.policy, tools_enabled=False))

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Привет",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    assert resolution.agent_request_plan.tool_policy is AgentToolPolicy.DISABLED
    assert resolution.agent_request_plan.allowed_tool_names == ()
    assert resolution.metadata["agent_tool_policy"] == "disabled"
    assert resolution.metadata["agent_allowed_tool_count"] == 0
    assert "agent_allowed_tool_names" not in resolution.metadata


def test_agent_request_plan_disables_tools_when_tool_budget_cannot_execute_tools() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    budget = replace(
        settings.runtime_budgets[LoopStrategyName.TOOL_REACT_LOOP.value],
        allow_tools=False,
        max_tool_calls=0,
    )
    settings = replace(
        settings,
        runtime_budgets={
            **settings.runtime_budgets,
            LoopStrategyName.TOOL_REACT_LOOP.value: budget,
        },
    )

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Привет",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    assert resolution.agent_request_plan.tool_policy is AgentToolPolicy.DISABLED
    assert resolution.agent_request_plan.allowed_tool_names == ()
    assert resolution.metadata["agent_tool_policy"] == "disabled"
    assert resolution.metadata["agent_allowed_tool_count"] == 0


def test_agent_request_plan_filters_allowed_tools_through_request_policy() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Привет",
                sensitivity=Sensitivity.SECRET,
                loop_strategy=None,
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
    )

    assert resolution.agent_request_plan.tool_policy is AgentToolPolicy.DISABLED
    assert resolution.agent_request_plan.allowed_tool_names == ()
    assert resolution.metadata["agent_tool_policy"] == "disabled"
    assert resolution.metadata["agent_allowed_tool_count"] == 0
    assert "agent_allowed_tool_names" not in resolution.metadata


def test_agent_request_plan_filters_allowed_tools_by_sensitivity_ceiling() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Проверь локальную систему",
                sensitivity=Sensitivity.PERSONAL,
                loop_strategy=None,
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
    )

    allowed = set(resolution.agent_request_plan.allowed_tool_names)
    assert "calculator.evaluate" not in allowed
    assert "calendar.diff" not in allowed
    assert "daemon.status" not in allowed
    assert "datetime.diff" not in allowed
    assert "datetime.now" not in allowed
    assert "datetime.until" not in allowed
    assert "tool.shell.read.project" not in allowed
    assert "tool.system.read.resources" in allowed


def test_agent_request_plan_uses_request_plan_reason_namespace() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько время?",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    assert resolution.decision.reason_code == "request_plan_auto_agent_loop"
    assert resolution.agent_request_plan.request_plan_reason_code == "request_plan_auto_agent_loop"
    assert resolution.metadata["request_plan_reason_code"] == "request_plan_auto_agent_loop"
    assert not resolution.agent_request_plan.request_plan_reason_code.startswith("tool_intent_")
    assert not resolution.agent_request_plan.request_plan_reason_code.startswith("classifier_")


def test_agent_request_plan_chat_reason_does_not_copy_classifier_reason() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Проверь состояние системы",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy="chat",
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
    )

    assert resolution.decision.reason_code == "request_plan_chat_agent_loop"
    assert resolution.agent_request_plan.request_plan_reason_code == "request_plan_chat_agent_loop"
    assert resolution.metadata["request_plan_reason_code"] == "request_plan_chat_agent_loop"


def test_agent_request_plan_metadata_excludes_classifier_fields() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько время?",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    plan_metadata = resolution.agent_request_plan.redacted_metadata()
    assert plan_metadata["requested_loop_mode"] == "auto"
    assert plan_metadata["request_plan_status"] == "selected"
    assert plan_metadata["request_plan_reason_code"]
    assert "agent_tool_policy" in plan_metadata
    assert "intent_family" not in plan_metadata
    assert "loop_selection_intent_family" not in plan_metadata
    assert "loop_selection_confidence" not in plan_metadata
    assert "loop_selection_classification_source" not in plan_metadata
    assert "direct_tool_plan" not in plan_metadata
    assert "loop_selection_direct_tool_plan" not in plan_metadata
    assert "loop_selection_intent_family" not in resolution.metadata
    assert "loop_selection_confidence" not in resolution.metadata
    assert "loop_selection_classification_source" not in resolution.metadata
    assert "loop_selection_direct_tool_plan" not in resolution.metadata


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


def test_model_profile_selection_failure_event_preserves_specific_reason() -> None:
    class FakeEventLog:
        def __init__(self) -> None:
            self.events = []

        async def append(self, event):
            self.events.append(event)
            return event

    async def scenario():
        settings = ConfigLoader(Path("config")).load("test")
        try:
            await runtime_request_metadata(
                SimpleNamespace(
                    content="hello",
                    sensitivity=Sensitivity.PROJECT,
                    loop_strategy=None,
                    model_profile="local_embedding",
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
        except LoopSelectionError as exc:
            event_log = FakeEventLog()
            await emit_loop_selection_failure(event_log, exc)
            return exc, event_log.events
        raise AssertionError("model profile selection must fail")

    error, events = asyncio.run(scenario())

    assert error.decision is not None
    assert error.decision.reason_code == "model_profile_invalid_for_selected_loop"
    failed = next(
        event for event in events if event.event_type.value == "request.loop_selection.failed"
    )
    assert (
        failed.payload["request_plan_reason_code"]
        == "request_plan_model_profile_invalid_for_selected_loop"
    )


def test_default_auto_request_builds_agent_loop_plan_without_classifier() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько время?",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    assert resolution.metadata["requested_loop_mode"] == "auto"
    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["selected_model_profile"] == "local_main"
    assert resolution.metadata["model_profile"] == "local_main"
    assert resolution.agent_request_plan.tool_policy is AgentToolPolicy.AVAILABLE
    assert "calendar.diff" in resolution.metadata["agent_live_state_tool_names"]
    assert "datetime.diff" in resolution.metadata["agent_live_state_tool_names"]
    assert "datetime.now" in resolution.metadata["agent_live_state_tool_names"]
    assert "datetime.until" in resolution.metadata["agent_live_state_tool_names"]
    assert "tool.system.read.resources" in resolution.metadata["agent_live_state_tool_names"]
    assert "calculator.evaluate" not in resolution.metadata["agent_live_state_tool_names"]
    assert resolution.agent_request_plan.live_state_tool_names == tuple(
        resolution.metadata["agent_live_state_tool_names"],
    )
    summaries = resolution.metadata["agent_allowed_tool_summaries"]
    resources = next(
        item for item in summaries if item["tool_name"] == "tool.system.read.resources"
    )
    hardware = next(
        item for item in summaries if item["tool_name"] == "tool.system.read.hardware"
    )
    calendar_diff = next(item for item in summaries if item["tool_name"] == "calendar.diff")
    datetime_diff = next(item for item in summaries if item["tool_name"] == "datetime.diff")
    assert "CPU load" in resources["description"]
    assert "memory usage" in resources["description"]
    assert "not live CPU load" in hardware["description"]
    assert "two known timezone-aware ISO timestamps" in calendar_diff["description"]
    assert "does not resolve event names or holidays" in calendar_diff["description"]
    assert "microseconds through weeks" in datetime_diff["description"]
    assert "does not resolve event names or holidays" in datetime_diff["description"]
    assert "loop_selection_confidence" not in resolution.metadata
    assert "loop_selection_intent_family" not in resolution.metadata
    assert "loop_selection_classification_source" not in resolution.metadata
    assert "loop_selection_direct_tool_plan" not in resolution.metadata
    assert "loop_selection_tool_names" not in resolution.metadata


def test_natural_language_calculator_request_has_no_direct_plan() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько будет четырнадцать умножить на сорок восемь?",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    assert resolution.metadata["requested_loop_mode"] == "auto"
    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["agent_tool_policy"] == "available"
    assert "calculator.evaluate" in resolution.agent_request_plan.allowed_tool_names
    assert "calculator.evaluate" in resolution.metadata["agent_allowed_tool_names"]
    assert "loop_selection_direct_tool_plan" not in resolution.metadata
    assert "direct_tool_plan" not in resolution.agent_request_plan.redacted_metadata()


def test_calendar_event_request_has_no_pre_router_guess() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Когда в 2026 году будет День благодарения?",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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
    )

    assert resolution.metadata["requested_loop_mode"] == "auto"
    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["agent_tool_policy"] == "available"
    assert "loop_selection_tool_names" not in resolution.metadata
    assert "loop_selection_direct_tool_plan" not in resolution.metadata


def test_voice_transcript_uses_same_agent_request_plan_shape() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    async def resolve(content: str):
        return await runtime_request_metadata(
            SimpleNamespace(
                content=content,
                input_channel="voice_transcript",
                transcript_id="transcript-1",
                sensitivity=Sensitivity.PROJECT,
                loop_strategy=None,
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

    typed = asyncio.run(resolve("Сколько времени?"))
    transcript = asyncio.run(resolve("Джарвис, сколько времени?"))

    assert transcript.metadata["requested_loop_mode"] == typed.metadata["requested_loop_mode"]
    assert transcript.metadata["selected_loop_strategy"] == typed.metadata["selected_loop_strategy"]
    assert transcript.metadata["loop_strategy"] == "tool_react_loop"
    assert transcript.metadata["model_profile"] == "local_main"
    assert transcript.agent_request_plan.tool_policy is AgentToolPolicy.AVAILABLE
    assert transcript.agent_request_plan.allowed_tool_names == typed.agent_request_plan.allowed_tool_names
    assert "loop_selection_direct_tool_plan" not in transcript.metadata
    assert "loop_selection_classification_source" not in transcript.metadata


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
