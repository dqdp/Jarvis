from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
    SelectionDecisionStatus,
    SelectionFallbackPreference,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import Capability, RiskClass
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


def test_request_metadata_records_direct_safe_builtin_tool_hint() -> None:
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["datetime.now"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "datetime.now"


def test_request_metadata_records_direct_christmas_countdown_scenario() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="через сколько дней Рождество?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["datetime.now"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "datetime.now"
    assert resolution.metadata["loop_selection_direct_scenario"] == "christmas_countdown"


def test_request_metadata_records_direct_system_sensors_tool_hint() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Текущая температура процессора.",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.sensors"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.sensors"


def test_request_metadata_records_direct_system_resources_tool_hint_for_free_memory() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько памяти сейчас свободно в системе?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.resources"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.resources"


def test_request_metadata_records_direct_disk_free_scenario() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько свободного места на диске?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.resources"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.resources"
    assert resolution.metadata["loop_selection_direct_scenario"] == "disk_free"


def test_request_metadata_records_direct_battery_charge_scenario() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько процентов заряда аккумулятора осталось на макбуке?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.hardware"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.hardware"
    assert resolution.metadata["loop_selection_direct_scenario"] == "battery_charge"


def test_request_metadata_records_direct_cpu_overview_tool_plan() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько ядер у центрального процессора и на сколько они загружены?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == [
        "tool.system.read.hardware",
        "tool.system.read.resources",
    ]
    assert resolution.metadata["loop_selection_direct_tool_names"] == [
        "tool.system.read.hardware",
        "tool.system.read.resources",
    ]
    assert "loop_selection_direct_tool_name" not in resolution.metadata


def test_request_metadata_records_direct_os_version_scenario() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Какая версия операционной системы?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.hardware"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.hardware"
    assert resolution.metadata["loop_selection_direct_scenario"] == "os_version"


def test_request_metadata_records_direct_process_name_search_scenario() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content='Запущен ли сейчас процесс, в имени которого есть "HFT"?',
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.process"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.process"
    assert resolution.metadata["loop_selection_direct_scenario"] == "process_name_search"


def test_request_metadata_records_direct_vpn_status_scenario() -> None:
    settings = ConfigLoader(Path("config")).load("test")

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Включен ли VPN сейчас?",
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

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.network"]
    assert resolution.metadata["loop_selection_direct_tool_name"] == "tool.system.read.network"
    assert resolution.metadata["loop_selection_direct_scenario"] == "vpn_status"


def test_request_metadata_accepts_injected_intent_classifier() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    classifier = StaticIntentClassifier(
        IntentClassification(
            intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.88,
            candidate_capabilities=(
                CapabilityCandidate(
                    capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                    intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                    confidence=0.88,
                    requires_live_state=True,
                    requires_execution=True,
                    requires_write=False,
                    tool_names=("tool.system.read.hardware",),
                    risk_classes=frozenset({RiskClass.READ_ONLY}),
                    scope_hint="os_version",
                    evidence_codes=("model_os_version_request",),
                ),
            ),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
            reason_code="model_system_diagnostics",
            classification_source="deterministic",
            fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        )
    )

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Покажи данные о системе",
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
            intent_classifier=classifier,
        )
    )

    assert classifier.calls == 1
    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.hardware"]
    assert resolution.metadata["loop_selection_direct_scenario"] == "os_version"


def test_request_metadata_does_not_direct_execute_model_source_tool_hints() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    classifier = StaticIntentClassifier(
        IntentClassification(
            intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.91,
            candidate_capabilities=(
                CapabilityCandidate(
                    capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                    intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                    confidence=0.91,
                    requires_live_state=True,
                    requires_execution=True,
                    requires_write=False,
                    tool_names=("tool.system.read.resources",),
                    risk_classes=frozenset({RiskClass.READ_ONLY}),
                    scope_hint="disk_free",
                    evidence_codes=("model_disk_free_request",),
                ),
            ),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
            reason_code="model_system_diagnostics",
            classification_source="model",
            fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        )
    )

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Сколько свободного места на диске?",
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
            intent_classifier=classifier,
        )
    )

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert resolution.metadata["loop_selection_classification_source"] == "model"
    assert resolution.metadata["loop_selection_tool_names"] == ["tool.system.read.resources"]
    assert "loop_selection_direct_scenario" not in resolution.metadata
    assert "loop_selection_direct_tool_name" not in resolution.metadata


def test_request_metadata_rejects_mismatched_direct_tool_hint() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    classifier = StaticIntentClassifier(
        IntentClassification(
            intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=0.88,
            candidate_capabilities=(
                CapabilityCandidate(
                    capability=Capability.TOOL_SYSTEM_READ_NETWORK,
                    intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                    confidence=0.88,
                    requires_live_state=True,
                    requires_execution=True,
                    requires_write=False,
                    tool_names=("tool.system.read.hardware",),
                    risk_classes=frozenset({RiskClass.READ_ONLY}),
                    scope_hint="vpn_status",
                    evidence_codes=("bad_tool_hint",),
                ),
            ),
            requires_live_state=True,
            requires_execution=True,
            answer_without_tools_would_be_misleading=True,
            reason_code="system_diagnostics_hint",
            fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        )
    )

    resolution = asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="Включен ли VPN сейчас?",
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
            intent_classifier=classifier,
        )
    )

    assert resolution.metadata["selected_loop_strategy"] == "tool_react_loop"
    assert "loop_selection_tool_names" not in resolution.metadata
    assert "loop_selection_direct_scenario" not in resolution.metadata
    assert "loop_selection_direct_tool_name" not in resolution.metadata


def test_request_metadata_provides_tool_routing_summary_to_classifier() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    classifier = RecordingIntentClassifier(
        IntentClassification(
            intent_family=IntentFamily.ORDINARY_CHAT,
            confidence=0.9,
            reason_code="ordinary_chat",
            fallback_preference=SelectionFallbackPreference.CHAT,
        )
    )

    asyncio.run(
        runtime_request_metadata(
            SimpleNamespace(
                content="hello",
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
            intent_classifier=classifier,
        )
    )

    assert classifier.requests
    tool_names = {
        item["tool_name"]
        for item in classifier.requests[0].available_tools_summary
        if isinstance(item, dict)
    }
    assert "datetime.now" in tool_names
    assert "tool.system.read.hardware" in tool_names
    assert "tool.system.read.resources" in tool_names


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


class StaticIntentClassifier:
    def __init__(self, classification: IntentClassification) -> None:
        self.classification = classification
        self.calls = 0

    async def classify(self, request):
        self.calls += 1
        return self.classification


class RecordingIntentClassifier(StaticIntentClassifier):
    def __init__(self, classification: IntentClassification) -> None:
        super().__init__(classification)
        self.requests = []

    async def classify(self, request):
        self.requests.append(request)
        return await super().classify(request)
