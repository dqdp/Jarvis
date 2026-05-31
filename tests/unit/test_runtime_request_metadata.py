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


def _assert_direct_plan(
    metadata: dict,
    *,
    scenario: str,
    tool_names: list[str],
    capabilities: list[str],
    scope_hint: str | None = None,
    classification_source: str = "deterministic",
    required_arguments: dict[str, str] | None = None,
    provenance_contains: tuple[str, ...] = (),
) -> None:
    plan = metadata["loop_selection_direct_tool_plan"]
    assert plan["version"] == 1
    assert plan["scenario"] == scenario
    assert plan["tool_names"] == tool_names
    assert plan["capabilities"] == capabilities
    assert plan["scope_hint"] == scope_hint
    assert plan["classification_source"] == classification_source
    assert plan["required_arguments"] == (required_arguments or {})
    for evidence_code in provenance_contains:
        assert evidence_code in plan["provenance"]
    assert "loop_selection_direct_tool_name" not in metadata
    assert "loop_selection_direct_tool_names" not in metadata
    assert "loop_selection_direct_scenario" not in metadata


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="current_time",
        tool_names=["datetime.now"],
        capabilities=["tool.safe"],
        provenance_contains=("safe_builtin_request",),
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="christmas_countdown",
        tool_names=["datetime.now"],
        capabilities=["tool.safe"],
        scope_hint="christmas_countdown",
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="sensor_temperature",
        tool_names=["tool.system.read.sensors"],
        capabilities=["tool.system.read.sensors"],
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="memory_overview",
        tool_names=["tool.system.read.resources"],
        capabilities=["tool.system.read.resources"],
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="disk_free",
        tool_names=["tool.system.read.resources"],
        capabilities=["tool.system.read.resources"],
        scope_hint="disk_free",
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="battery_charge",
        tool_names=["tool.system.read.hardware"],
        capabilities=["tool.system.read.hardware"],
        scope_hint="battery_charge",
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="cpu_overview",
        tool_names=["tool.system.read.hardware", "tool.system.read.resources"],
        capabilities=["tool.system.read.hardware", "tool.system.read.resources"],
        scope_hint="cpu_overview",
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="os_version",
        tool_names=["tool.system.read.hardware"],
        capabilities=["tool.system.read.hardware"],
        scope_hint="os_version",
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="process_name_search",
        tool_names=["tool.system.read.process"],
        capabilities=["tool.system.read.process"],
        scope_hint="process_name_search",
        required_arguments={"process_pattern": "HFT"},
        provenance_contains=("process_name_search_pattern",),
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="vpn_status",
        tool_names=["tool.system.read.network"],
        capabilities=["tool.system.read.network"],
        scope_hint="vpn_status",
    )


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
    _assert_direct_plan(
        resolution.metadata,
        scenario="os_version",
        tool_names=["tool.system.read.hardware"],
        capabilities=["tool.system.read.hardware"],
        scope_hint="os_version",
    )


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
    assert "loop_selection_direct_tool_plan" not in resolution.metadata


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
    assert "loop_selection_direct_tool_plan" not in resolution.metadata


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
