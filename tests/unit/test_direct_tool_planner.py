from __future__ import annotations

from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentFamily,
    LoopSelectionDecision,
    LoopSelectionMode,
    SelectionDecisionStatus,
    SelectionFallbackPreference,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import Capability, PolicyDecisionOutcome, RiskClass
from assistant_core.runtime.direct_tools import DirectToolPlanner, direct_tool_plan_from_metadata
from assistant_core.runtime.routing import CapabilityRoutingRegistry


pytestmark = pytest.mark.unit


def test_direct_tool_planner_allows_known_safe_scenario() -> None:
    plan = _planner().plan(
        _decision(
            IntentFamily.SAFE_BUILTIN_TOOL,
            candidates=(
                _candidate(
                    Capability.TOOL_SAFE,
                    IntentFamily.SAFE_BUILTIN_TOOL,
                    "datetime.now",
                ),
            ),
            classification_source="deterministic",
        ),
        user_input="Сколько время?",
    )

    assert plan is not None
    assert plan.scenario == "current_time"
    assert plan.tool_names == ("datetime.now",)
    assert plan.redacted_metadata()["tool_names"] == ["datetime.now"]


def test_direct_tool_planner_denies_model_origin_direct_execution() -> None:
    plan = _planner().plan(
        _decision(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            candidates=(
                _candidate(
                    Capability.TOOL_SYSTEM_READ_RESOURCES,
                    IntentFamily.SYSTEM_DIAGNOSTICS,
                    "tool.system.read.resources",
                    scope_hint="disk_free",
                ),
            ),
            classification_source="model",
        ),
        user_input="Сколько свободного места на диске?",
    )

    assert plan is None


def test_direct_tool_planner_denies_tool_scope_mismatch() -> None:
    plan = _planner().plan(
        _decision(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            candidates=(
                _candidate(
                    Capability.TOOL_SYSTEM_READ_NETWORK,
                    IntentFamily.SYSTEM_DIAGNOSTICS,
                    "tool.system.read.hardware",
                    scope_hint="vpn_status",
                ),
            ),
            classification_source="deterministic",
        ),
        user_input="Включен ли VPN сейчас?",
    )

    assert plan is None


def test_direct_scope_evidence_comes_from_registry_backed_extractors() -> None:
    plan = _planner().plan(
        _decision(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            candidates=(
                _candidate(
                    Capability.TOOL_SYSTEM_READ_PROCESS,
                    IntentFamily.SYSTEM_DIAGNOSTICS,
                    "tool.system.read.process",
                    scope_hint="process_name_search",
                ),
            ),
            classification_source="deterministic",
        ),
        user_input='Запущен ли сейчас процесс, в имени которого есть "HFT"?',
    )

    assert plan is not None
    assert plan.scenario == "process_name_search"
    assert plan.required_arguments == {"process_pattern": "HFT"}
    assert "process_name_search_pattern" in plan.provenance


def test_direct_tool_plan_requires_process_search_pattern() -> None:
    plan = _planner().plan(
        _decision(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            candidates=(
                _candidate(
                    Capability.TOOL_SYSTEM_READ_PROCESS,
                    IntentFamily.SYSTEM_DIAGNOSTICS,
                    "tool.system.read.process",
                    scope_hint="process_name_search",
                ),
            ),
            classification_source="deterministic",
        ),
        user_input="Запущен ли сейчас процесс?",
    )

    assert plan is None


def test_direct_tool_plan_round_trips_through_request_metadata() -> None:
    plan = _planner().plan(
        _decision(
            IntentFamily.SYSTEM_DIAGNOSTICS,
            candidates=(
                _candidate(
                    Capability.TOOL_SYSTEM_READ_NETWORK,
                    IntentFamily.SYSTEM_DIAGNOSTICS,
                    "tool.system.read.network",
                    scope_hint="vpn_status",
                ),
            ),
            classification_source="guardrail",
        ),
        user_input="Включен ли VPN сейчас?",
    )

    assert plan is not None
    metadata = plan.redacted_metadata()
    restored = direct_tool_plan_from_metadata({"loop_selection_direct_tool_plan": metadata})

    assert restored == plan
    assert "argv" not in str(metadata).lower()


def test_direct_tool_plan_from_metadata_rejects_model_origin_plan() -> None:
    metadata = _direct_plan_payload(
        scenario="current_time",
        tool_names=["datetime.now"],
        capabilities=["tool.safe"],
        classification_source="model",
    )

    restored = direct_tool_plan_from_metadata({"loop_selection_direct_tool_plan": metadata})

    assert restored is None


def test_direct_tool_plan_from_metadata_rejects_tool_scope_mismatch() -> None:
    metadata = _direct_plan_payload(
        scenario="current_time",
        tool_names=["tool.system.read.process"],
        capabilities=["tool.safe"],
    )

    restored = direct_tool_plan_from_metadata({"loop_selection_direct_tool_plan": metadata})

    assert restored is None


def test_direct_tool_plan_from_metadata_rejects_unknown_tool_name() -> None:
    metadata = _direct_plan_payload(
        scenario="current_time",
        tool_names=["tool.system.read.unknown"],
        capabilities=["tool.safe"],
    )

    restored = direct_tool_plan_from_metadata({"loop_selection_direct_tool_plan": metadata})

    assert restored is None


def test_direct_tool_plan_from_metadata_ignores_legacy_scalar_direct_metadata() -> None:
    restored = direct_tool_plan_from_metadata(
        {
            "loop_selection_direct_tool_name": "datetime.now",
            "loop_selection_direct_scenario": "current_time",
        }
    )

    assert restored is None


def test_direct_tool_plan_from_metadata_rejects_non_string_process_pattern() -> None:
    metadata = _direct_plan_payload(
        scenario="process_name_search",
        tool_names=["tool.system.read.process"],
        capabilities=["tool.system.read.process"],
        scope_hint="process_name_search",
        required_arguments={"process_pattern": ["HFT"]},
    )

    restored = direct_tool_plan_from_metadata({"loop_selection_direct_tool_plan": metadata})

    assert restored is None


def _planner() -> DirectToolPlanner:
    settings = ConfigLoader(Path("config")).load("test")
    return DirectToolPlanner(CapabilityRoutingRegistry.from_settings(settings))


def _direct_plan_payload(
    *,
    scenario: str,
    tool_names: list[str],
    capabilities: list[str],
    scope_hint: str | None = None,
    classification_source: str = "deterministic",
    required_arguments: dict[str, str] | None = None,
) -> dict:
    return {
        "version": 1,
        "scenario": scenario,
        "tool_names": tool_names,
        "capabilities": capabilities,
        "scope_hint": scope_hint,
        "classification_source": classification_source,
        "provenance": ["unit_fixture"],
        "required_arguments": required_arguments or {},
    }


def _decision(
    intent_family: IntentFamily,
    *,
    candidates: tuple[CapabilityCandidate, ...],
    classification_source: str,
) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=LoopSelectionMode.AUTO,
        selected_loop_strategy=LoopStrategyName.TOOL_REACT_LOOP,
        selected_model_profile=None,
        intent_family=intent_family,
        reason_code="test_reason",
        confidence=0.9,
        candidate_capabilities=candidates,
        requires_tools=True,
        requires_live_state=True,
        policy_outcome=PolicyDecisionOutcome.ALLOW,
        approval_possible=False,
        fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        decision_status=SelectionDecisionStatus.SELECTED,
        classification_source=classification_source,
    )


def _candidate(
    capability: Capability,
    intent_family: IntentFamily,
    tool_name: str,
    *,
    scope_hint: str | None = None,
) -> CapabilityCandidate:
    return CapabilityCandidate(
        capability=capability,
        intent_family=intent_family,
        confidence=0.9,
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        tool_names=(tool_name,),
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        scope_hint=scope_hint,
        evidence_codes=("test_evidence",),
    )
