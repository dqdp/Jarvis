from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import (
    IntentFamily,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionDecisionStatus,
    SelectionFallbackPreference,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.models import StructuredModelRequest, StructuredModelResponse
from assistant_core.domain.policy import Capability, PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.runtime.loop_selection import DeterministicIntentClassifier
from assistant_core.runtime.model_intent_classifier import ModelBackedIntentClassifier
from assistant_core.runtime.loop_selection import LoopStrategySelector
from assistant_core.runtime.request_metadata import metadata_from_decision
from assistant_core.runtime.request_metadata import available_tools_summary
from assistant_core.runtime.routing import CapabilityRoutingRegistry


pytestmark = pytest.mark.unit

CORPUS_PATH = Path("tests/fixtures/intent_routing/tool_intent_corpus.json")
STABLE_ID = re.compile(r"^[a-z0-9_.-]+$")
STABLE_TOOL_NAME = re.compile(r"^[a-z0-9_.:-]+$")


def _baseline_cases(flag: str) -> list[dict[str, Any]]:
    return [case for case in _cases() if case.get(flag)]


def _runtime_baseline_cases() -> list[dict[str, Any]]:
    return [
        case
        for case in _cases()
        if case.get("ci_baseline") or case.get("guardrail_baseline")
    ]


def _cases() -> list[dict[str, Any]]:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    return payload["cases"]


def test_tool_intent_corpus_has_broad_multilingual_coverage() -> None:
    cases = _cases()
    languages = {case["language"] for case in cases}
    categories = {case["category"] for case in cases}
    tool_cases = [
        case
        for case in cases
        if case["expected"]["intent_family"] != IntentFamily.ORDINARY_CHAT.value
    ]

    assert len(cases) >= 70
    assert {"ru", "en", "es", "fr", "de"}.issubset(languages)
    assert {
        "safe.current_time",
        "safe.date_countdown",
        "safe.calculator",
        "safe.daemon_status",
        "system.os_version",
        "system.cpu_overview",
        "system.memory",
        "system.disk",
        "system.battery",
        "system.temperature",
        "system.processes",
        "system.network",
        "system.vpn",
        "project.inspection",
        "ordinary.conceptual",
    }.issubset(categories)
    assert len(tool_cases) >= 60


def test_tool_intent_corpus_has_required_categories_for_pre_voice_gate() -> None:
    categories = {case["category"] for case in _cases()}

    assert {
        "safe.current_time",
        "safe.date_countdown",
        "safe.calculator",
        "safe.daemon_status",
        "system.os_version",
        "system.cpu_overview",
        "system.memory",
        "system.disk",
        "system.battery",
        "system.temperature",
        "system.processes",
        "system.network",
        "system.vpn",
        "project.inspection",
        "project.docs_question",
        "ordinary.conceptual",
        "ordinary.near_miss_live_state",
        "tools_disabled.live_state",
        "spoken_transcript_variants",
    }.issubset(categories)


def test_tool_intent_corpus_required_categories_have_runtime_baselines() -> None:
    cases = _cases()
    runtime_baseline_flags = (
        "ci_baseline",
        "guardrail_baseline",
        "tools_disabled_baseline",
    )

    for category in (
        "safe.current_time",
        "safe.date_countdown",
        "safe.calculator",
        "safe.daemon_status",
        "system.os_version",
        "system.cpu_overview",
        "system.memory",
        "system.disk",
        "system.battery",
        "system.temperature",
        "system.processes",
        "system.network",
        "system.vpn",
        "project.inspection",
        "project.docs_question",
        "ordinary.conceptual",
        "ordinary.near_miss_live_state",
        "tools_disabled.live_state",
        "spoken_transcript_variants",
    ):
        assert any(
            case["category"] == category
            and any(case.get(flag) for flag in runtime_baseline_flags)
            for case in cases
        ), category


def test_tool_intent_corpus_covers_priority_live_state_variants() -> None:
    cases = _cases()

    assert _case_count(cases, category="system.disk") >= 3
    assert _case_count(cases, category="system.battery") >= 3
    assert _case_count(cases, category="system.vpn") >= 3
    assert _case_count(cases, category="safe.date_countdown") >= 3
    assert _case_count(cases, scope_hint="process_name_search") >= 2
    assert _case_count(cases, category="ordinary.conceptual") >= 10


def test_tool_intent_corpus_covers_negative_live_state_near_misses() -> None:
    cases = _cases()
    near_misses = [
        case for case in cases if case["category"] == "ordinary.near_miss_live_state"
    ]

    assert len(near_misses) >= 6
    assert {"ru", "en"}.issubset({case["language"] for case in near_misses})
    assert all(
        case["expected"]["intent_family"] == IntentFamily.ORDINARY_CHAT.value
        for case in near_misses
    )


def test_tool_intent_corpus_covers_direct_plan_scenarios() -> None:
    direct_cases = [
        case
        for case in _cases()
        if case["expected"].get("direct_plan", {}).get("expected") is True
    ]

    assert {
        "current_time",
        "christmas_countdown",
        "sensor_temperature",
        "memory_overview",
        "disk_free",
        "battery_charge",
        "os_version",
        "process_name_search",
        "vpn_status",
        "cpu_overview",
    }.issubset(
        {
            case["expected"]["direct_plan"]["scenario"]
            for case in direct_cases
        }
    )


def test_tool_intent_corpus_covers_spoken_transcript_variants() -> None:
    spoken_cases = [
        case for case in _cases() if case["category"] == "spoken_transcript_variants"
    ]

    assert len(spoken_cases) >= 10
    assert any("джарвис" in case["text"].casefold() for case in spoken_cases)
    assert any("jarvis" in case["text"].casefold() for case in spoken_cases)
    assert any(
        case["expected"].get("transcript_noise") == "mixed_ru_en"
        for case in spoken_cases
    )
    assert any(
        case["expected"].get("transcript_noise") == "missing_punctuation"
        for case in spoken_cases
    )


def test_tool_intent_corpus_asserts_policy_outcome_for_relevant_cases() -> None:
    for case in _cases():
        expected = case["expected"]
        if expected["intent_family"] in {
            IntentFamily.ORDINARY_CHAT.value,
            IntentFamily.PROJECT_DOCS_QUESTION.value,
        }:
            continue
        if case["category"] == "tools_disabled.live_state":
            assert expected["fallback_behavior"] == SelectionFallbackPreference.FAIL_UNAVAILABLE.value
            assert expected["policy_outcome"] == "deny", case["id"]
            assert expected["approval_possible"] is False, case["id"]
        else:
            assert expected["policy_outcome"] == "allow", case["id"]
            assert expected["approval_possible"] is False, case["id"]


@pytest.mark.parametrize("case", _runtime_baseline_cases(), ids=lambda case: case["id"])
def test_ci_baseline_selector_matches_policy_fallback_and_runtime_strategy(
    case: dict[str, Any],
) -> None:
    settings = ConfigLoader(Path("config")).load("test")
    decision = asyncio.run(
        LoopStrategySelector(
            intent_classifier=_classifier_for_case(case),
            policy=ConfigPolicyEngine(settings),
            tools_enabled=settings.policy.tools_enabled,
        ).select(_request(case["text"]))
    )
    expected = case["expected"]

    assert decision.intent_family.value == expected["intent_family"], case["id"]
    assert decision.fallback_behavior.value == expected["fallback_behavior"], case["id"]
    if expected["intent_family"] in {
        IntentFamily.ORDINARY_CHAT.value,
        IntentFamily.PROJECT_DOCS_QUESTION.value,
    }:
        assert decision.selected_loop_strategy is LoopStrategyName.MEMORY_AUGMENTED_ANSWER
        assert decision.decision_status is SelectionDecisionStatus.SELECTED
        assert decision.policy_outcome is None
        assert decision.approval_possible is False
        return

    assert decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP, case["id"]
    assert decision.decision_status is SelectionDecisionStatus.SELECTED, case["id"]
    assert decision.policy_outcome is not None, case["id"]
    assert decision.policy_outcome.value == expected["policy_outcome"], case["id"]
    assert decision.approval_possible is expected["approval_possible"], case["id"]


def test_tool_intent_corpus_requires_languages_for_priority_categories() -> None:
    cases = _cases()
    for category in (
        "safe.current_time",
        "system.os_version",
        "system.cpu_overview",
        "system.memory",
        "system.disk",
        "system.battery",
        "system.temperature",
        "system.processes",
        "system.vpn",
        "ordinary.near_miss_live_state",
    ):
        languages = {
            case["language"]
            for case in cases
            if case["category"] == category
        }
        assert {"ru", "en"}.issubset(languages), category


def test_tool_intent_corpus_schema_is_valid() -> None:
    seen_ids: set[str] = set()
    for case in _cases():
        assert STABLE_ID.fullmatch(case["id"]), case["id"]
        assert case["id"] not in seen_ids
        seen_ids.add(case["id"])
        assert case["language"]
        assert case["category"]
        assert case["text"]
        expected = case["expected"]
        assert IntentFamily(expected["intent_family"])
        assert "fallback_behavior" in expected
        assert expected["fallback_behavior"] in {"chat", "fail_unavailable"}
        for capability in expected["capabilities"]:
            assert Capability(capability)
        for tool_name in expected["tool_names"]:
            assert STABLE_TOOL_NAME.fullmatch(tool_name), case["id"]
        if "scope_hint" in expected:
            assert STABLE_TOOL_NAME.fullmatch(expected["scope_hint"]), case["id"]
        if "direct_plan" in expected:
            direct_plan = expected["direct_plan"]
            assert isinstance(direct_plan["expected"], bool), case["id"]
            if direct_plan["expected"]:
                assert STABLE_TOOL_NAME.fullmatch(direct_plan["scenario"]), case["id"]
                assert direct_plan["tool_names"] == expected["tool_names"], case["id"]
        if "transcript_noise" in expected:
            assert expected["transcript_noise"] in {
                "filler_prefix",
                "wake_prefix",
                "missing_punctuation",
                "mixed_ru_en",
                "casing",
            }
        if expected["intent_family"] not in {
            IntentFamily.ORDINARY_CHAT.value,
            IntentFamily.PROJECT_DOCS_QUESTION.value,
        }:
            assert expected["capabilities"], case["id"]


@pytest.mark.parametrize("case", _baseline_cases("ci_baseline"), ids=lambda case: case["id"])
def test_ci_baseline_classifier_routes_tool_intent_corpus_cases(case: dict[str, Any]) -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(_request(case["text"]))
    )

    _assert_classification_matches_expected(case, classification)


@pytest.mark.parametrize("case", _baseline_cases("ci_baseline"), ids=lambda case: case["id"])
def test_tool_intent_corpus_exact_ci_baseline_matches_expected_tools(
    case: dict[str, Any],
) -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(_request(case["text"]))
    )

    _assert_classification_matches_expected(case, classification, exact=True)


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in _cases()
        if (case.get("ci_baseline") or case.get("guardrail_baseline"))
        and "direct_plan" in case["expected"]
    ],
    ids=lambda case: case["id"],
)
def test_tool_intent_corpus_direct_plan_expectations(case: dict[str, Any]) -> None:
    settings = ConfigLoader(Path("config")).load("test")
    request = _request(case["text"])
    decision = asyncio.run(
        LoopStrategySelector(
            intent_classifier=_classifier_for_case(case),
            policy=ConfigPolicyEngine(settings),
            tools_enabled=settings.policy.tools_enabled,
        ).select(request)
    )

    metadata = metadata_from_decision(
        decision,
        body=_Body(case["text"]),
        model_profile="local_structured",
        routing_registry=CapabilityRoutingRegistry.from_settings(settings),
    )
    expected = case["expected"]["direct_plan"]
    direct_plan = metadata.get("loop_selection_direct_tool_plan")
    if expected["expected"]:
        assert direct_plan is not None, case["id"]
        assert direct_plan["scenario"] == expected["scenario"], case["id"]
        assert direct_plan["tool_names"] == expected["tool_names"], case["id"]
    else:
        assert direct_plan is None, case["id"]


@pytest.mark.parametrize(
    "case",
    _baseline_cases("tools_disabled_baseline"),
    ids=lambda case: case["id"],
)
def test_pre_voice_routing_gate_blocks_missing_priority_categories(
    case: dict[str, Any],
) -> None:
    request = _request(case["text"])
    decision = asyncio.run(
        LoopStrategySelector(
            intent_classifier=DeterministicIntentClassifier(),
            policy=None,
            tools_enabled=False,
        ).select(request)
    )

    assert decision.selected_loop_strategy is None
    assert decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE
    assert decision.reason_code == "tools_disabled_for_tool_intent"
    assert decision.policy_outcome is not None
    assert decision.policy_outcome.value == case["expected"]["policy_outcome"]
    assert decision.approval_possible is case["expected"]["approval_possible"]
    assert decision.fallback_behavior.value == case["expected"]["fallback_behavior"]


@pytest.mark.parametrize(
    "case",
    _baseline_cases("guardrail_baseline"),
    ids=lambda case: case["id"],
)
def test_model_guardrail_routes_tool_intent_corpus_cases(case: dict[str, Any]) -> None:
    classification = asyncio.run(
        ModelBackedIntentClassifier(
            router=FakeOrdinaryChatRouter(),
            fallback=DeterministicIntentClassifier(),
        ).classify(_request(case["text"]))
    )

    _assert_classification_matches_expected(case, classification, exact=True)


def test_guardrail_baseline_does_not_turn_conceptual_questions_into_tools() -> None:
    for case in _cases():
        if case["category"] not in {
            "ordinary.conceptual",
            "ordinary.near_miss_live_state",
        }:
            continue
        classification = asyncio.run(
            ModelBackedIntentClassifier(
                router=FakeOrdinaryChatRouter(),
                fallback=DeterministicIntentClassifier(),
            ).classify(_request(case["text"]))
        )
        assert classification.intent_family is IntentFamily.ORDINARY_CHAT, case["id"]
        assert classification.candidate_capabilities == (), case["id"]


def test_deterministic_classifier_keeps_conceptual_near_misses_in_chat() -> None:
    for case in _cases():
        if case["category"] not in {
            "ordinary.conceptual",
            "ordinary.near_miss_live_state",
        }:
            continue
        classification = asyncio.run(
            DeterministicIntentClassifier().classify(_request(case["text"]))
        )
        assert classification.intent_family is IntentFamily.ORDINARY_CHAT, case["id"]
        assert classification.candidate_capabilities == (), case["id"]


def test_tool_intent_corpus_covers_model_classifier_fake_payloads() -> None:
    case = next(case for case in _cases() if case["id"] == "os.ru.001")
    router = FakeStructuredRouter(
        {
            "intent_family": "system_diagnostics",
            "confidence": 0.91,
            "candidate_capabilities": [
                {
                    "capability": "tool.system.read.hardware",
                    "intent_family": "system_diagnostics",
                    "confidence": 0.91,
                    "requires_live_state": True,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["tool.system.read.hardware"],
                    "risk_classes": ["read_only"],
                    "scope_hint": "os_version",
                    "evidence_codes": ["fake_model_os_version"],
                }
            ],
            "requires_live_state": True,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "model_system_diagnostics",
            "fallback_preference": "fail_unavailable",
        }
    )

    classification = asyncio.run(
        ModelBackedIntentClassifier(router=router).classify(_request(case["text"]))
    )

    _assert_classification_matches_expected(case, classification, exact=True)


def _assert_classification_matches_expected(
    case: dict[str, Any],
    classification,
    *,
    exact: bool = False,
) -> None:
    expected = case["expected"]
    assert classification.intent_family.value == expected["intent_family"], case["id"]
    actual_capabilities = {
        candidate.capability.value for candidate in classification.candidate_capabilities
    }
    if exact:
        assert actual_capabilities == set(expected["capabilities"]), case["id"]
    else:
        assert set(expected["capabilities"]).issubset(actual_capabilities), case["id"]
    actual_tool_names = {
        tool_name
        for candidate in classification.candidate_capabilities
        for tool_name in candidate.tool_names
    }
    if exact:
        assert actual_tool_names == set(expected["tool_names"]), case["id"]
    else:
        assert set(expected["tool_names"]).issubset(actual_tool_names), case["id"]
    if scope_hint := expected.get("scope_hint"):
        assert any(
            candidate.scope_hint == scope_hint
            for candidate in classification.candidate_capabilities
        ), case["id"]


def _case_count(
    cases: list[dict[str, Any]],
    *,
    category: str | None = None,
    scope_hint: str | None = None,
) -> int:
    return sum(
        1
        for case in cases
        if (category is None or case["category"] == category)
        and (scope_hint is None or case["expected"].get("scope_hint") == scope_hint)
    )


class FakeOrdinaryChatRouter:
    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        return StructuredModelResponse(
            value={
                "intent_family": "ordinary_chat",
                "confidence": 0.8,
                "candidate_capabilities": [],
                "requires_live_state": False,
                "requires_execution": False,
                "answer_without_tools_would_be_misleading": False,
                "reason_code": "ordinary_chat",
                "fallback_preference": "chat",
            }
        )


class FakeStructuredRouter:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        return StructuredModelResponse(value=self.value)


class _Body:
    def __init__(self, content: str) -> None:
        self.content = content
        self.model_profile = None
        self.working_directory = str(Path.cwd())


def _classifier_for_case(case: dict[str, Any]):
    if case.get("guardrail_baseline"):
        return ModelBackedIntentClassifier(
            router=FakeOrdinaryChatRouter(),
            fallback=DeterministicIntentClassifier(),
        )
    return DeterministicIntentClassifier()


def _request(text: str) -> LoopSelectionRequest:
    return LoopSelectionRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        user_id="user-1",
        requested_mode=LoopSelectionMode.AUTO,
        user_input=text,
        current_message_sensitivity=Sensitivity.PROJECT,
        active_project_namespace="project.personal_assistant",
        working_directory=str(Path.cwd()),
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
        available_capabilities=frozenset(Capability),
        available_tools_summary=available_tools_summary(ConfigLoader(Path("config")).load("test")),
        runtime_budget_summary={},
        metadata={"source": "intent_routing_corpus"},
    )
