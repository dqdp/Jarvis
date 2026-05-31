from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import IntentFamily, LoopSelectionMode, LoopSelectionRequest
from assistant_core.domain.models import StructuredModelRequest, StructuredModelResponse
from assistant_core.domain.policy import Capability, PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.loop_selection import DeterministicIntentClassifier
from assistant_core.runtime.model_intent_classifier import ModelBackedIntentClassifier
from assistant_core.runtime.request_metadata import available_tools_summary


pytestmark = pytest.mark.unit

CORPUS_PATH = Path("tests/fixtures/intent_routing/tool_intent_corpus.json")
STABLE_ID = re.compile(r"^[a-z0-9_.-]+$")
STABLE_TOOL_NAME = re.compile(r"^[a-z0-9_.:-]+$")


def _baseline_cases(flag: str) -> list[dict[str, Any]]:
    return [case for case in _cases() if case.get(flag)]


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


def test_tool_intent_corpus_covers_priority_live_state_variants() -> None:
    cases = _cases()

    assert _case_count(cases, category="system.disk") >= 3
    assert _case_count(cases, category="system.battery") >= 3
    assert _case_count(cases, category="system.vpn") >= 3
    assert _case_count(cases, category="safe.date_countdown") >= 3
    assert _case_count(cases, scope_hint="process_name_search") >= 2
    assert _case_count(cases, category="ordinary.conceptual") >= 10


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
        for capability in expected["capabilities"]:
            assert Capability(capability)
        for tool_name in expected["tool_names"]:
            assert STABLE_TOOL_NAME.fullmatch(tool_name), case["id"]
        if "scope_hint" in expected:
            assert STABLE_TOOL_NAME.fullmatch(expected["scope_hint"]), case["id"]
        if expected["intent_family"] != IntentFamily.ORDINARY_CHAT.value:
            assert expected["capabilities"], case["id"]


@pytest.mark.parametrize("case", _baseline_cases("ci_baseline"), ids=lambda case: case["id"])
def test_ci_baseline_classifier_routes_tool_intent_corpus_cases(case: dict[str, Any]) -> None:
    classification = asyncio.run(
        DeterministicIntentClassifier().classify(_request(case["text"]))
    )

    _assert_classification_matches_expected(case, classification)


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

    _assert_classification_matches_expected(case, classification)


def _assert_classification_matches_expected(case: dict[str, Any], classification) -> None:
    expected = case["expected"]
    assert classification.intent_family.value == expected["intent_family"], case["id"]
    actual_capabilities = {
        candidate.capability.value for candidate in classification.candidate_capabilities
    }
    assert set(expected["capabilities"]).issubset(actual_capabilities), case["id"]
    actual_tool_names = {
        tool_name
        for candidate in classification.candidate_capabilities
        for tool_name in candidate.tool_names
    }
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


def _request(text: str) -> LoopSelectionRequest:
    return LoopSelectionRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        user_id="user-1",
        requested_mode=LoopSelectionMode.AUTO,
        user_input=text,
        current_message_sensitivity=Sensitivity.PROJECT,
        active_project_namespace="project.personal_assistant",
        working_directory="/tmp/project",
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
        available_capabilities=frozenset(Capability),
        available_tools_summary=available_tools_summary(ConfigLoader(Path("config")).load("test")),
        runtime_budget_summary={},
        metadata={"source": "intent_routing_corpus"},
    )
