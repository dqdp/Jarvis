from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import LoopSelectionMode, LoopSelectionRequest
from assistant_core.domain.models import StructuredModelRequest, StructuredModelResponse
from assistant_core.domain.policy import Capability, PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.runtime.loop_selection import DeterministicIntentClassifier, LoopStrategySelector
from assistant_core.runtime.model_intent_classifier import ModelBackedIntentClassifier
from assistant_core.runtime.request_metadata import available_tools_summary, metadata_from_decision
from assistant_core.runtime.routing import CapabilityRoutingRegistry


pytestmark = pytest.mark.evaluation

CORPUS_PATH = Path("tests/fixtures/intent_routing/tool_intent_corpus.json")
REPORT_PATH = Path("tests/fixtures/intent_routing/pre_voice_local_model_eval_report.json")
MODEL_COMPARISON_PATH = Path(
    "tests/fixtures/intent_routing/pm08i_classifier_model_comparison.json"
)


def test_pre_voice_local_model_eval_report_blocks_voice_until_real_local_run() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    _assert_voice_readiness_report_contract(report)


def test_pre_voice_local_model_eval_report_rejects_false_pass_evidence() -> None:
    bad_report = {
        "version": 1,
        "corpus_version": _payload()["version"],
        "slice": "PM-08h",
        "model": "local-classifier",
        "status": "passed",
        "voice_ready_for_pm09": True,
        "summary": {
            "evaluated_cases": 12,
            "failure_count": 1,
        },
        "failures": [
            {
                "case_id": "case-1",
                "category": "system.memory",
                "language": "en",
                "field": "direct_plan",
                "actual": None,
                "expected": {"expected": True},
            }
        ],
        "notes": ["bad report fixture"],
    }

    with pytest.raises(AssertionError):
        _assert_voice_readiness_report_contract(bad_report)


def test_pre_voice_local_model_eval_report_rejects_partial_pass_evidence() -> None:
    bad_report = {
        "version": 1,
        "corpus_version": _payload()["version"],
        "slice": "PM-08h",
        "model": "local-classifier",
        "status": "passed",
        "voice_ready_for_pm09": True,
        "summary": {
            "evaluated_cases": 1,
            "failure_count": 0,
            "model_called_cases": 1,
            "fallback_routed_cases": 0,
            "guardrail_corrected_cases": 0,
        },
        "failures": [],
        "notes": ["bad report fixture"],
    }

    with pytest.raises(AssertionError):
        _assert_voice_readiness_report_contract(bad_report)


def test_local_model_classifier_eval_default_model_tracks_ollama_structured_profile(
    monkeypatch,
) -> None:
    monkeypatch.delenv("JARVIS_INTENT_ROUTING_EVAL_MODEL", raising=False)

    settings = ConfigLoader(Path("config")).load("ollama")

    assert _eval_model_name() == settings.model_profiles["local_structured"].model


def test_pre_voice_local_model_eval_report_tracks_structured_profile() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    settings = ConfigLoader(Path("config")).load("ollama")

    assert report["model"] == settings.model_profiles["local_structured"].model


def test_pm08i_classifier_model_comparison_records_default_decision() -> None:
    report = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))

    assert report["slice"] == "PM-08i"
    assert report["selected_default_model"] == "qwen3.5:2b"
    assert report["candidate_models"] == ["qwen3.5:2b", "qwen3.5:0.8b"]
    assert report["recommendation"] == "keep_qwen3.5:2b"
    assert report["summary"]["deterministic_fast_path_threshold"] == 0.9
    assert report["summary"]["qwen3.5:0.8b"]["failed_cases"] > 0


def _assert_voice_readiness_report_contract(report: dict[str, Any]) -> None:
    assert report["version"] == 1
    assert report["corpus_version"] == _payload()["version"]
    assert report["slice"] == "PM-08h"
    assert report["model"]
    assert report["status"] in {"passed", "accepted_known_failures", "not_run_in_sandbox"}
    assert isinstance(report["summary"]["evaluated_cases"], int)
    assert isinstance(report["summary"]["failure_count"], int)
    assert report["notes"]
    if report["status"] == "not_run_in_sandbox":
        assert report["voice_ready_for_pm09"] is False
        assert report["blocking_reason"] == "local_model_eval_not_run"
        assert report["summary"]["evaluated_cases"] == 0
    else:
        assert report["voice_ready_for_pm09"] is True
        assert report["summary"]["evaluated_cases"] == len(_cases())
        assert report["summary"]["model_called_cases"] == report["summary"]["evaluated_cases"]
        assert report["summary"]["fallback_routed_cases"] == 0
        assert report["summary"]["guardrail_corrected_cases"] == 0
        if report["status"] == "passed":
            assert report["summary"]["failure_count"] == 0
            assert report["failures"] == []
        if report["status"] == "accepted_known_failures":
            assert report["summary"]["failure_count"] > 0
            assert report.get("accepted_failures")


def test_local_model_classifier_reports_failures_without_ci_network_or_real_llm() -> None:
    cases = [
        case
        for case in _cases()
        if case["expected"]["intent_family"] == "ordinary_chat"
    ][:3]
    report = asyncio.run(
        evaluate_local_model_classifier(
            cases,
            router=StaticStructuredRouter(
                {
                    "intent_family": "safe_builtin_tool",
                    "confidence": 0.8,
                    "candidate_capabilities": [
                        {
                            "capability": "tool.safe",
                            "intent_family": "safe_builtin_tool",
                            "confidence": 0.8,
                            "requires_live_state": True,
                            "requires_execution": True,
                            "requires_write": False,
                            "tool_names": ["datetime.now"],
                            "risk_classes": ["safe"],
                            "scope_hint": None,
                            "evidence_codes": ["fake_wrong_safe_tool"],
                        }
                    ],
                    "requires_live_state": True,
                    "requires_execution": True,
                    "answer_without_tools_would_be_misleading": True,
                    "reason_code": "fake_wrong_safe_tool",
                    "fallback_preference": "fail_unavailable",
                }
            ),
        )
    )

    assert report["summary"]["evaluated_cases"] == 3
    assert report["summary"]["failure_count"] == 3
    assert report["failures"] == [
        {
            "case_id": cases[0]["id"],
            "category": cases[0]["category"],
            "language": cases[0]["language"],
            "field": "intent_family",
            "actual": "safe_builtin_tool",
            "expected": cases[0]["expected"]["intent_family"],
        },
        {
            "case_id": cases[1]["id"],
            "category": cases[1]["category"],
            "language": cases[1]["language"],
            "field": "intent_family",
            "actual": "safe_builtin_tool",
            "expected": cases[1]["expected"]["intent_family"],
        },
        {
            "case_id": cases[2]["id"],
            "category": cases[2]["category"],
            "language": cases[2]["language"],
            "field": "intent_family",
            "actual": "safe_builtin_tool",
            "expected": cases[2]["expected"]["intent_family"],
        },
    ]


def test_local_model_classifier_routes_pre_voice_corpus_opt_in() -> None:
    if os.environ.get("JARVIS_RUN_INTENT_ROUTING_CORPUS_EVAL") != "1":
        pytest.skip("set JARVIS_RUN_INTENT_ROUTING_CORPUS_EVAL=1 to run local model eval")

    cases = _cases()
    router = OllamaStructuredRouter(
        endpoint=os.environ.get("JARVIS_INTENT_ROUTING_EVAL_OLLAMA_URL", "http://127.0.0.1:11434"),
        model=_eval_model_name(),
        timeout_seconds=int(os.environ.get("JARVIS_INTENT_ROUTING_EVAL_TIMEOUT_SECONDS", "60")),
    )
    report = asyncio.run(
        evaluate_local_model_classifier(
            cases,
            router=router,
            deterministic_fast_path_threshold=float(
                os.environ.get(
                    "JARVIS_INTENT_ROUTING_EVAL_FAST_PATH_THRESHOLD",
                    "0.9",
                )
            ),
            use_deterministic_fallback=(
                os.environ.get("JARVIS_INTENT_ROUTING_EVAL_USE_DETERMINISTIC_FALLBACK")
                == "1"
            ),
        )
    )

    assert report["failures"] == []


def test_local_model_classifier_eval_applies_tools_disabled_case_semantics() -> None:
    case = next(case for case in _cases() if case["id"] == "disabled.en.001")
    report = asyncio.run(
        evaluate_local_model_classifier(
            [case],
            router=StaticStructuredRouter(
                {
                    "intent_family": "system_diagnostics",
                    "confidence": 0.9,
                    "candidate_capabilities": [
                        {
                            "capability": "tool.system.read.resources",
                            "intent_family": "system_diagnostics",
                            "confidence": 0.9,
                            "requires_live_state": True,
                            "requires_execution": True,
                            "requires_write": False,
                            "tool_names": ["tool.system.read.resources"],
                            "risk_classes": ["read_only"],
                            "scope_hint": "disk_free",
                            "evidence_codes": ["fake_disk_free"],
                        }
                    ],
                    "requires_live_state": True,
                    "requires_execution": True,
                    "answer_without_tools_would_be_misleading": True,
                    "reason_code": "fake_disk_free",
                    "fallback_preference": "fail_unavailable",
                }
            ),
        )
    )

    assert report["failures"] == []


def test_local_model_classifier_eval_calls_model_for_safe_builtin_cases() -> None:
    cases = [
        case
        for case in _cases()
        if case["category"] in {"safe.calculator", "safe.daemon_status"}
    ][:2]
    router = CountingStructuredRouter(
        {
            "intent_family": "safe_builtin_tool",
            "confidence": 0.9,
            "candidate_capabilities": [
                {
                    "capability": "tool.safe",
                    "intent_family": "safe_builtin_tool",
                    "confidence": 0.9,
                    "requires_live_state": False,
                    "requires_execution": True,
                    "requires_write": False,
                    "tool_names": ["calculator.evaluate"],
                    "risk_classes": ["safe"],
                    "scope_hint": None,
                    "evidence_codes": ["fake_safe_builtin"],
                }
            ],
            "requires_live_state": False,
            "requires_execution": True,
            "answer_without_tools_would_be_misleading": True,
            "reason_code": "fake_safe_builtin",
            "fallback_preference": "fail_unavailable",
        }
    )

    report = asyncio.run(evaluate_local_model_classifier(cases, router=router))

    assert router.call_count == len(cases)
    assert report["summary"]["model_called_cases"] == len(cases)
    assert report["summary"]["fallback_routed_cases"] == 0
    assert report["summary"]["guardrail_corrected_cases"] == 0


def test_local_model_classifier_eval_reports_guardrail_correction_as_failure() -> None:
    case = next(case for case in _cases() if case["id"] == "os.ru.002")
    report = asyncio.run(
        evaluate_local_model_classifier(
            [case],
            router=StaticStructuredRouter(
                {
                    "intent_family": "ordinary_chat",
                    "confidence": 0.9,
                    "candidate_capabilities": [],
                    "requires_live_state": False,
                    "requires_execution": False,
                    "answer_without_tools_would_be_misleading": False,
                    "reason_code": "fake_ordinary_chat",
                    "fallback_preference": "chat",
                }
            ),
        )
    )

    assert report["summary"]["model_called_cases"] == 1
    assert report["summary"]["guardrail_corrected_cases"] == 1
    assert {
        "case_id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "field": "classification_source",
        "actual": "guardrail",
        "expected": "model",
    } in report["failures"]


async def evaluate_local_model_classifier(
    cases: list[dict[str, Any]],
    *,
    router,
    deterministic_fast_path_threshold: float = 0.9,
    use_deterministic_fallback: bool = False,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    settings = ConfigLoader(Path("config")).load("test")
    registry = CapabilityRoutingRegistry.from_settings(settings)
    counting_router = CountingRouter(router)
    guardrail_corrected_cases = 0
    for case in cases:
        request = _request(case["text"])
        classification = await ModelBackedIntentClassifier(
            router=counting_router,
            deterministic_fast_path_threshold=deterministic_fast_path_threshold,
            fallback=(
                DeterministicIntentClassifier()
                if use_deterministic_fallback
                else None
            ),
        ).classify(request)
        decision = await LoopStrategySelector(
            intent_classifier=StaticIntentClassifier(classification),
            policy=ConfigPolicyEngine(settings),
            tools_enabled=not case.get("tools_disabled_baseline")
            and settings.policy.tools_enabled,
        ).select(request)
        metadata = metadata_from_decision(
            decision,
            body=_Body(case["text"]),
            model_profile="local_structured",
            routing_registry=registry,
        )
        expected = case["expected"]
        if classification.classification_source != "model":
            guardrail_corrected_cases += 1
            failures.append(
                _failure(
                    case,
                    "classification_source",
                    classification.classification_source,
                    expected="model",
                )
            )
        actual_capabilities = {
            candidate.capability.value for candidate in classification.candidate_capabilities
        }
        actual_tool_names = {
            tool_name
            for candidate in classification.candidate_capabilities
            for tool_name in candidate.tool_names
        }
        if classification.intent_family.value != expected["intent_family"]:
            failures.append(_failure(case, "intent_family", classification.intent_family.value))
            continue
        if actual_capabilities != set(expected["capabilities"]):
            failures.append(_failure(case, "capabilities", sorted(actual_capabilities)))
        if actual_tool_names != set(expected["tool_names"]):
            failures.append(_failure(case, "tool_names", sorted(actual_tool_names)))
        if decision.fallback_behavior.value != expected["fallback_behavior"]:
            failures.append(
                _failure(case, "fallback_behavior", decision.fallback_behavior.value)
            )
        if "policy_outcome" in expected:
            actual_policy = decision.policy_outcome.value if decision.policy_outcome else None
            if actual_policy != expected["policy_outcome"]:
                failures.append(_failure(case, "policy_outcome", actual_policy))
        if "approval_possible" in expected and decision.approval_possible != expected["approval_possible"]:
            failures.append(_failure(case, "approval_possible", decision.approval_possible))
        if "direct_plan" in expected and not case.get("tools_disabled_baseline"):
            actual_plan = metadata.get("loop_selection_direct_tool_plan")
            expected_plan = expected["direct_plan"]
            if expected_plan["expected"]:
                if actual_plan is None:
                    failures.append(_failure(case, "direct_plan", None))
                elif (
                    actual_plan.get("scenario") != expected_plan["scenario"]
                    or actual_plan.get("tool_names") != expected_plan["tool_names"]
                ):
                    failures.append(_failure(case, "direct_plan", actual_plan))
            elif actual_plan is not None:
                failures.append(_failure(case, "direct_plan", actual_plan))

    return {
        "version": 1,
        "corpus_version": _payload()["version"],
        "summary": {
            "evaluated_cases": len(cases),
            "failure_count": len(failures),
            "model_called_cases": counting_router.call_count,
            "fallback_routed_cases": len(cases) - counting_router.call_count,
            "guardrail_corrected_cases": guardrail_corrected_cases,
            "deterministic_fast_path_threshold": deterministic_fast_path_threshold,
            "deterministic_fallback_enabled": use_deterministic_fallback,
        },
        "failures": failures,
    }


def _failure(
    case: dict[str, Any],
    field: str,
    actual: Any,
    *,
    expected: Any | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "field": field,
        "actual": actual,
        "expected": case["expected"].get(field) if expected is None else expected,
    }


class OllamaStructuredRouter:
    def __init__(self, *, endpoint: str, model: str, timeout_seconds: int) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": "\n".join(part.text for part in message.content),
                }
                for message in request.messages
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 1024,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(f"{self._endpoint}/api/chat", json=payload)
            response.raise_for_status()
        content = response.json()["message"]["content"]
        return StructuredModelResponse(value=_json_object(content))


class StaticStructuredRouter:
    def __init__(self, value: dict[str, Any]) -> None:
        self._value = value

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        return StructuredModelResponse(value=self._value)


class CountingRouter:
    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self.call_count = 0

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        self.call_count += 1
        return await self._wrapped.structured(request)


class CountingStructuredRouter(StaticStructuredRouter):
    def __init__(self, value: dict[str, Any]) -> None:
        super().__init__(value)
        self.call_count = 0

    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse:
        self.call_count += 1
        return await super().structured(request)


class StaticIntentClassifier:
    def __init__(self, classification) -> None:
        self._classification = classification

    async def classify(self, request: LoopSelectionRequest):
        return self._classification


class _Body:
    def __init__(self, content: str) -> None:
        self.content = content
        self.model_profile = None
        self.working_directory = str(Path.cwd())


def _json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model did not return a JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model did not return a JSON object")
    return value


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
        metadata={"source": "intent_routing_corpus_eval"},
    )


def _payload() -> dict[str, Any]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _cases() -> list[dict[str, Any]]:
    return _payload()["cases"]


def _eval_model_name() -> str:
    configured = os.environ.get("JARVIS_INTENT_ROUTING_EVAL_MODEL")
    if configured:
        return configured
    settings = ConfigLoader(Path("config")).load("ollama")
    return settings.model_profiles["local_structured"].model
