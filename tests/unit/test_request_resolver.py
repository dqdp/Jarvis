from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.loop_selection import (
    IntentFamily,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionFallbackPreference,
)
from assistant_core.domain.policy import Capability, PermissionMode
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.runtime.request_resolver import (
    Abstain,
    CalibrationObservation,
    Clarify,
    HybridRequestResolver,
    MODEL_ROUTE_SCHEMA,
    RequestRoute,
    RequestResolverIntentClassifier,
    RouteDecision,
    RouteRegistry,
    Unavailable,
    build_classifier_calibration_report,
    parse_model_route_output,
)
from assistant_core.runtime.routing import CapabilityRoutingRegistry


pytestmark = pytest.mark.unit


def test_request_resolver_returns_explicit_result_types() -> None:
    assert RouteDecision(route=RequestRoute.ORDINARY_CHAT, confidence=0.9, reason_code="test")
    assert Abstain(reason_code="ambiguous").reason_code == "ambiguous"
    assert Clarify(reason_code="needs_choice").fallback_preference is (
        SelectionFallbackPreference.ASK_CLARIFICATION
    )
    assert Unavailable(reason_code="no_route").fallback_preference is (
        SelectionFallbackPreference.FAIL_UNAVAILABLE
    )


def test_request_resolver_bypasses_classifier_for_obvious_ordinary_chat() -> None:
    adjudicator = RecordingResolver(RouteDecision(RequestRoute.SYSTEM_MEMORY, 0.91, "wrong"))
    resolver = HybridRequestResolver(llm_adjudicator=adjudicator)

    result = asyncio.run(
        resolver.resolve(_request(user_input="Расскажи, как решаются кубические уравнения."))
    )

    assert isinstance(result, RouteDecision)
    assert result.route is RequestRoute.ORDINARY_CHAT
    assert result.reason_code == "ordinary_chat_bypass"
    assert adjudicator.requests == []


def test_request_resolver_uses_deterministic_guards_for_obvious_safe_routes() -> None:
    resolver = HybridRequestResolver()

    current_time = asyncio.run(resolver.resolve(_request(user_input="Сколько времени?")))
    calculator = asyncio.run(resolver.resolve(_request(user_input="Посчитай 128 * 64")))
    daemon = asyncio.run(resolver.resolve(_request(user_input="Check the assistant daemon status")))

    assert current_time == RouteDecision(RequestRoute.CURRENT_TIME, 0.76, "current_time_hint")
    assert calculator == RouteDecision(RequestRoute.CALCULATOR, 0.76, "calculator_hint")
    assert daemon == RouteDecision(RequestRoute.DAEMON_STATUS, 0.76, "daemon_status_hint")


def test_request_resolver_clarifies_risky_live_state_ambiguity() -> None:
    resolver = HybridRequestResolver()

    result = asyncio.run(resolver.resolve(_request(user_input="память")))

    assert isinstance(result, Clarify)
    assert result.reason_code == "ambiguous_live_state_or_conceptual"


@pytest.mark.parametrize("user_input", ["cpu?", "memory.", "память?", "vpn!"])
def test_request_resolver_clarifies_punctuated_live_state_fragments(user_input: str) -> None:
    resolver = HybridRequestResolver()

    result = asyncio.run(resolver.resolve(_request(user_input=user_input)))

    assert isinstance(result, Clarify)
    assert result.reason_code == "ambiguous_live_state_or_conceptual"


def test_request_resolver_keeps_non_llm_semantic_layer_evaluation_only_by_default() -> None:
    semantic = RecordingResolver(RouteDecision(RequestRoute.SYSTEM_MEMORY, 0.91, "semantic"))
    resolver = HybridRequestResolver(semantic_resolver=semantic)

    result = asyncio.run(resolver.resolve(_request(user_input="проверь что-нибудь")))

    assert isinstance(result, Clarify)
    assert semantic.requests == []


def test_request_resolver_calls_llm_adjudicator_only_after_abstain() -> None:
    adjudicator = RecordingResolver(RouteDecision(RequestRoute.PROJECT_INSPECTION, 0.88, "llm_route"))
    resolver = HybridRequestResolver(llm_adjudicator=adjudicator)

    result = asyncio.run(resolver.resolve(_request(user_input="проверь что-нибудь")))

    assert result == RouteDecision(RequestRoute.PROJECT_INSPECTION, 0.88, "llm_route")
    assert len(adjudicator.requests) == 1


def test_route_registry_maps_current_time_to_safe_datetime_candidate() -> None:
    classification = RouteRegistry().classification_for(
        RouteDecision(RequestRoute.CURRENT_TIME, 0.91, "current_time_hint"),
        _request(),
    )

    assert classification.intent_family is IntentFamily.SAFE_BUILTIN_TOOL
    assert classification.candidate_capabilities[0].capability is Capability.TOOL_SAFE
    assert classification.candidate_capabilities[0].tool_names == ("datetime.now",)
    assert classification.reason_code == "current_time_hint"


def test_route_registry_maps_system_memory_to_read_resources_candidate() -> None:
    classification = RouteRegistry().classification_for(
        RouteDecision(RequestRoute.SYSTEM_MEMORY, 0.91, "system_memory_hint"),
        _request(),
    )

    assert classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS
    assert classification.candidate_capabilities[0].capability is (
        Capability.TOOL_SYSTEM_READ_RESOURCES
    )
    assert classification.candidate_capabilities[0].tool_names == (
        "tool.system.read.resources",
    )


def test_request_resolver_intent_classifier_maps_clarify_without_tool_metadata() -> None:
    classifier = RequestResolverIntentClassifier(
        resolver=StaticResolver(Clarify(reason_code="needs_user_choice"))
    )

    classification = asyncio.run(classifier.classify(_request()))

    assert classification.intent_family is IntentFamily.UNKNOWN
    assert classification.candidate_capabilities == ()
    assert classification.reason_code == "needs_user_choice"
    assert classification.fallback_preference is SelectionFallbackPreference.ASK_CLARIFICATION


def test_request_resolver_intent_classifier_maps_unavailable_without_tool_metadata() -> None:
    classifier = RequestResolverIntentClassifier(
        resolver=StaticResolver(Unavailable(reason_code="route_unavailable"))
    )

    classification = asyncio.run(classifier.classify(_request()))

    assert classification.intent_family is IntentFamily.UNKNOWN
    assert classification.candidate_capabilities == ()
    assert classification.reason_code == "route_unavailable"
    assert classification.fallback_preference is SelectionFallbackPreference.FAIL_UNAVAILABLE


def test_model_route_schema_contains_only_route_confidence_flags_and_abstain() -> None:
    assert set(MODEL_ROUTE_SCHEMA["properties"]) == {
        "route",
        "confidence",
        "requires_live_state",
        "is_conceptual_question",
        "abstain",
    }
    assert MODEL_ROUTE_SCHEMA["additionalProperties"] is False


def test_model_route_schema_uses_closed_route_enum() -> None:
    route_schema = MODEL_ROUTE_SCHEMA["properties"]["route"]

    assert route_schema["enum"] == [route.value for route in RequestRoute]
    assert "system_memory" in route_schema["enum"]


def test_model_route_parser_rejects_unknown_route() -> None:
    with pytest.raises(ValueError, match="unknown route"):
        parse_model_route_output(
            {
                "route": "general_inquiry",
                "confidence": 0.91,
                "requires_live_state": False,
                "is_conceptual_question": True,
                "abstain": False,
            }
        )


def test_model_route_parser_rejects_non_boolean_flags() -> None:
    with pytest.raises(ValueError, match="requires_live_state must be boolean"):
        parse_model_route_output(
            {
                "route": "ordinary_chat",
                "confidence": 0.91,
                "requires_live_state": "false",
                "is_conceptual_question": True,
                "abstain": False,
            }
        )


def test_model_route_output_cannot_supply_tool_names_or_capabilities() -> None:
    with pytest.raises(ValueError, match="unexpected model route field"):
        parse_model_route_output(
            {
                "route": "system_memory",
                "confidence": 0.91,
                "requires_live_state": True,
                "is_conceptual_question": False,
                "abstain": False,
                "tool_names": ["tool.system.read.resources"],
                "capability": "tool.system.read.resources",
            }
        )


def test_model_route_parser_maps_abstain_without_tool_metadata() -> None:
    result = parse_model_route_output(
        {
            "route": "unknown",
            "confidence": 0.41,
            "requires_live_state": False,
            "is_conceptual_question": False,
            "abstain": True,
        }
    )

    assert result == Abstain(
        reason_code="model_route_abstain",
        confidence=0.41,
        classification_source="model_route",
    )


def test_request_resolver_intent_classifier_preserves_abstain_source() -> None:
    classifier = RequestResolverIntentClassifier(
        resolver=StaticResolver(
            Abstain(
                reason_code="model_route_abstain",
                confidence=0.41,
                classification_source="model_route",
            )
        )
    )

    classification = asyncio.run(classifier.classify(_request()))

    assert classification.classification_source == "model_route"


def test_route_registry_keeps_project_docs_as_chat_with_rag_context() -> None:
    classification = RouteRegistry().classification_for(
        RouteDecision(RequestRoute.PROJECT_DOCS_QUESTION, 0.91, "project_docs_hint"),
        _request(user_input="Что написано в ADR-035?"),
    )

    assert classification.intent_family is IntentFamily.PROJECT_DOCS_QUESTION
    assert classification.candidate_capabilities == ()
    assert classification.fallback_preference is SelectionFallbackPreference.CHAT


def test_route_registry_maps_unknown_or_abstain_to_safe_fallback() -> None:
    classification = RouteRegistry().classification_for(
        RouteDecision(RequestRoute.UNKNOWN, 0.0, "model_route_abstain"),
        _request(),
    )

    assert classification.intent_family is IntentFamily.UNKNOWN
    assert classification.candidate_capabilities == ()
    assert classification.fallback_preference is SelectionFallbackPreference.ASK_CLARIFICATION


def test_route_registry_preserves_route_decision_source() -> None:
    classification = RouteRegistry().classification_for(
        RouteDecision(
            RequestRoute.SYSTEM_MEMORY,
            0.91,
            "model_route_system_memory",
            classification_source="model_route",
        ),
        _request(),
    )

    assert classification.classification_source == "model_route"


def test_conceptual_live_state_near_miss_does_not_map_to_tool_route() -> None:
    resolver = HybridRequestResolver()

    result = asyncio.run(resolver.resolve(_request(user_input="Объясни, что такое VPN.")))

    assert result == RouteDecision(RequestRoute.ORDINARY_CHAT, 0.76, "ordinary_chat_bypass")


def test_classifier_calibration_report_records_route_accuracy_and_latency() -> None:
    report = build_classifier_calibration_report(
        observations=[
            CalibrationObservation(
                layer_name="deterministic",
                expected_route=RequestRoute.CURRENT_TIME,
                actual_output=RouteDecision(RequestRoute.CURRENT_TIME, 0.91, "time"),
                expected_intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
                mapped_intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
                expected_requires_live_state=True,
                latency_ms=1.0,
            ),
            CalibrationObservation(
                layer_name="deterministic",
                expected_route=RequestRoute.ORDINARY_CHAT,
                actual_output=RouteDecision(RequestRoute.ORDINARY_CHAT, 0.76, "chat"),
                expected_intent_family=IntentFamily.ORDINARY_CHAT,
                mapped_intent_family=IntentFamily.ORDINARY_CHAT,
                expected_requires_live_state=False,
                latency_ms=3.0,
            ),
        ],
        threshold_candidates=(0.87, 0.9),
    )

    metrics = report.metrics_by_layer["deterministic"]
    assert metrics.route_accuracy == 1.0
    assert metrics.mapped_domain_accuracy == 1.0
    assert metrics.p50_latency_ms == 2.0
    assert metrics.p95_latency_ms == 3.0


def test_classifier_calibration_report_compares_deterministic_embedding_and_llm() -> None:
    report = build_classifier_calibration_report(
        observations=[
            _calibration_hit("deterministic", RequestRoute.CURRENT_TIME),
            _calibration_hit("embedding", RequestRoute.CURRENT_TIME),
            _calibration_hit("llm_adjudicator", RequestRoute.CURRENT_TIME, model_call_count=1),
        ],
        threshold_candidates=(0.87,),
    )

    assert set(report.metrics_by_layer) == {
        "deterministic",
        "embedding",
        "llm_adjudicator",
    }
    assert report.metrics_by_layer["llm_adjudicator"].model_call_count == 1


def test_classifier_calibration_report_records_threshold_precision_coverage() -> None:
    report = build_classifier_calibration_report(
        observations=[
            _calibration_hit("embedding", RequestRoute.CURRENT_TIME, confidence=0.91),
            _calibration_miss(
                "embedding",
                expected_route=RequestRoute.ORDINARY_CHAT,
                actual_route=RequestRoute.SYSTEM_MEMORY,
                confidence=0.88,
            ),
            CalibrationObservation(
                layer_name="embedding",
                expected_route=RequestRoute.SYSTEM_DISK,
                actual_output=Abstain("semantic_abstain", confidence=0.4),
                expected_intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                mapped_intent_family=None,
                expected_requires_live_state=True,
                latency_ms=2.0,
            ),
        ],
        threshold_candidates=(0.87, 0.9),
    )

    thresholds = report.metrics_by_layer["embedding"].thresholds
    assert thresholds[0.87].precision == 0.5
    assert thresholds[0.87].coverage == pytest.approx(2 / 3)
    assert thresholds[0.9].precision == 1.0
    assert thresholds[0.9].coverage == pytest.approx(1 / 3)
    assert report.metrics_by_layer["embedding"].abstain_rate == pytest.approx(1 / 3)


def test_classifier_calibration_report_blocks_default_model_change_on_regression() -> None:
    report = build_classifier_calibration_report(
        observations=[
            _calibration_hit("deterministic", RequestRoute.ORDINARY_CHAT, confidence=0.91),
            _calibration_miss(
                "llm_adjudicator",
                expected_route=RequestRoute.ORDINARY_CHAT,
                actual_route=RequestRoute.SYSTEM_MEMORY,
                confidence=0.91,
                expected_requires_live_state=False,
            ),
        ],
        threshold_candidates=(0.87,),
        baseline_layer="deterministic",
        candidate_default_layer="llm_adjudicator",
    )

    assert report.default_change_allowed is False
    assert "false_live_state_positive_regression" in report.default_change_blockers


class RecordingResolver:
    def __init__(self, result) -> None:
        self._result = result
        self.requests: list[LoopSelectionRequest] = []

    async def resolve(self, request: LoopSelectionRequest):
        self.requests.append(request)
        return self._result


class StaticResolver:
    def __init__(self, result) -> None:
        self._result = result

    async def resolve(self, request: LoopSelectionRequest):
        return self._result


def _request(*, user_input: str = "Сколько времени?") -> LoopSelectionRequest:
    settings = ConfigLoader(Path("config")).load("test")
    return LoopSelectionRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        user_id="user-1",
        requested_mode=LoopSelectionMode.AUTO,
        user_input=user_input,
        current_message_sensitivity=Sensitivity.PROJECT,
        active_project_namespace="project.personal_assistant",
        working_directory="/tmp/project",
        permission_mode=PermissionMode.DEVELOPER_LOCAL,
        available_capabilities=frozenset(Capability),
        available_tools_summary=CapabilityRoutingRegistry.from_settings(
            settings,
        ).available_tools_summary(),
        runtime_budget_summary={},
        metadata={"source": "test"},
    )


def _calibration_hit(
    layer_name: str,
    route: RequestRoute,
    *,
    confidence: float = 0.91,
    model_call_count: int = 0,
) -> CalibrationObservation:
    return CalibrationObservation(
        layer_name=layer_name,
        expected_route=route,
        actual_output=RouteDecision(route, confidence, f"{layer_name}_hit"),
        expected_intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        mapped_intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        expected_requires_live_state=True,
        latency_ms=2.0,
        model_call_count=model_call_count,
    )


def _calibration_miss(
    layer_name: str,
    *,
    expected_route: RequestRoute,
    actual_route: RequestRoute,
    confidence: float,
    expected_requires_live_state: bool = True,
) -> CalibrationObservation:
    return CalibrationObservation(
        layer_name=layer_name,
        expected_route=expected_route,
        actual_output=RouteDecision(actual_route, confidence, f"{layer_name}_miss"),
        expected_intent_family=IntentFamily.ORDINARY_CHAT,
        mapped_intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        expected_requires_live_state=expected_requires_live_state,
        latency_ms=4.0,
    )
