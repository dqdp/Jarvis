from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
import math
from statistics import median
import re
from typing import Any, Protocol

from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionFallbackPreference,
)
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.ports.intent_classifier import IntentClassifierPort
from assistant_core.runtime.loop_selection import DeterministicIntentClassifier


_STABLE_LABEL = re.compile(r"^[a-z0-9_.:-]+$")
_MODEL_ROUTE_FIELDS = frozenset(
    {
        "route",
        "confidence",
        "requires_live_state",
        "is_conceptual_question",
        "abstain",
    }
)


class RequestRoute(StrEnum):
    ORDINARY_CHAT = "ordinary_chat"
    PROJECT_DOCS_QUESTION = "project_docs_question"
    PROJECT_INSPECTION = "project_inspection"
    CURRENT_TIME = "current_time"
    DATE_COUNTDOWN = "date_countdown"
    CALCULATOR = "calculator"
    DAEMON_STATUS = "daemon_status"
    SYSTEM_OS_VERSION = "system_os_version"
    SYSTEM_CPU_OVERVIEW = "system_cpu_overview"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_BATTERY = "system_battery"
    SYSTEM_TEMPERATURE = "system_temperature"
    SYSTEM_PROCESSES = "system_processes"
    SYSTEM_NETWORK = "system_network"
    SYSTEM_VPN = "system_vpn"
    UNKNOWN = "unknown"


MODEL_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": [route.value for route in RequestRoute],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "requires_live_state": {"type": "boolean"},
        "is_conceptual_question": {"type": "boolean"},
        "abstain": {"type": "boolean"},
    },
    "required": sorted(_MODEL_ROUTE_FIELDS),
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RouteDecision:
    route: RequestRoute | str
    confidence: float
    reason_code: str
    requires_live_state: bool = False
    is_conceptual_question: bool = False
    classification_source: str = "request_resolver"

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", RequestRoute(self.route))
        _require_confidence(self.confidence)
        _require_stable_label(self.reason_code, "reason_code")
        _require_stable_label(self.classification_source, "classification_source")


@dataclass(frozen=True)
class Abstain:
    reason_code: str
    confidence: float = 0.0
    classification_source: str = "request_resolver"

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        _require_stable_label(self.reason_code, "reason_code")
        _require_stable_label(self.classification_source, "classification_source")


@dataclass(frozen=True)
class Clarify:
    reason_code: str
    fallback_preference: SelectionFallbackPreference = (
        SelectionFallbackPreference.ASK_CLARIFICATION
    )
    classification_source: str = "request_resolver"

    def __post_init__(self) -> None:
        _require_stable_label(self.reason_code, "reason_code")
        _require_stable_label(self.classification_source, "classification_source")
        object.__setattr__(
            self,
            "fallback_preference",
            SelectionFallbackPreference(self.fallback_preference),
        )


@dataclass(frozen=True)
class Unavailable:
    reason_code: str
    fallback_preference: SelectionFallbackPreference = (
        SelectionFallbackPreference.FAIL_UNAVAILABLE
    )
    classification_source: str = "request_resolver"

    def __post_init__(self) -> None:
        _require_stable_label(self.reason_code, "reason_code")
        _require_stable_label(self.classification_source, "classification_source")
        object.__setattr__(
            self,
            "fallback_preference",
            SelectionFallbackPreference(self.fallback_preference),
        )


@dataclass(frozen=True)
class ModelRouteOutput:
    route: RequestRoute
    confidence: float
    requires_live_state: bool
    is_conceptual_question: bool
    abstain: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", RequestRoute(self.route))
        _require_confidence(self.confidence)


RequestResolverResult = RouteDecision | Abstain | Clarify | Unavailable


class RequestResolver(Protocol):
    async def resolve(self, request: LoopSelectionRequest) -> RequestResolverResult: ...


class RouteRegistry:
    def classification_for(
        self,
        decision: RouteDecision,
        request: LoopSelectionRequest,
    ) -> IntentClassification:
        route = decision.route
        if route is RequestRoute.ORDINARY_CHAT:
            return _chat_classification(
                intent_family=IntentFamily.ORDINARY_CHAT,
                decision=decision,
            )
        if route is RequestRoute.PROJECT_DOCS_QUESTION:
            return _chat_classification(
                intent_family=IntentFamily.PROJECT_DOCS_QUESTION,
                decision=decision,
            )
        if route is RequestRoute.PROJECT_INSPECTION:
            return _tool_classification(
                decision=decision,
                intent_family=IntentFamily.PROJECT_INSPECTION,
                capability=Capability.TOOL_SHELL_READ,
                tool_names=("tool.shell.read.project",),
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                scope_hint=None,
                evidence_codes=("project_file_lookup",),
            )
        if route is RequestRoute.CURRENT_TIME:
            return _safe_builtin_classification(
                decision=decision,
                tool_name="datetime.now",
            )
        if route is RequestRoute.DATE_COUNTDOWN:
            return _safe_builtin_classification(
                decision=decision,
                tool_name="datetime.now",
                scope_hint="christmas_countdown",
            )
        if route is RequestRoute.CALCULATOR:
            return _safe_builtin_classification(
                decision=decision,
                tool_name="calculator.evaluate",
                requires_live_state=False,
            )
        if route is RequestRoute.DAEMON_STATUS:
            return _safe_builtin_classification(
                decision=decision,
                tool_name="daemon.status",
            )
        if route is RequestRoute.SYSTEM_OS_VERSION:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                tool_names=("tool.system.read.hardware",),
                scope_hint="os_version",
                evidence_codes=("os_version_request",),
            )
        if route is RequestRoute.SYSTEM_CPU_OVERVIEW:
            return _cpu_overview_classification(decision)
        if route is RequestRoute.SYSTEM_MEMORY:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                tool_names=("tool.system.read.resources",),
                scope_hint=None,
                evidence_codes=("system_metric_request",),
            )
        if route is RequestRoute.SYSTEM_DISK:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                tool_names=("tool.system.read.resources",),
                scope_hint="disk_free",
                evidence_codes=("system_metric_request",),
            )
        if route is RequestRoute.SYSTEM_BATTERY:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                tool_names=("tool.system.read.hardware",),
                scope_hint="battery_charge",
                evidence_codes=("system_metric_request",),
            )
        if route is RequestRoute.SYSTEM_TEMPERATURE:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_SENSORS,
                tool_names=("tool.system.read.sensors",),
                scope_hint=None,
                evidence_codes=("system_metric_request",),
            )
        if route is RequestRoute.SYSTEM_PROCESSES:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_PROCESS,
                tool_names=("tool.system.read.process",),
                scope_hint=_scope_hint_for_process_route(request.user_input),
                evidence_codes=("system_metric_request",),
            )
        if route is RequestRoute.SYSTEM_NETWORK:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_NETWORK,
                tool_names=("tool.system.read.network",),
                scope_hint=None,
                evidence_codes=("system_metric_request",),
            )
        if route is RequestRoute.SYSTEM_VPN:
            return _system_classification(
                decision=decision,
                capability=Capability.TOOL_SYSTEM_READ_NETWORK,
                tool_names=("tool.system.read.network",),
                scope_hint="vpn_status",
                evidence_codes=("system_metric_request",),
            )
        return _unknown_classification(decision)


class HybridRequestResolver:
    def __init__(
        self,
        *,
        semantic_resolver: RequestResolver | None = None,
        llm_adjudicator: RequestResolver | None = None,
        enable_semantic_runtime: bool = False,
        deterministic_classifier: IntentClassifierPort | None = None,
    ) -> None:
        self._semantic_resolver = semantic_resolver
        self._llm_adjudicator = llm_adjudicator
        self._enable_semantic_runtime = enable_semantic_runtime
        self._deterministic_classifier = deterministic_classifier or DeterministicIntentClassifier()

    async def resolve(self, request: LoopSelectionRequest) -> RequestResolverResult:
        if request.requested_mode is LoopSelectionMode.CHAT:
            return RouteDecision(
                RequestRoute.ORDINARY_CHAT,
                1.0,
                "explicit_chat_mode",
            )
        if request.requested_mode is LoopSelectionMode.TOOLS:
            return Unavailable("explicit_tools_mode_requires_selector_override")

        text = request.user_input.casefold().strip()
        if _is_ambiguous_live_state_fragment(text):
            return Clarify("ambiguous_live_state_or_conceptual")
        if _is_vague_check_request(text):
            return await self._resolve_after_abstain(request, Abstain("ambiguous_request"))

        classification = await self._deterministic_classifier.classify(request)
        route = _route_from_classification(classification)
        if route is RequestRoute.UNKNOWN:
            return await self._resolve_after_abstain(
                request,
                Abstain("deterministic_abstain", confidence=classification.confidence),
            )
        if route is RequestRoute.ORDINARY_CHAT:
            return RouteDecision(
                RequestRoute.ORDINARY_CHAT,
                classification.confidence,
                "ordinary_chat_bypass",
            )
        return RouteDecision(
            route,
            classification.confidence,
            classification.reason_code,
        )

    async def _resolve_after_abstain(
        self,
        request: LoopSelectionRequest,
        abstain: Abstain,
    ) -> RequestResolverResult:
        if self._enable_semantic_runtime and self._semantic_resolver is not None:
            semantic_result = await self._semantic_resolver.resolve(request)
            if not isinstance(semantic_result, Abstain):
                return semantic_result
        if self._llm_adjudicator is not None:
            llm_result = await self._llm_adjudicator.resolve(request)
            if not isinstance(llm_result, Abstain):
                return llm_result
        return Clarify(abstain.reason_code)


class RequestResolverIntentClassifier:
    def __init__(
        self,
        *,
        resolver: RequestResolver | None = None,
        registry: RouteRegistry | None = None,
    ) -> None:
        self._resolver = resolver or HybridRequestResolver()
        self._registry = registry or RouteRegistry()

    async def classify(self, request: LoopSelectionRequest) -> IntentClassification:
        result = await self._resolver.resolve(request)
        if isinstance(result, RouteDecision):
            return self._registry.classification_for(result, request)
        if isinstance(result, Clarify | Abstain):
            return _unknown_classification(
                RouteDecision(
                    RequestRoute.UNKNOWN,
                    0.0 if isinstance(result, Clarify) else result.confidence,
                    result.reason_code,
                    classification_source=result.classification_source,
                )
            )
        return IntentClassification(
            intent_family=IntentFamily.UNKNOWN,
            confidence=0.0,
            candidate_capabilities=(),
            requires_live_state=False,
            requires_execution=False,
            answer_without_tools_would_be_misleading=False,
            reason_code=result.reason_code,
            fallback_preference=result.fallback_preference,
            classification_source=result.classification_source,
        )


@dataclass(frozen=True)
class ClassifierThresholdMetrics:
    threshold: float
    precision: float
    coverage: float


@dataclass(frozen=True)
class ClassifierLayerMetrics:
    layer_name: str
    route_accuracy: float
    mapped_domain_accuracy: float
    false_live_state_positive_rate: float
    abstain_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    model_call_count: int
    thresholds: dict[float, ClassifierThresholdMetrics]


@dataclass(frozen=True)
class HybridClassifierCalibrationReport:
    metrics_by_layer: dict[str, ClassifierLayerMetrics]
    threshold_candidates: tuple[float, ...]
    default_change_allowed: bool
    default_change_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationObservation:
    layer_name: str
    expected_route: RequestRoute | str
    actual_output: RequestResolverResult
    expected_intent_family: IntentFamily | str | None
    mapped_intent_family: IntentFamily | str | None
    expected_requires_live_state: bool
    latency_ms: float
    model_call_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_route", RequestRoute(self.expected_route))
        if self.expected_intent_family is not None:
            object.__setattr__(
                self,
                "expected_intent_family",
                IntentFamily(self.expected_intent_family),
            )
        if self.mapped_intent_family is not None:
            object.__setattr__(
                self,
                "mapped_intent_family",
                IntentFamily(self.mapped_intent_family),
            )
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.model_call_count < 0:
            raise ValueError("model_call_count must be non-negative")


def parse_model_route_output(payload: dict[str, Any]) -> RequestResolverResult:
    missing = _MODEL_ROUTE_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"missing model route field: {sorted(missing)[0]}")
    extra = set(payload).difference(_MODEL_ROUTE_FIELDS)
    if extra:
        raise ValueError(f"unexpected model route field: {sorted(extra)[0]}")
    for field_name in ("requires_live_state", "is_conceptual_question", "abstain"):
        if not isinstance(payload[field_name], bool):
            raise ValueError(f"{field_name} must be boolean")
    try:
        route = RequestRoute(payload["route"])
    except ValueError as exc:
        raise ValueError("unknown route") from exc

    confidence = float(payload["confidence"])
    output = ModelRouteOutput(
        route=route,
        confidence=confidence,
        requires_live_state=payload["requires_live_state"],
        is_conceptual_question=payload["is_conceptual_question"],
        abstain=payload["abstain"],
    )
    if output.abstain:
        return Abstain(
            "model_route_abstain",
            confidence=output.confidence,
            classification_source="model_route",
        )
    return RouteDecision(
        output.route,
        output.confidence,
        f"model_route_{output.route.value}",
        requires_live_state=output.requires_live_state,
        is_conceptual_question=output.is_conceptual_question,
        classification_source="model_route",
    )


def build_classifier_calibration_report(
    *,
    observations: list[CalibrationObservation],
    threshold_candidates: tuple[float, ...],
    baseline_layer: str | None = None,
    candidate_default_layer: str | None = None,
) -> HybridClassifierCalibrationReport:
    if not observations:
        raise ValueError("observations are required")
    if not threshold_candidates:
        raise ValueError("threshold_candidates are required")
    for threshold in threshold_candidates:
        _require_confidence(threshold)

    grouped: dict[str, list[CalibrationObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.layer_name].append(observation)

    metrics_by_layer = {
        layer_name: _layer_metrics(layer_name, layer_observations, threshold_candidates)
        for layer_name, layer_observations in sorted(grouped.items())
    }
    blockers = _default_change_blockers(
        metrics_by_layer,
        baseline_layer=baseline_layer,
        candidate_default_layer=candidate_default_layer,
    )
    return HybridClassifierCalibrationReport(
        metrics_by_layer=metrics_by_layer,
        threshold_candidates=threshold_candidates,
        default_change_allowed=not blockers,
        default_change_blockers=blockers,
    )


def _chat_classification(
    *,
    intent_family: IntentFamily,
    decision: RouteDecision,
) -> IntentClassification:
    return IntentClassification(
        intent_family=intent_family,
        confidence=decision.confidence,
        candidate_capabilities=(),
        requires_live_state=False,
        requires_execution=False,
        answer_without_tools_would_be_misleading=False,
        reason_code=decision.reason_code,
        fallback_preference=SelectionFallbackPreference.CHAT,
        classification_source=decision.classification_source,
    )


def _safe_builtin_classification(
    *,
    decision: RouteDecision,
    tool_name: str,
    scope_hint: str | None = None,
    requires_live_state: bool = True,
) -> IntentClassification:
    return _tool_classification(
        decision=decision,
        intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        capability=Capability.TOOL_SAFE,
        tool_names=(tool_name,),
        risk_classes=frozenset({RiskClass.SAFE}),
        scope_hint=scope_hint,
        evidence_codes=("safe_builtin_request",),
        requires_live_state=requires_live_state,
    )


def _system_classification(
    *,
    decision: RouteDecision,
    capability: Capability,
    tool_names: tuple[str, ...],
    scope_hint: str | None,
    evidence_codes: tuple[str, ...],
) -> IntentClassification:
    return _tool_classification(
        decision=decision,
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        capability=capability,
        tool_names=tool_names,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        scope_hint=scope_hint,
        evidence_codes=evidence_codes,
    )


def _tool_classification(
    *,
    decision: RouteDecision,
    intent_family: IntentFamily,
    capability: Capability,
    tool_names: tuple[str, ...],
    risk_classes: frozenset[RiskClass],
    scope_hint: str | None,
    evidence_codes: tuple[str, ...],
    requires_live_state: bool = True,
) -> IntentClassification:
    return IntentClassification(
        intent_family=intent_family,
        confidence=decision.confidence,
        candidate_capabilities=(
            CapabilityCandidate(
                capability=capability,
                intent_family=intent_family,
                confidence=decision.confidence,
                requires_live_state=requires_live_state,
                requires_execution=True,
                requires_write=False,
                tool_names=tool_names,
                risk_classes=risk_classes,
                scope_hint=scope_hint,
                evidence_codes=evidence_codes,
            ),
        ),
        requires_live_state=requires_live_state,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code=decision.reason_code,
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        classification_source=decision.classification_source,
    )


def _cpu_overview_classification(decision: RouteDecision) -> IntentClassification:
    candidates = (
        CapabilityCandidate(
            capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
            intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=decision.confidence,
            requires_live_state=True,
            requires_execution=True,
            requires_write=False,
            tool_names=("tool.system.read.hardware",),
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            scope_hint="cpu_overview",
            evidence_codes=("cpu_hardware_request",),
        ),
        CapabilityCandidate(
            capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
            intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
            confidence=decision.confidence,
            requires_live_state=True,
            requires_execution=True,
            requires_write=False,
            tool_names=("tool.system.read.resources",),
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            scope_hint="cpu_overview",
            evidence_codes=("cpu_load_request",),
        ),
    )
    return IntentClassification(
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        confidence=decision.confidence,
        candidate_capabilities=candidates,
        requires_live_state=True,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code=decision.reason_code,
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        classification_source=decision.classification_source,
    )


def _unknown_classification(decision: RouteDecision) -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.UNKNOWN,
        confidence=decision.confidence,
        candidate_capabilities=(),
        requires_live_state=False,
        requires_execution=False,
        answer_without_tools_would_be_misleading=False,
        reason_code=decision.reason_code,
        fallback_preference=SelectionFallbackPreference.ASK_CLARIFICATION,
        classification_source=decision.classification_source,
    )


def _route_from_classification(classification: IntentClassification) -> RequestRoute:
    if classification.intent_family is IntentFamily.ORDINARY_CHAT:
        return RequestRoute.ORDINARY_CHAT
    if classification.intent_family is IntentFamily.PROJECT_DOCS_QUESTION:
        return RequestRoute.PROJECT_DOCS_QUESTION
    if classification.intent_family is IntentFamily.PROJECT_INSPECTION:
        return RequestRoute.PROJECT_INSPECTION
    if classification.intent_family is IntentFamily.SAFE_BUILTIN_TOOL:
        return _safe_route_from_classification(classification)
    if classification.intent_family is IntentFamily.SYSTEM_DIAGNOSTICS:
        return _system_route_from_classification(classification)
    return RequestRoute.UNKNOWN


def _safe_route_from_classification(classification: IntentClassification) -> RequestRoute:
    candidate = _first_candidate(classification)
    if candidate is None:
        return RequestRoute.UNKNOWN
    if candidate.tool_names == ("calculator.evaluate",):
        return RequestRoute.CALCULATOR
    if candidate.tool_names == ("daemon.status",):
        return RequestRoute.DAEMON_STATUS
    if candidate.tool_names == ("datetime.now",):
        if candidate.scope_hint == "christmas_countdown":
            return RequestRoute.DATE_COUNTDOWN
        return RequestRoute.CURRENT_TIME
    return RequestRoute.UNKNOWN


def _system_route_from_classification(classification: IntentClassification) -> RequestRoute:
    candidate = _first_candidate(classification)
    if candidate is None:
        return RequestRoute.UNKNOWN
    if _has_cpu_overview_candidates(classification):
        return RequestRoute.SYSTEM_CPU_OVERVIEW
    if candidate.tool_names == ("tool.system.read.sensors",):
        return RequestRoute.SYSTEM_TEMPERATURE
    if candidate.tool_names == ("tool.system.read.resources",):
        if candidate.scope_hint == "disk_free":
            return RequestRoute.SYSTEM_DISK
        return RequestRoute.SYSTEM_MEMORY
    if candidate.tool_names == ("tool.system.read.hardware",):
        if candidate.scope_hint == "battery_charge":
            return RequestRoute.SYSTEM_BATTERY
        if candidate.scope_hint == "cpu_overview":
            return RequestRoute.SYSTEM_CPU_OVERVIEW
        return RequestRoute.SYSTEM_OS_VERSION
    if candidate.tool_names == ("tool.system.read.process",):
        return RequestRoute.SYSTEM_PROCESSES
    if candidate.tool_names == ("tool.system.read.network",):
        if candidate.scope_hint == "vpn_status":
            return RequestRoute.SYSTEM_VPN
        return RequestRoute.SYSTEM_NETWORK
    return RequestRoute.UNKNOWN


def _first_candidate(classification: IntentClassification) -> CapabilityCandidate | None:
    if not classification.candidate_capabilities:
        return None
    return classification.candidate_capabilities[0]


def _has_cpu_overview_candidates(classification: IntentClassification) -> bool:
    return {
        (candidate.tool_names, candidate.scope_hint)
        for candidate in classification.candidate_capabilities
    } == {
        (("tool.system.read.hardware",), "cpu_overview"),
        (("tool.system.read.resources",), "cpu_overview"),
    }


def _is_ambiguous_live_state_fragment(text: str) -> bool:
    return _normalized_fragment(text) in {
        "память",
        "memory",
        "ram",
        "vpn",
        "впн",
        "cpu",
        "процесс",
        "process",
    }


def _is_vague_check_request(text: str) -> bool:
    if not _contains_any(text, ("проверь", "check", "inspect", "посмотри")):
        return False
    return _contains_any(text, ("что-нибудь", "anything", "something"))


def _scope_hint_for_process_route(user_input: str) -> str | None:
    text = user_input.casefold()
    if _contains_any(text, ("имени", "именем", "назв", "содерж", "contains", "named")):
        return "process_name_search"
    return None


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _normalized_fragment(text: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in text).split()
    )


def _layer_metrics(
    layer_name: str,
    observations: list[CalibrationObservation],
    threshold_candidates: tuple[float, ...],
) -> ClassifierLayerMetrics:
    total = len(observations)
    route_hits = sum(_actual_route(observation) == observation.expected_route for observation in observations)
    mapped_hits = sum(
        observation.expected_intent_family == observation.mapped_intent_family
        for observation in observations
    )
    false_live_state_positives = sum(
        not observation.expected_requires_live_state and _actual_requires_live_state(observation)
        for observation in observations
    )
    abstains = sum(isinstance(observation.actual_output, Abstain) for observation in observations)
    latencies = sorted(observation.latency_ms for observation in observations)
    thresholds = {
        threshold: _threshold_metrics(threshold, observations)
        for threshold in threshold_candidates
    }
    return ClassifierLayerMetrics(
        layer_name=layer_name,
        route_accuracy=route_hits / total,
        mapped_domain_accuracy=mapped_hits / total,
        false_live_state_positive_rate=false_live_state_positives / total,
        abstain_rate=abstains / total,
        p50_latency_ms=float(median(latencies)),
        p95_latency_ms=_p95(latencies),
        model_call_count=sum(observation.model_call_count for observation in observations),
        thresholds=thresholds,
    )


def _threshold_metrics(
    threshold: float,
    observations: list[CalibrationObservation],
) -> ClassifierThresholdMetrics:
    selected = [
        observation
        for observation in observations
        if isinstance(observation.actual_output, RouteDecision)
        and observation.actual_output.confidence >= threshold
    ]
    if not selected:
        return ClassifierThresholdMetrics(threshold=threshold, precision=0.0, coverage=0.0)
    hits = sum(_actual_route(observation) == observation.expected_route for observation in selected)
    return ClassifierThresholdMetrics(
        threshold=threshold,
        precision=hits / len(selected),
        coverage=len(selected) / len(observations),
    )


def _default_change_blockers(
    metrics_by_layer: dict[str, ClassifierLayerMetrics],
    *,
    baseline_layer: str | None,
    candidate_default_layer: str | None,
) -> tuple[str, ...]:
    if baseline_layer is None or candidate_default_layer is None:
        return ()
    baseline = metrics_by_layer.get(baseline_layer)
    candidate = metrics_by_layer.get(candidate_default_layer)
    if baseline is None or candidate is None:
        return ("missing_calibration_layer",)

    blockers: list[str] = []
    if candidate.false_live_state_positive_rate > baseline.false_live_state_positive_rate:
        blockers.append("false_live_state_positive_regression")
    if candidate.route_accuracy < baseline.route_accuracy:
        blockers.append("route_accuracy_regression")
    if candidate.mapped_domain_accuracy < baseline.mapped_domain_accuracy:
        blockers.append("mapped_domain_accuracy_regression")
    return tuple(blockers)


def _actual_route(observation: CalibrationObservation) -> RequestRoute | None:
    if isinstance(observation.actual_output, RouteDecision):
        return observation.actual_output.route
    return None


def _actual_requires_live_state(observation: CalibrationObservation) -> bool:
    if not isinstance(observation.actual_output, RouteDecision):
        return False
    if observation.actual_output.requires_live_state:
        return True
    return observation.actual_output.route in {
        RequestRoute.CURRENT_TIME,
        RequestRoute.DATE_COUNTDOWN,
        RequestRoute.DAEMON_STATUS,
        RequestRoute.PROJECT_INSPECTION,
        RequestRoute.SYSTEM_OS_VERSION,
        RequestRoute.SYSTEM_CPU_OVERVIEW,
        RequestRoute.SYSTEM_MEMORY,
        RequestRoute.SYSTEM_DISK,
        RequestRoute.SYSTEM_BATTERY,
        RequestRoute.SYSTEM_TEMPERATURE,
        RequestRoute.SYSTEM_PROCESSES,
        RequestRoute.SYSTEM_NETWORK,
        RequestRoute.SYSTEM_VPN,
    }


def _p95(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    index = math.ceil(len(latencies) * 0.95) - 1
    return float(latencies[max(0, min(index, len(latencies) - 1))])


def _require_confidence(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("confidence must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("confidence must be between 0 and 1")


def _require_stable_label(value: str, field_name: str) -> None:
    if not _STABLE_LABEL.match(value):
        raise ValueError(f"{field_name} must be a stable label")
