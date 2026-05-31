from __future__ import annotations

from dataclasses import replace
from typing import Any

from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
    LoopSelectionRequest,
    SelectionFallbackPreference,
)
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import StructuredModelRequest
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.ports.intent_classifier import IntentClassifierPort
from assistant_core.ports.model_router import ModelRouterPort


INTENT_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent_family": {
            "type": "string",
            "enum": [item.value for item in IntentFamily],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "candidate_capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "intent_family": {
                        "type": "string",
                        "enum": [item.value for item in IntentFamily],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "requires_live_state": {"type": "boolean"},
                    "requires_execution": {"type": "boolean"},
                    "requires_write": {"type": "boolean"},
                    "tool_names": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[a-z0-9_.:-]+$"},
                    },
                    "risk_classes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "scope_hint": {"type": ["string", "null"], "pattern": "^[a-z0-9_.:-]+$"},
                    "evidence_codes": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[a-z0-9_.:-]+$"},
                    },
                },
                "required": [
                    "capability",
                    "intent_family",
                    "confidence",
                    "requires_live_state",
                    "requires_execution",
                    "requires_write",
                ],
                "additionalProperties": False,
            },
        },
        "requires_live_state": {"type": "boolean"},
        "requires_execution": {"type": "boolean"},
        "answer_without_tools_would_be_misleading": {"type": "boolean"},
        "reason_code": {"type": "string", "pattern": "^[a-z0-9_.:-]+$"},
        "fallback_preference": {
            "type": "string",
            "enum": [item.value for item in SelectionFallbackPreference],
        },
    },
    "required": [
        "intent_family",
        "confidence",
        "candidate_capabilities",
        "requires_live_state",
        "requires_execution",
        "answer_without_tools_would_be_misleading",
        "reason_code",
        "fallback_preference",
    ],
    "additionalProperties": False,
}


class ModelBackedIntentClassifier:
    def __init__(
        self,
        *,
        router: ModelRouterPort,
        profile: str = "local_structured",
        fallback: IntentClassifierPort | None = None,
    ) -> None:
        self._router = router
        self._profile = profile
        self._fallback = fallback

    async def classify(self, request: LoopSelectionRequest) -> IntentClassification:
        fallback_classification = await self._fallback_classification(request)
        if (
            fallback_classification is not None
            and _fallback_is_allowlisted_direct_tool_intent(fallback_classification)
        ):
            return fallback_classification
        try:
            response = await self._router.structured(
                StructuredModelRequest(
                    profile=self._profile,
                    messages=_classification_messages(request),
                    schema=INTENT_CLASSIFICATION_SCHEMA,
                    sensitivity=request.current_message_sensitivity,
                    request_id=request.request_id,
                    conversation_id=request.conversation_id,
                )
            )
            classification = _mark_model_origin(_classification_from_payload(response.value))
            return await self._guardrail_with_fallback(
                request,
                classification,
                fallback_classification=fallback_classification,
            )
        except Exception:
            if fallback_classification is not None:
                return fallback_classification
            return _classifier_unavailable()

    async def _guardrail_with_fallback(
        self,
        request: LoopSelectionRequest,
        classification: IntentClassification,
        *,
        fallback_classification: IntentClassification | None,
    ) -> IntentClassification:
        guardrailed = _local_system_state_guardrail(request, classification)
        if self._fallback is None:
            return guardrailed
        if fallback_classification is None:
            fallback_classification = await self._fallback_classification(request)
        if fallback_classification is None:
            return guardrailed
        if _fallback_overrides_model_tool_hint(guardrailed, fallback_classification):
            return fallback_classification
        if guardrailed.intent_family is not IntentFamily.ORDINARY_CHAT:
            return guardrailed
        if _fallback_is_allowlisted_direct_tool_intent(fallback_classification):
            return fallback_classification
        return guardrailed

    async def _fallback_classification(
        self,
        request: LoopSelectionRequest,
    ) -> IntentClassification | None:
        if self._fallback is None:
            return None
        try:
            fallback_classification = await self._fallback.classify(request)
        except Exception:
            return None
        return _local_system_state_guardrail(request, fallback_classification)


def _classification_messages(request: LoopSelectionRequest) -> list[ChatMessage]:
    system_text = "\n".join(
        [
            "Classify the user request for Jarvis loop selection.",
            "Return only JSON matching the provided schema.",
            "Required JSON fields: intent_family, confidence, candidate_capabilities, "
            "requires_live_state, requires_execution, answer_without_tools_would_be_misleading, "
            "reason_code, fallback_preference.",
            "candidate_capabilities items use: capability, intent_family, confidence, "
            "requires_live_state, requires_execution, requires_write, tool_names, "
            "risk_classes, scope_hint, evidence_codes.",
            "Do not output shell commands, tool arguments, executable code, or provider payloads.",
            "Candidate tool_names must be stable registry names from available metadata only.",
            "Selector and policy are authoritative; this classification only proposes intent.",
            "Requests for current local machine/system state are system_diagnostics.",
            "Local system state includes operating system version/build, CPU, memory, "
            "processes, network sockets, hardware, sensors, temperature, and daemon status.",
            "For local system state set requires_live_state=true, requires_execution=true, "
            "answer_without_tools_would_be_misleading=true, and fallback_preference=fail_unavailable.",
            "Conceptual explanations about operating systems or hardware remain ordinary_chat.",
            f"Available capabilities: {_capability_values(request)}",
            f"Available tools: {_tool_summary_values(request)}",
            f"Permission mode: {request.permission_mode.value}",
            f"Active project namespace: {request.active_project_namespace or 'none'}",
        ]
    )
    return [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[TextPart(system_text)],
            sensitivity=request.current_message_sensitivity,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=[TextPart(request.user_input)],
            sensitivity=request.current_message_sensitivity,
        ),
    ]


def _capability_values(request: LoopSelectionRequest) -> str:
    return ", ".join(
        sorted(
            value.value if hasattr(value, "value") else str(value)
            for value in request.available_capabilities
        )
    )


def _tool_summary_values(request: LoopSelectionRequest) -> str:
    if not request.available_tools_summary:
        return "none"
    lines: list[str] = []
    for item in request.available_tools_summary:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        capability = item.get("capability")
        description = item.get("description")
        if not tool_name or not capability:
            continue
        lines.append(f"{tool_name} ({capability}): {description or 'no description'}")
    return "; ".join(lines) if lines else "none"


def _classification_from_payload(payload: dict[str, Any]) -> IntentClassification:
    return IntentClassification(
        intent_family=payload["intent_family"],
        confidence=float(payload["confidence"]),
        candidate_capabilities=tuple(
            _candidate_from_payload(candidate)
            for candidate in payload.get("candidate_capabilities", [])
        ),
        requires_live_state=_required_bool(payload, "requires_live_state"),
        requires_execution=_required_bool(payload, "requires_execution"),
        answer_without_tools_would_be_misleading=_required_bool(
            payload,
            "answer_without_tools_would_be_misleading",
        ),
        reason_code=str(payload["reason_code"]),
        fallback_preference=payload["fallback_preference"],
    )


def _local_system_state_guardrail(
    request: LoopSelectionRequest,
    classification: IntentClassification,
) -> IntentClassification:
    if classification.intent_family is not IntentFamily.ORDINARY_CHAT:
        return classification
    text = request.user_input.casefold()
    if not _looks_like_local_state_request(text):
        return classification
    if _looks_like_os_or_hardware_state(text):
        candidate = _candidate_if_available(
            request,
            capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
            tool_name="tool.system.read.hardware",
            scope_hint="os_version",
            evidence_code="local_system_state_guardrail",
        )
        if candidate is not None:
            return _guardrail_classification(candidate)
    return classification


def _looks_like_local_state_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "current",
            "right now",
            "this machine",
            "this computer",
            "my machine",
            "my computer",
            "on my mac",
            "am i on",
            "installed on",
            "у меня",
            "на моем",
            "на моём",
            "на этой машине",
            "на этом компьютере",
            "сейчас",
            "текущ",
            "tengo",
            "mi mac",
            "mi equipo",
            "mi ordenador",
            "en mi sistema",
            "en este equipo",
            "sur mon",
            "sur ma",
            "mon mac",
            "mon ordinateur",
            "cet ordinateur",
            "cette machine",
            "mein mac",
            "meinem mac",
            "meinem system",
            "diesem computer",
            "auf meinem",
            "läuft auf",
        ),
    )


def _looks_like_os_or_hardware_state(text: str) -> bool:
    return _contains_any(
        text,
        (
            "operating system",
            "os ",
            "os version",
            "macos",
            "mac os",
            "build",
            "hardware",
            "system version",
            "sistema operativo",
            "versión",
            "compilación",
            "système",
            "version de macos",
            "numéro de build",
            "système d'exploitation",
            "betriebssystem",
            "macos-version",
            "buildnummer",
            "операцион",
            "версия ос",
            "сборк",
            "билд",
            "желез",
        ),
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _candidate_if_available(
    request: LoopSelectionRequest,
    *,
    capability: Capability,
    tool_name: str,
    scope_hint: str,
    evidence_code: str,
) -> CapabilityCandidate | None:
    if capability not in request.available_capabilities:
        return None
    return CapabilityCandidate(
        capability=capability,
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        confidence=0.78,
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        tool_names=(tool_name,),
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        scope_hint=scope_hint,
        evidence_codes=(evidence_code,),
    )


def _guardrail_classification(candidate: CapabilityCandidate) -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        confidence=candidate.confidence,
        candidate_capabilities=(candidate,),
        requires_live_state=True,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code="local_system_state_guardrail",
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        classification_source="guardrail",
    )


def _fallback_overrides_ordinary_chat(classification: IntentClassification) -> bool:
    if classification.intent_family is IntentFamily.ORDINARY_CHAT:
        return False
    if not classification.candidate_capabilities:
        return False
    return classification.requires_live_state or classification.requires_execution


def _fallback_overrides_model_tool_hint(
    model_classification: IntentClassification,
    fallback_classification: IntentClassification,
) -> bool:
    if model_classification.classification_source != "model":
        return False
    if fallback_classification.intent_family not in {
        IntentFamily.SAFE_BUILTIN_TOOL,
        IntentFamily.SYSTEM_DIAGNOSTICS,
    }:
        return False
    return _fallback_is_allowlisted_direct_tool_intent(fallback_classification)


def _fallback_is_allowlisted_direct_tool_intent(classification: IntentClassification) -> bool:
    if not _fallback_overrides_ordinary_chat(classification):
        return False
    return _classification_has_direct_scope(classification)


def _mark_model_origin(classification: IntentClassification) -> IntentClassification:
    if classification.classification_source == "model":
        return classification
    return replace(classification, classification_source="model")


_DIRECT_SINGLE_CANDIDATES = {
    (
        IntentFamily.SAFE_BUILTIN_TOOL,
        Capability.TOOL_SAFE,
        "datetime.now",
        None,
    ),
    (
        IntentFamily.SAFE_BUILTIN_TOOL,
        Capability.TOOL_SAFE,
        "datetime.now",
        "christmas_countdown",
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_SENSORS,
        "tool.system.read.sensors",
        None,
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_RESOURCES,
        "tool.system.read.resources",
        None,
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_RESOURCES,
        "tool.system.read.resources",
        "disk_free",
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_HARDWARE,
        "tool.system.read.hardware",
        "battery_charge",
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_HARDWARE,
        "tool.system.read.hardware",
        "os_version",
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_PROCESS,
        "tool.system.read.process",
        "process_name_search",
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_NETWORK,
        "tool.system.read.network",
        "vpn_status",
    ),
}

_DIRECT_CPU_CANDIDATES = {
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_HARDWARE,
        "tool.system.read.hardware",
        "cpu_overview",
    ),
    (
        IntentFamily.SYSTEM_DIAGNOSTICS,
        Capability.TOOL_SYSTEM_READ_RESOURCES,
        "tool.system.read.resources",
        "cpu_overview",
    ),
}


def _classification_has_direct_scope(classification: IntentClassification) -> bool:
    keys = {_direct_candidate_key(candidate) for candidate in classification.candidate_capabilities}
    if not keys or None in keys:
        return False
    return keys.issubset(_DIRECT_SINGLE_CANDIDATES) or keys == _DIRECT_CPU_CANDIDATES


def _direct_candidate_key(
    candidate: CapabilityCandidate,
) -> tuple[IntentFamily, Capability, str, str | None] | None:
    if len(candidate.tool_names) != 1:
        return None
    return (
        candidate.intent_family,
        candidate.capability,
        candidate.tool_names[0],
        candidate.scope_hint,
    )


def _candidate_from_payload(payload: dict[str, Any]) -> CapabilityCandidate:
    return CapabilityCandidate(
        capability=payload["capability"],
        intent_family=payload["intent_family"],
        confidence=float(payload["confidence"]),
        requires_live_state=_required_bool(payload, "requires_live_state"),
        requires_execution=_required_bool(payload, "requires_execution"),
        requires_write=_required_bool(payload, "requires_write"),
        tool_names=tuple(payload.get("tool_names", ())),
        risk_classes=frozenset(payload.get("risk_classes", ())),
        scope_hint=payload.get("scope_hint"),
        evidence_codes=tuple(payload.get("evidence_codes", ())),
    )


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _classifier_unavailable() -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.UNKNOWN,
        confidence=0.0,
        candidate_capabilities=(),
        requires_live_state=False,
        requires_execution=False,
        answer_without_tools_would_be_misleading=False,
        reason_code="classifier_unavailable",
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        classification_source="unavailable",
    )
