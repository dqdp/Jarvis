from __future__ import annotations

from assistant_core.domain.loop_selection import (
    CapabilityCandidate,
    IntentClassification,
    IntentFamily,
    LoopSelectionDecision,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionDecisionStatus,
    SelectionFallbackPreference,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import (
    Capability,
    CapabilityPolicyRequest,
    PolicyDecisionOutcome,
    RiskClass,
)
from assistant_core.ports.intent_classifier import IntentClassifierPort
from assistant_core.ports.policy import PolicyPort


_TOOL_INTENTS = {
    IntentFamily.PROJECT_INSPECTION,
    IntentFamily.SYSTEM_DIAGNOSTICS,
    IntentFamily.SAFE_BUILTIN_TOOL,
}


class FakeIntentClassifier:
    def __init__(self, classification: IntentClassification) -> None:
        self._classification = classification
        self._requests: list[LoopSelectionRequest] = []

    @property
    def requests(self) -> tuple[LoopSelectionRequest, ...]:
        return tuple(self._requests)

    async def classify(self, request: LoopSelectionRequest) -> IntentClassification:
        self._requests.append(request)
        return self._classification


class DeterministicIntentClassifier:
    async def classify(self, request: LoopSelectionRequest) -> IntentClassification:
        text = request.user_input.casefold()
        if _contains_any(text, ("adr", "docs", "documentation", "roadmap")):
            return IntentClassification(
                intent_family=IntentFamily.PROJECT_DOCS_QUESTION,
                confidence=0.76,
                reason_code="project_docs_hint",
                fallback_preference=SelectionFallbackPreference.CHAT,
            )
        if _is_christmas_countdown_request(text):
            return _safe_builtin_classification(
                tool_name="datetime.now",
                reason_code="date_countdown_hint",
                scope_hint="christmas_countdown",
            )
        if _is_current_time_request(text):
            return _safe_builtin_classification(
                tool_name="datetime.now",
                reason_code="current_time_hint",
            )
        if _is_sensor_temperature_request(text):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_SENSORS,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.sensors",
            )
        if _is_cpu_overview_request(text):
            return _cpu_overview_classification()
        if _is_os_version_request(text):
            return _os_version_classification()
        if _is_memory_resource_request(text):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.resources",
            )
        if _is_disk_free_request(text):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.resources",
                scope_hint="disk_free",
            )
        if _is_battery_charge_request(text):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.hardware",
                scope_hint="battery_charge",
            )
        if _is_process_name_search_request(text):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_PROCESS,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.process",
                scope_hint="process_name_search",
            )
        if _contains_any(
            text,
            (
                "ps ",
                "show process",
                "show processes",
                "list process",
                "list processes",
                "process status",
                "running process",
                "running processes",
            ),
        ):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_PROCESS,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.process",
            )
        if not _is_conceptual_diagnostics_question(text) and _contains_any(
            text,
            ("cpu", "memory usage", "ram", "resources", "htop"),
        ):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.resources",
            )
        if _contains_any(
            text,
            ("netstat", "listening port", "listening on port", "listening on this port"),
        ):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_NETWORK,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.network",
            )
        if _is_vpn_status_request(text):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_NETWORK,
                reason_code="system_diagnostics_hint",
                tool_name="tool.system.read.network",
                scope_hint="vpn_status",
            )
        if _is_calculator_request(text):
            return _safe_builtin_classification(
                tool_name="calculator.evaluate",
                reason_code="calculator_hint",
                requires_live_state=False,
            )
        if _is_daemon_status_request(text):
            return _safe_builtin_classification(
                tool_name="daemon.status",
                reason_code="daemon_status_hint",
            )
        if _contains_any(text, ("find in project", "inspect", "grep", "rg ")) or (
            "where" in text and _contains_any(text, ("defined", "implemented", "in project"))
        ):
            return IntentClassification(
                intent_family=IntentFamily.PROJECT_INSPECTION,
                confidence=0.76,
                candidate_capabilities=(
                    CapabilityCandidate(
                        capability=Capability.TOOL_SHELL_READ,
                        intent_family=IntentFamily.PROJECT_INSPECTION,
                        confidence=0.76,
                        requires_live_state=True,
                        requires_execution=True,
                        requires_write=False,
                        tool_names=("tool.shell.read.project",),
                        risk_classes=frozenset({RiskClass.READ_ONLY}),
                        evidence_codes=("project_file_lookup",),
                    ),
                ),
                requires_live_state=True,
                requires_execution=True,
                answer_without_tools_would_be_misleading=True,
                reason_code="project_inspection_hint",
                fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
            )
        if _is_explicit_ordinary_chat_request(text):
            return IntentClassification(
                intent_family=IntentFamily.ORDINARY_CHAT,
                confidence=0.95,
                reason_code="ordinary_chat_explicit_hint",
                fallback_preference=SelectionFallbackPreference.CHAT,
            )
        return IntentClassification(
            intent_family=IntentFamily.ORDINARY_CHAT,
            confidence=0.76,
            reason_code="ordinary_chat_hint",
            fallback_preference=SelectionFallbackPreference.CHAT,
        )


class LoopStrategySelector:
    def __init__(
        self,
        *,
        intent_classifier: IntentClassifierPort,
        policy: PolicyPort | None = None,
        tools_enabled: bool = True,
        high_confidence: float = 0.75,
        medium_confidence: float = 0.45,
    ) -> None:
        self._intent_classifier = intent_classifier
        self._policy = policy
        self._tools_enabled = tools_enabled
        self._high_confidence = high_confidence
        self._medium_confidence = medium_confidence

    async def select(self, request: LoopSelectionRequest) -> LoopSelectionDecision:
        if request.requested_mode is LoopSelectionMode.CHAT:
            return _chat_decision(
                request=request,
                intent_family=IntentFamily.ORDINARY_CHAT,
                reason_code="explicit_memory_loop",
                confidence=1.0,
                status=SelectionDecisionStatus.SELECTED,
                classification_source="override",
            )
        if request.requested_mode is LoopSelectionMode.TOOLS:
            classification = _classification_for_override(
                IntentFamily.SAFE_BUILTIN_TOOL,
                candidate_capabilities=(_explicit_tools_candidate(),),
            )
            if not self._tools_enabled:
                return _tools_unavailable_decision(
                    request=request,
                    classification=classification,
                    reason_code="tools_disabled_for_tool_intent",
                )
            if unavailable_candidate := _unavailable_capability_candidate(request, classification):
                return _tools_unavailable_decision(
                    request=request,
                    classification=classification,
                    reason_code="capability_unavailable_for_tool_intent",
                    candidate_capabilities=(unavailable_candidate,),
                )
            gate = await self._tool_gate(request, classification)
            if isinstance(gate, LoopSelectionDecision):
                return gate
            policy_outcome, approval_possible = gate
            return LoopSelectionDecision(
                requested_mode=request.requested_mode,
                selected_loop_strategy=LoopStrategyName.TOOL_REACT_LOOP,
                selected_model_profile=None,
                intent_family=classification.intent_family,
                reason_code="explicit_tool_loop",
                confidence=1.0,
                candidate_capabilities=classification.candidate_capabilities,
                requires_tools=True,
                requires_live_state=True,
                policy_outcome=policy_outcome,
                approval_possible=approval_possible,
                fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
                decision_status=SelectionDecisionStatus.SELECTED,
                classification_source=classification.classification_source,
            )

        try:
            classification = await self._intent_classifier.classify(request)
        except Exception:
            return LoopSelectionDecision(
                requested_mode=request.requested_mode,
                selected_loop_strategy=None,
                selected_model_profile=None,
                intent_family=IntentFamily.UNKNOWN,
                reason_code="classifier_unavailable",
                confidence=0.0,
                candidate_capabilities=(),
                requires_tools=False,
                requires_live_state=False,
                policy_outcome=None,
                approval_possible=False,
                fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
                decision_status=SelectionDecisionStatus.CLASSIFIER_UNAVAILABLE,
                classification_source="unavailable",
            )

        if (
            classification.intent_family is IntentFamily.UNKNOWN
            and classification.fallback_preference
            is SelectionFallbackPreference.ASK_CLARIFICATION
        ):
            return _clarification_decision(request=request, classification=classification)
        if (
            classification.intent_family is IntentFamily.UNKNOWN
            and classification.fallback_preference
            is SelectionFallbackPreference.FAIL_UNAVAILABLE
        ):
            return _unknown_unavailable_decision(request=request, classification=classification)

        if classification.confidence < self._medium_confidence:
            return _chat_decision(
                request=request,
                intent_family=classification.intent_family,
                reason_code="classifier_low_confidence",
                confidence=classification.confidence,
                status=SelectionDecisionStatus.FALLBACK_CHAT,
                classification_source=classification.classification_source,
            )

        if classification.intent_family in {
            IntentFamily.ORDINARY_CHAT,
            IntentFamily.PROJECT_DOCS_QUESTION,
        }:
            return _chat_decision(
                request=request,
                intent_family=classification.intent_family,
                reason_code=_chat_reason(classification.intent_family),
                confidence=classification.confidence,
                status=SelectionDecisionStatus.SELECTED,
                classification_source=classification.classification_source,
            )

        if classification.intent_family is IntentFamily.UNKNOWN:
            return _chat_decision(
                request=request,
                intent_family=classification.intent_family,
                reason_code="unknown_intent_default_chat",
                confidence=classification.confidence,
                status=SelectionDecisionStatus.FALLBACK_CHAT,
                classification_source=classification.classification_source,
            )

        if classification.intent_family not in _TOOL_INTENTS:
            return _tools_unavailable_decision(
                request=request,
                classification=classification,
                reason_code="tools_unavailable_for_future_intent",
            )

        if not classification.candidate_capabilities:
            return _tools_unavailable_decision(
                request=request,
                classification=classification,
                reason_code="missing_capability_candidate_for_tool_intent",
            )

        if classification.confidence < self._high_confidence:
            if classification.answer_without_tools_would_be_misleading:
                return _tools_unavailable_decision(
                    request=request,
                    classification=classification,
                    reason_code="tool_intent_medium_confidence_requires_tools",
                )
            return _chat_decision(
                request=request,
                intent_family=classification.intent_family,
                reason_code="tool_intent_medium_confidence_fallback_chat",
                confidence=classification.confidence,
                status=SelectionDecisionStatus.FALLBACK_CHAT,
                candidate_capabilities=classification.candidate_capabilities,
                requires_live_state=classification.requires_live_state,
                classification_source=classification.classification_source,
            )

        if not self._tools_enabled:
            return _tools_unavailable_decision(
                request=request,
                classification=classification,
                reason_code="tools_disabled_for_tool_intent",
            )

        if unavailable_candidate := _unavailable_capability_candidate(request, classification):
            return _tools_unavailable_decision(
                request=request,
                classification=classification,
                reason_code="capability_unavailable_for_tool_intent",
                candidate_capabilities=(unavailable_candidate,),
            )

        gate = await self._tool_gate(request, classification)
        if isinstance(gate, LoopSelectionDecision):
            return gate
        policy_outcome, approval_possible = gate

        return LoopSelectionDecision(
            requested_mode=request.requested_mode,
            selected_loop_strategy=LoopStrategyName.TOOL_REACT_LOOP,
            selected_model_profile=None,
            intent_family=classification.intent_family,
            reason_code=_selected_tool_reason(classification),
            confidence=classification.confidence,
            candidate_capabilities=classification.candidate_capabilities,
            requires_tools=True,
            requires_live_state=classification.requires_live_state,
            policy_outcome=policy_outcome,
            approval_possible=approval_possible,
            fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
            decision_status=SelectionDecisionStatus.SELECTED,
            classification_source=classification.classification_source,
        )

    async def _tool_gate(
        self,
        request: LoopSelectionRequest,
        classification: IntentClassification,
    ) -> LoopSelectionDecision | tuple[PolicyDecisionOutcome, bool]:
        if self._policy is None:
            return _policy_rejected_decision(
                request=request,
                classification=classification,
                reason_code="policy_unavailable_for_tool_intent",
                policy_outcome=PolicyDecisionOutcome.DENY,
                approval_possible=False,
            )

        policy_outcome = PolicyDecisionOutcome.ALLOW
        approval_possible = False
        for candidate in classification.candidate_capabilities:
            policy_decision = await self._policy.evaluate_capability_request(
                _capability_policy_request(request, candidate)
            )
            candidate_outcome = policy_decision.outcome
            if candidate_outcome is PolicyDecisionOutcome.DENY:
                return _policy_rejected_decision(
                    request=request,
                    classification=classification,
                    reason_code=policy_decision.code,
                    policy_outcome=candidate_outcome,
                    approval_possible=False,
                )
            if candidate_outcome is PolicyDecisionOutcome.APPROVAL_REQUIRED:
                policy_outcome = PolicyDecisionOutcome.APPROVAL_REQUIRED
                approval_possible = True
        return policy_outcome, approval_possible


def _clarification_decision(
    *,
    request: LoopSelectionRequest,
    classification: IntentClassification,
) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=request.requested_mode,
        selected_loop_strategy=None,
        selected_model_profile=None,
        intent_family=classification.intent_family,
        reason_code=classification.reason_code,
        confidence=classification.confidence,
        candidate_capabilities=(),
        requires_tools=False,
        requires_live_state=False,
        policy_outcome=None,
        approval_possible=False,
        fallback_behavior=SelectionFallbackPreference.ASK_CLARIFICATION,
        decision_status=SelectionDecisionStatus.CLARIFICATION_REQUIRED,
        classification_source=classification.classification_source,
    )


def _unknown_unavailable_decision(
    *,
    request: LoopSelectionRequest,
    classification: IntentClassification,
) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=request.requested_mode,
        selected_loop_strategy=None,
        selected_model_profile=None,
        intent_family=classification.intent_family,
        reason_code=classification.reason_code,
        confidence=classification.confidence,
        candidate_capabilities=(),
        requires_tools=False,
        requires_live_state=False,
        policy_outcome=PolicyDecisionOutcome.DENY,
        approval_possible=False,
        fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        decision_status=SelectionDecisionStatus.TOOLS_UNAVAILABLE,
        classification_source=classification.classification_source,
    )


def _chat_decision(
    *,
    request: LoopSelectionRequest,
    intent_family: IntentFamily,
    reason_code: str,
    confidence: float,
    status: SelectionDecisionStatus,
    candidate_capabilities: tuple[CapabilityCandidate, ...] = (),
    requires_live_state: bool = False,
    classification_source: str = "deterministic",
) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=request.requested_mode,
        selected_loop_strategy=LoopStrategyName.MEMORY_AUGMENTED_ANSWER,
        selected_model_profile=None,
        intent_family=intent_family,
        reason_code=reason_code,
        confidence=confidence,
        candidate_capabilities=candidate_capabilities,
        requires_tools=False,
        requires_live_state=requires_live_state,
        policy_outcome=None,
        approval_possible=False,
        fallback_behavior=SelectionFallbackPreference.CHAT,
        decision_status=status,
        classification_source=classification_source,
    )


def _tools_unavailable_decision(
    *,
    request: LoopSelectionRequest,
    classification: IntentClassification,
    reason_code: str,
    candidate_capabilities: tuple[CapabilityCandidate, ...] | None = None,
) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=request.requested_mode,
        selected_loop_strategy=None,
        selected_model_profile=None,
        intent_family=classification.intent_family,
        reason_code=reason_code,
        confidence=classification.confidence,
        candidate_capabilities=(
            classification.candidate_capabilities
            if candidate_capabilities is None
            else candidate_capabilities
        ),
        requires_tools=True,
        requires_live_state=classification.requires_live_state,
        policy_outcome=PolicyDecisionOutcome.DENY,
        approval_possible=False,
        fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        decision_status=SelectionDecisionStatus.TOOLS_UNAVAILABLE,
        classification_source=classification.classification_source,
    )


def _policy_rejected_decision(
    *,
    request: LoopSelectionRequest,
    classification: IntentClassification,
    reason_code: str,
    policy_outcome: PolicyDecisionOutcome,
    approval_possible: bool,
) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=request.requested_mode,
        selected_loop_strategy=None,
        selected_model_profile=None,
        intent_family=classification.intent_family,
        reason_code=reason_code,
        confidence=classification.confidence,
        candidate_capabilities=classification.candidate_capabilities,
        requires_tools=True,
        requires_live_state=classification.requires_live_state,
        policy_outcome=policy_outcome,
        approval_possible=approval_possible,
        fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        decision_status=SelectionDecisionStatus.REJECTED_BY_POLICY,
        classification_source=classification.classification_source,
    )


def _classification_for_override(
    intent_family: IntentFamily,
    *,
    candidate_capabilities: tuple[CapabilityCandidate, ...] = (),
) -> IntentClassification:
    return IntentClassification(
        intent_family=intent_family,
        confidence=1.0,
        candidate_capabilities=candidate_capabilities,
        requires_live_state=bool(candidate_capabilities),
        requires_execution=bool(candidate_capabilities),
        answer_without_tools_would_be_misleading=bool(candidate_capabilities),
        reason_code="explicit_override",
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
        classification_source="override",
    )


def _capability_policy_request(
    request: LoopSelectionRequest,
    candidate: CapabilityCandidate,
) -> CapabilityPolicyRequest:
    return CapabilityPolicyRequest(
        capability=candidate.capability,
        risk_classes=candidate.risk_classes,
        sensitivity=request.current_message_sensitivity,
        permission_mode=request.permission_mode,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        request_id=request.request_id,
        project_namespace=request.active_project_namespace,
        working_directory=request.working_directory,
        tool_name=candidate.tool_names[0] if candidate.tool_names else None,
        scope={
            "intent_family": candidate.intent_family.value,
        },
    )


def _chat_reason(intent_family: IntentFamily) -> str:
    if intent_family is IntentFamily.PROJECT_DOCS_QUESTION:
        return "project_docs_question"
    return "ordinary_chat"


def _tool_reason(intent_family: IntentFamily) -> str:
    if intent_family is IntentFamily.PROJECT_INSPECTION:
        return "tool_intent_project_read"
    if intent_family is IntentFamily.SYSTEM_DIAGNOSTICS:
        return "tool_intent_system_diagnostics"
    return "tool_intent_safe_builtin"


def _selected_tool_reason(classification: IntentClassification) -> str:
    return _tool_reason(classification.intent_family)


def _system_diagnostics_classification(
    *,
    capability: Capability,
    reason_code: str,
    tool_name: str | None = None,
    scope_hint: str | None = None,
) -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        confidence=0.76,
        candidate_capabilities=(
            CapabilityCandidate(
                capability=capability,
                intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                confidence=0.76,
                requires_live_state=True,
                requires_execution=True,
                requires_write=False,
                tool_names=(tool_name,) if tool_name is not None else (),
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                scope_hint=scope_hint,
                evidence_codes=("system_metric_request",),
            ),
        ),
        requires_live_state=True,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code=reason_code,
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
    )


def _cpu_overview_classification() -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        confidence=0.76,
        candidate_capabilities=(
            CapabilityCandidate(
                capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                confidence=0.76,
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
                confidence=0.76,
                requires_live_state=True,
                requires_execution=True,
                requires_write=False,
                tool_names=("tool.system.read.resources",),
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                scope_hint="cpu_overview",
                evidence_codes=("cpu_load_request",),
            ),
        ),
        requires_live_state=True,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code="system_diagnostics_hint",
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
    )


def _os_version_classification() -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
        confidence=0.76,
        candidate_capabilities=(
            CapabilityCandidate(
                capability=Capability.TOOL_SYSTEM_READ_HARDWARE,
                intent_family=IntentFamily.SYSTEM_DIAGNOSTICS,
                confidence=0.76,
                requires_live_state=True,
                requires_execution=True,
                requires_write=False,
                tool_names=("tool.system.read.hardware",),
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                scope_hint="os_version",
                evidence_codes=("os_version_request",),
            ),
        ),
        requires_live_state=True,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code="system_diagnostics_hint",
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
    )


def _safe_builtin_classification(
    *,
    tool_name: str,
    reason_code: str,
    scope_hint: str | None = None,
    requires_live_state: bool = True,
) -> IntentClassification:
    return IntentClassification(
        intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        confidence=0.76,
        candidate_capabilities=(
            CapabilityCandidate(
                capability=Capability.TOOL_SAFE,
                intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
                confidence=0.76,
                requires_live_state=requires_live_state,
                requires_execution=True,
                requires_write=False,
                tool_names=(tool_name,),
                risk_classes=frozenset({RiskClass.SAFE}),
                scope_hint=scope_hint,
                evidence_codes=("safe_builtin_request",),
            ),
        ),
        requires_live_state=requires_live_state,
        requires_execution=True,
        answer_without_tools_would_be_misleading=True,
        reason_code=reason_code,
        fallback_preference=SelectionFallbackPreference.FAIL_UNAVAILABLE,
    )


def _explicit_tools_candidate() -> CapabilityCandidate:
    return CapabilityCandidate(
        capability=Capability.TOOL_SAFE,
        intent_family=IntentFamily.SAFE_BUILTIN_TOOL,
        confidence=1.0,
        requires_live_state=True,
        requires_execution=True,
        requires_write=False,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        evidence_codes=("explicit_tools_override",),
    )


def _unavailable_capability_candidate(
    request: LoopSelectionRequest,
    classification: IntentClassification,
) -> CapabilityCandidate | None:
    for candidate in classification.candidate_capabilities:
        if candidate.capability not in request.available_capabilities:
            return candidate
    return None


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _contains_temperature_signal(text: str) -> bool:
    if _contains_any(
        text,
        (
            "temperature",
            "температура",
            "температуру",
            "температуры",
        ),
    ):
        return True
    return "temp" in _word_set(text)


def _word_set(text: str) -> set[str]:
    normalized = "".join(character if character.isalnum() else " " for character in text)
    return set(normalized.split())


def _is_current_time_request(text: str) -> bool:
    if _contains_any(
        text,
        (
            "how do i",
            "how to",
            "python",
            "javascript",
            "typescript",
            "code",
            "print ",
            "program",
            "function",
            "implement",
            "format ",
        ),
    ):
        return False
    return _contains_any(
        text,
        (
            "сколько время",
            "сколько времени",
            "который час",
            "текущее время",
            "какое сейчас время",
            "сейчас времени",
            "what time is it",
            "current time",
            "what date is it",
            "current date",
            "today's date",
        ),
    )


def _is_christmas_countdown_request(text: str) -> bool:
    if "рождеств" not in text and "christmas" not in text:
        return False
    return _contains_any(
        text,
        (
            "через сколько",
            "сколько дней",
            "дней до",
            "days until",
            "how many days",
        ),
    )


def _is_calculator_request(text: str) -> bool:
    if _is_explicit_explanation_question(text):
        return False
    if not _has_arithmetic_signal(text):
        return False
    return _contains_any(
        text,
        (
            "calculate",
            "calcula",
            "calcule",
            "berechne",
            "посчитай",
            "вычисли",
            "сколько будет",
        ),
    )


def _has_arithmetic_signal(text: str) -> bool:
    if any(character.isdigit() for character in text):
        return True
    return _contains_any(
        text,
        (
            "+",
            "-",
            "*",
            "/",
            "%",
            " plus ",
            " minus ",
            " divided ",
            "multiplied",
            "процент",
            "плюс",
            "минус",
            "умнож",
            "раздел",
            "делен",
        ),
    )


def _is_daemon_status_request(text: str) -> bool:
    if _is_explicit_explanation_question(text):
        return False
    if not _contains_any(text, ("daemon", "демон", "демона")):
        return False
    return _contains_any(
        text,
        (
            "status",
            "статус",
            "estado",
            "état",
            "etat",
        ),
    )


def _is_sensor_temperature_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    if _contains_any(text, ("sensor", "sensors", "датчик", "датчики")):
        return True
    if not _contains_temperature_signal(text):
        return False
    return _contains_any(
        text,
        (
            "cpu",
            "gpu",
            "processor",
            "process",
            "core",
            "chip",
            "hardware",
            "system",
            "mac",
            "macbook",
            "laptop",
            "thermal",
            "процессор",
            "процессора",
            "цп",
            "гпу",
            "систем",
            "желез",
            "ноутбук",
            "макбук",
        ),
    )


def _is_conceptual_diagnostics_question(text: str) -> bool:
    return _contains_any(
        text,
        (
            "explain ",
            "what is ",
            "what are ",
            "how does ",
            "how do ",
            "how can ",
            "расскажи что такое",
            "объясни",
            "как работает",
            "что такое",
            "explica",
            "explique",
            "erkläre",
        ),
    )


def _is_cpu_overview_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    if not _contains_any(
        text,
        (
            "cpu",
            "processor",
            "процессор",
            "процессора",
            "центрального процессора",
            "цп",
        ),
    ):
        return False
    has_core_signal = _contains_any(
        text,
        (
            "core",
            "cores",
            "logicalcpu",
            "ядро",
            "ядер",
            "ядра",
        ),
    )
    has_load_signal = _contains_any(
        text,
        (
            "load",
            "usage",
            "utilization",
            "busy",
            "загруз",
            "загруж",
            "нагруз",
            "использ",
        ),
    )
    return has_core_signal and has_load_signal


def _is_os_version_request(text: str) -> bool:
    if _is_explicit_explanation_question(text):
        return False
    has_os_signal = _contains_any(
        text,
        (
            "operating system",
            "os version",
            "macos",
            "mac os",
            "операционной системы",
            "операционная система",
            "версии ос",
            "версия ос",
        ),
    )
    has_version_signal = _contains_any(
        text,
        (
            "version",
            "release",
            "build",
            "версия",
            "версии",
            "сборка",
            "какая",
        ),
    )
    return has_os_signal and has_version_signal


def _is_explicit_explanation_question(text: str) -> bool:
    return _contains_any(
        text,
        (
            "explain ",
            "how does ",
            "how do ",
            "how can ",
            "расскажи что такое",
            "объясни",
            "как работает",
            "что такое",
            "explica",
            "explique",
            "erkläre",
        ),
    )


def _is_explicit_ordinary_chat_request(text: str) -> bool:
    if _contains_any(
        text,
        (
            "project",
            "repo",
            "repository",
            "code",
            "file",
            "local",
            "current",
            "this system",
            "daemon",
            "process",
            "cpu",
            "memory",
            "проект",
            "репозитор",
            "код",
            "файл",
            "сейчас",
            "текущ",
            "у меня",
            "демон",
            "процесс",
            "память",
            "архитектур",
        ),
    ):
        return False
    return _contains_any(
        text,
        (
            "расскажи, как",
            "расскажи как",
            "объясни, как",
            "объясни как",
            "ответь ",
            "ответь:",
            "напиши ",
            "explain how ",
            "answer ",
            "say ",
            "tell me how ",
            "tell me about ",
        ),
    )


def _is_memory_resource_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    if not _contains_any(
        text,
        (
            "memory",
            "ram",
            "память",
            "памяти",
            "памятью",
            "оперативка",
            "оперативной памяти",
        ),
    ):
        return False
    return _contains_any(
        text,
        (
            "free",
            "available",
            "usage",
            "used",
            "свобод",
            "доступ",
            "занят",
            "использ",
            "осталось",
            "сейчас",
            "текущ",
            "в системе",
        ),
    )


def _is_disk_free_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    has_disk_signal = _contains_any(
        text,
        (
            "disk",
            "drive",
            "storage",
            "диск",
            "диске",
            "диска",
            "накопител",
            "хранилищ",
        ),
    )
    if not has_disk_signal:
        return False
    return _contains_any(
        text,
        (
            "free",
            "available",
            "space",
            "свобод",
            "доступ",
            "мест",
            "осталось",
        ),
    )


def _is_battery_charge_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    has_battery_signal = _contains_any(
        text,
        (
            "battery",
            "аккумулятор",
            "аккумулятора",
            "батаре",
            "заряд",
            "заряда",
        ),
    )
    if not has_battery_signal:
        return False
    return _contains_any(
        text,
        (
            "percent",
            "percentage",
            "%",
            "процент",
            "процентов",
            "осталось",
            "left",
            "charge",
            "заряд",
        ),
    )


def _is_process_name_search_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    has_process_signal = _contains_any(
        text,
        (
            "process",
            "processes",
            "процесс",
            "процесса",
            "процессы",
            "процессов",
        ),
    )
    if not has_process_signal:
        return False
    has_name_search_signal = _contains_any(
        text,
        (
            "name",
            "named",
            "contains",
            "matching",
            "with ",
            "имени",
            "именем",
            "назв",
            "содерж",
            "есть ",
        ),
    ) or _has_process_name_token(text)
    has_running_signal = _contains_any(
        text,
        (
            "running",
            "current",
            "is there",
            "exists",
            "запущ",
            "работает",
            "есть",
        ),
    )
    return has_name_search_signal and has_running_signal


def _has_process_name_token(text: str) -> bool:
    words = "".join(character if character.isalnum() else " " for character in text).split()
    process_words = {"process", "processes", "процесс", "процесса", "процессы", "процессов"}
    stop_words = {
        "a",
        "is",
        "now",
        "status",
        "the",
        "running",
        "есть",
        "ли",
        "сейчас",
        "статус",
    }
    for index, word in enumerate(words[:-1]):
        if word not in process_words:
            continue
        candidate = words[index + 1]
        if len(candidate) > 1 and candidate not in process_words and candidate not in stop_words:
            return True
    return False


def _is_vpn_status_request(text: str) -> bool:
    if _is_conceptual_diagnostics_question(text):
        return False
    if not _contains_any(text, ("vpn", "впн")):
        return False
    return _contains_any(
        text,
        (
            "включ",
            "подключ",
            "сейчас",
            "status",
            "connected",
            "enabled",
            "running",
            "on",
        ),
    )
