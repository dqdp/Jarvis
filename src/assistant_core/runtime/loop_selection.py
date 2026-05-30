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
        if _contains_any(text, ("temperature", "temp", "sensor")):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_SENSORS,
                reason_code="system_diagnostics_hint",
            )
        if _contains_any(text, ("cpu", "memory usage", "ram", "resources", "htop")):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_RESOURCES,
                reason_code="system_diagnostics_hint",
            )
        if _contains_any(text, ("netstat", "listening port", "listening on port", "listening on this port")):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_NETWORK,
                reason_code="system_diagnostics_hint",
            )
        if _contains_any(text, ("process", "ps ")):
            return _system_diagnostics_classification(
                capability=Capability.TOOL_SYSTEM_READ_PROCESS,
                reason_code="system_diagnostics_hint",
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
            )

        if classification.confidence < self._medium_confidence:
            return _chat_decision(
                request=request,
                intent_family=classification.intent_family,
                reason_code="classifier_low_confidence",
                confidence=classification.confidence,
                status=SelectionDecisionStatus.FALLBACK_CHAT,
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
            )

        if classification.intent_family is IntentFamily.UNKNOWN:
            return _chat_decision(
                request=request,
                intent_family=classification.intent_family,
                reason_code="unknown_intent_default_chat",
                confidence=classification.confidence,
                status=SelectionDecisionStatus.FALLBACK_CHAT,
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
            reason_code=_tool_reason(classification.intent_family),
            confidence=classification.confidence,
            candidate_capabilities=classification.candidate_capabilities,
            requires_tools=True,
            requires_live_state=classification.requires_live_state,
            policy_outcome=policy_outcome,
            approval_possible=approval_possible,
            fallback_behavior=SelectionFallbackPreference.FAIL_UNAVAILABLE,
            decision_status=SelectionDecisionStatus.SELECTED,
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


def _chat_decision(
    *,
    request: LoopSelectionRequest,
    intent_family: IntentFamily,
    reason_code: str,
    confidence: float,
    status: SelectionDecisionStatus,
    candidate_capabilities: tuple[CapabilityCandidate, ...] = (),
    requires_live_state: bool = False,
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


def _system_diagnostics_classification(
    *,
    capability: Capability,
    reason_code: str,
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
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                evidence_codes=("system_metric_request",),
            ),
        ),
        requires_live_state=True,
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
