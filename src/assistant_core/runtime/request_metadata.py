from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from assistant_core.config.settings import Settings
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.loop_selection import (
    LoopSelectionDecision,
    LoopSelectionMode,
    LoopSelectionRequest,
    SelectionDecisionStatus,
)
from assistant_core.domain.loops import LoopStrategyName
from assistant_core.domain.policy import Capability, PolicyDecisionOutcome
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.intent_classifier import IntentClassifierPort
from assistant_core.ports.policy import PolicyPort
from assistant_core.runtime.direct_tools import DirectToolPlanner
from assistant_core.runtime.loop_selection import DeterministicIntentClassifier, LoopStrategySelector
from assistant_core.runtime.routing import CapabilityRoutingRegistry


@dataclass(frozen=True)
class RuntimeRequestMetadataResolution:
    metadata: dict[str, Any]
    selection_request: LoopSelectionRequest
    decision: LoopSelectionDecision


class LoopSelectionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        selection_request: LoopSelectionRequest,
        decision: LoopSelectionDecision | None = None,
    ) -> None:
        super().__init__(message)
        self.selection_request = selection_request
        self.decision = decision


async def runtime_request_metadata(
    body: Any,
    settings: Settings,
    *,
    request_id: str,
    conversation_id: str,
    user_id: str,
    active_project_namespace: str | None,
    working_directory: str | None,
    policy: PolicyPort | None,
    event_log: EventLogPort | None = None,
    intent_classifier: IntentClassifierPort | None = None,
) -> RuntimeRequestMetadataResolution:
    del event_log
    routing_registry = CapabilityRoutingRegistry.from_settings(settings)
    try:
        requested_mode = resolve_loop_selection_mode(body.loop_strategy)
    except ValueError as exc:
        selection_request = _selection_request(
            body,
            settings,
            request_id=request_id,
            conversation_id=conversation_id,
            user_id=user_id,
            active_project_namespace=active_project_namespace,
            working_directory=working_directory,
            requested_mode=LoopSelectionMode.INVALID_OVERRIDE,
            routing_registry=routing_registry,
        )
        raise LoopSelectionError(
            str(exc),
            selection_request=selection_request,
            decision=_invalid_override_decision(selection_request.requested_mode),
        ) from exc
    selection_request = _selection_request(
        body,
        settings,
        request_id=request_id,
        conversation_id=conversation_id,
        user_id=user_id,
        active_project_namespace=active_project_namespace,
        working_directory=working_directory,
        requested_mode=requested_mode,
        routing_registry=routing_registry,
    )
    selector = LoopStrategySelector(
        intent_classifier=intent_classifier or DeterministicIntentClassifier(),
        policy=policy,
        tools_enabled=settings.policy.tools_enabled,
    )
    decision = await selector.select(selection_request)
    if decision.selected_loop_strategy is None:
        raise LoopSelectionError(
            _selection_error_message(decision),
            selection_request=selection_request,
            decision=decision,
        )
    if budget_error := _selected_loop_budget_error(decision.selected_loop_strategy, settings):
        message, reason_code = budget_error
        raise LoopSelectionError(
            message,
            selection_request=selection_request,
            decision=_runtime_budget_failure_decision(decision, reason_code=reason_code),
        )

    try:
        model_profile = resolve_model_profile(
            body.model_profile,
            decision.selected_loop_strategy,
            settings,
        )
    except ValueError as exc:
        raise LoopSelectionError(
            str(exc),
            selection_request=selection_request,
            decision=_model_profile_failure_decision(decision, str(exc)),
        ) from exc

    decision = replace(decision, selected_model_profile=model_profile)
    return RuntimeRequestMetadataResolution(
        metadata=metadata_from_decision(
            decision,
            body=body,
            model_profile=model_profile,
            routing_registry=routing_registry,
        ),
        selection_request=selection_request,
        decision=decision,
    )


async def emit_loop_selection_success(
    event_log: EventLogPort | None,
    resolution: RuntimeRequestMetadataResolution,
) -> None:
    await _emit_loop_selection_started(event_log, request=resolution.selection_request)
    await _emit_loop_selection_completed(
        event_log,
        request=resolution.selection_request,
        decision=resolution.decision,
    )


async def emit_loop_selection_failure(
    event_log: EventLogPort | None,
    error: LoopSelectionError,
) -> None:
    await _emit_loop_selection_started(event_log, request=error.selection_request)
    if error.decision is not None:
        await _emit_loop_selection_failed(
            event_log,
            request=error.selection_request,
            decision=error.decision,
        )


def resolve_loop_selection_mode(requested: str | None) -> LoopSelectionMode:
    if requested is None:
        return LoopSelectionMode.AUTO
    aliases = {
        LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value: LoopSelectionMode.CHAT,
        LoopStrategyName.TOOL_REACT_LOOP.value: LoopSelectionMode.TOOLS,
    }
    if requested in aliases:
        return aliases[requested]
    if requested == LoopSelectionMode.INVALID_OVERRIDE.value:
        raise ValueError("loop strategy is not configured")
    try:
        return LoopSelectionMode(requested)
    except ValueError as exc:
        raise ValueError("loop strategy is not configured") from exc


def metadata_from_decision(
    decision: LoopSelectionDecision,
    *,
    body: Any,
    model_profile: str,
    routing_registry: CapabilityRoutingRegistry,
) -> dict[str, Any]:
    selected_loop_strategy = _loop_strategy_value(decision.selected_loop_strategy)
    metadata = {
        "requested_loop_mode": decision.requested_mode.value,
        "selected_loop_strategy": selected_loop_strategy,
        "loop_strategy": selected_loop_strategy,
        "selected_model_profile": model_profile,
        "model_profile": model_profile,
        "loop_selection_status": decision.decision_status.value,
        "loop_selection_reason_code": decision.reason_code,
        "loop_selection_classification_source": decision.classification_source,
        "loop_selection_confidence": decision.confidence,
        "loop_selection_intent_family": decision.intent_family.value,
        "loop_selection_requires_tools": decision.requires_tools,
        "loop_selection_requires_live_state": decision.requires_live_state,
        "loop_selection_policy_outcome": _policy_outcome_value(decision.policy_outcome),
        "loop_selection_approval_possible": decision.approval_possible,
    }
    tool_names = _decision_tool_names(decision, routing_registry)
    if tool_names:
        metadata["loop_selection_tool_names"] = tool_names
    direct_plan = DirectToolPlanner(routing_registry).plan(decision, user_input=body.content)
    if direct_plan is not None:
        metadata["loop_selection_direct_tool_plan"] = direct_plan.redacted_metadata()
    metadata.update(static_request_metadata(body))
    return metadata


def static_request_metadata(body: Any) -> dict[str, Any]:
    return {
        "requested_model_profile": body.model_profile,
        "working_directory": body.working_directory,
        "working_directory_scope": "provided" if body.working_directory is not None else None,
    }


def _selection_request(
    body: Any,
    settings: Settings,
    *,
    request_id: str,
    conversation_id: str,
    user_id: str,
    active_project_namespace: str | None,
    working_directory: str | None,
    requested_mode: LoopSelectionMode,
    routing_registry: CapabilityRoutingRegistry,
) -> LoopSelectionRequest:
    return LoopSelectionRequest(
        request_id=request_id,
        conversation_id=conversation_id,
        user_id=user_id,
        requested_mode=requested_mode,
        user_input=body.content,
        current_message_sensitivity=body.sensitivity,
        active_project_namespace=active_project_namespace,
        working_directory=working_directory,
        permission_mode=settings.permissions.mode,
        available_capabilities=routing_registry.available_capabilities(),
        available_tools_summary=routing_registry.available_tools_summary(),
        runtime_budget_summary=runtime_budget_summary(settings),
        model_profile_override=body.model_profile,
        metadata=static_request_metadata(body),
    )


def _invalid_override_decision(requested_mode: LoopSelectionMode) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=requested_mode,
        selected_loop_strategy=None,
        selected_model_profile=None,
        intent_family="unknown",
        reason_code="invalid_loop_selection_mode",
        confidence=1.0,
        candidate_capabilities=(),
        requires_tools=False,
        requires_live_state=False,
        policy_outcome=None,
        approval_possible=False,
        fallback_behavior="fail_unavailable",
        decision_status=SelectionDecisionStatus.INVALID_OVERRIDE,
    )


def _model_profile_failure_decision(
    decision: LoopSelectionDecision,
    message: str,
) -> LoopSelectionDecision:
    reason_code = "model_profile_unavailable"
    if "purpose" in message:
        reason_code = "model_profile_invalid_for_selected_loop"
    return replace(
        decision,
        selected_model_profile=None,
        reason_code=reason_code,
        decision_status=SelectionDecisionStatus.INVALID_OVERRIDE,
    )


def _runtime_budget_failure_decision(
    decision: LoopSelectionDecision,
    *,
    reason_code: str,
) -> LoopSelectionDecision:
    if decision.selected_loop_strategy is LoopStrategyName.TOOL_REACT_LOOP:
        status = SelectionDecisionStatus.TOOLS_UNAVAILABLE
        policy_outcome = PolicyDecisionOutcome.DENY
    else:
        status = SelectionDecisionStatus.INVALID_OVERRIDE
        policy_outcome = decision.policy_outcome
    return replace(
        decision,
        selected_loop_strategy=None,
        selected_model_profile=None,
        reason_code=reason_code,
        policy_outcome=policy_outcome,
        decision_status=status,
    )


def _selected_loop_budget_error(
    loop_strategy: LoopStrategyName,
    settings: Settings,
) -> tuple[str, str] | None:
    budget = settings.runtime_budgets.get(loop_strategy.value)
    if budget is None:
        return "loop strategy is not configured", "selected_loop_budget_unavailable"
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP and (
        not budget.allow_tools or budget.max_tool_calls <= 0
    ):
        return (
            "tool loop is not executable by runtime budget",
            "selected_tool_loop_budget_unavailable",
        )
    return None


def resolve_loop_strategy(
    requested: str | None,
    settings: Settings,
) -> LoopStrategyName:
    mode = resolve_loop_selection_mode(requested)
    if mode is LoopSelectionMode.TOOLS:
        loop_strategy = LoopStrategyName.TOOL_REACT_LOOP
    else:
        loop_strategy = LoopStrategyName.MEMORY_AUGMENTED_ANSWER
    if loop_strategy.value not in settings.runtime_budgets:
        raise ValueError("loop strategy is not configured")
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP and not settings.policy.tools_enabled:
        raise ValueError("tool loop is disabled by policy")
    return loop_strategy


def resolve_model_profile(
    requested: str | None,
    loop_strategy: LoopStrategyName,
    settings: Settings,
) -> str:
    profile_name = requested or default_model_profile(loop_strategy)
    profile = settings.model_profiles.get(profile_name)
    if profile is None or not profile.enabled or profile.cloud:
        raise ValueError("model profile is not available for this request")
    if profile.purpose != required_model_profile_purpose(loop_strategy):
        raise ValueError("model profile purpose is not valid for selected loop")
    return profile_name


def default_model_profile(loop_strategy: LoopStrategyName) -> str:
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP:
        return "local_structured"
    return "local_main"


def required_model_profile_purpose(loop_strategy: LoopStrategyName) -> str:
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP:
        return "structured"
    return "chat"


def available_capabilities(settings: Settings) -> frozenset[Capability]:
    return CapabilityRoutingRegistry.from_settings(settings).available_capabilities()


def available_tools_summary(settings: Settings) -> tuple[dict[str, Any], ...]:
    return CapabilityRoutingRegistry.from_settings(settings).available_tools_summary()


def runtime_budget_summary(settings: Settings) -> dict[str, Any]:
    return {
        name: {
            "allow_tools": budget.allow_tools,
            "allow_cloud": budget.allow_cloud,
            "max_tool_calls": budget.max_tool_calls,
            "max_model_calls": budget.max_model_calls,
        }
        for name, budget in settings.runtime_budgets.items()
    }


async def _emit_loop_selection_started(
    event_log: EventLogPort | None,
    *,
    request: LoopSelectionRequest,
) -> None:
    if event_log is None:
        return
    await event_log.append(
        _loop_selection_event(
            event_type=EventType.LOOP_SELECTION_STARTED,
            request=request,
            payload={
                "request_id": request.request_id,
                "conversation_id": request.conversation_id,
                "requested_mode": request.requested_mode.value,
            },
        ),
    )


async def _emit_loop_selection_completed(
    event_log: EventLogPort | None,
    *,
    request: LoopSelectionRequest,
    decision: LoopSelectionDecision,
) -> None:
    if event_log is None:
        return
    await event_log.append(
        _loop_selection_event(
            event_type=EventType.LOOP_SELECTION_COMPLETED,
            request=request,
            payload=decision.redacted_event_payload(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
            ),
        ),
    )


async def _emit_loop_selection_failed(
    event_log: EventLogPort | None,
    *,
    request: LoopSelectionRequest,
    decision: LoopSelectionDecision,
) -> None:
    if event_log is None:
        return
    await event_log.append(
        _loop_selection_event(
            event_type=EventType.LOOP_SELECTION_FAILED,
            request=request,
            payload=decision.redacted_event_payload(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
            ),
        ),
    )


def _loop_selection_event(
    *,
    event_type: EventType,
    request: LoopSelectionRequest,
    payload: dict[str, Any],
) -> EventEnvelope:
    now = datetime.now(UTC)
    return EventEnvelope(
        event_id=str(uuid4()),
        event_seq=0,
        event_type=event_type,
        event_version=1,
        occurred_at=now,
        recorded_at=now,
        conversation_id=request.conversation_id,
        request_id=request.request_id,
        correlation_id=request.request_id,
        causation_id=None,
        parent_event_id=None,
        actor_type=ActorType.SYSTEM,
        actor_id=request.user_id,
        source_component="runtime.request_metadata",
        source_node=None,
        sensitivity=request.current_message_sensitivity,
        visibility=EventVisibility.INTERNAL,
        idempotency_key=None,
        payload=payload,
        metadata={},
    )


def _selection_error_message(decision: LoopSelectionDecision) -> str:
    if decision.reason_code == "tools_disabled_for_tool_intent":
        return "tool loop is disabled by policy"
    if decision.decision_status is SelectionDecisionStatus.CLASSIFIER_UNAVAILABLE:
        return "loop selector is unavailable"
    if decision.decision_status is SelectionDecisionStatus.REJECTED_BY_POLICY:
        return "tool loop is rejected by policy"
    return "tool loop is unavailable for request"


def _loop_strategy_value(loop_strategy: LoopStrategyName | str | None) -> str | None:
    if loop_strategy is None:
        return None
    return loop_strategy.value if isinstance(loop_strategy, LoopStrategyName) else str(loop_strategy)


def _policy_outcome_value(outcome: PolicyDecisionOutcome | str | None) -> str | None:
    if outcome is None:
        return None
    return outcome.value if isinstance(outcome, PolicyDecisionOutcome) else str(outcome)


def _decision_tool_names(
    decision: LoopSelectionDecision,
    routing_registry: CapabilityRoutingRegistry,
) -> list[str]:
    names: list[str] = []
    for candidate in decision.candidate_capabilities:
        for tool_name in routing_registry.valid_tool_names(candidate.capability, candidate.tool_names):
            if tool_name not in names:
                names.append(tool_name)
    return names
