from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
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
from assistant_core.domain.policy import Capability, CapabilityPolicyRequest, PolicyDecisionOutcome
from assistant_core.domain.tools import SENSITIVITY_ORDER
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.policy import PolicyPort
from assistant_core.runtime.routing import CapabilityRoutingRegistry


class AgentToolPolicy(StrEnum):
    DISABLED = "disabled"
    AVAILABLE = "available"
    REQUIRED = "required"


@dataclass(frozen=True)
class AgentRequestPlan:
    requested_mode: str
    selected_loop_strategy: str
    tool_policy: AgentToolPolicy
    allowed_tool_names: tuple[str, ...]
    selected_model_profile: str
    request_plan_status: str
    request_plan_reason_code: str

    def redacted_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "requested_loop_mode": self.requested_mode,
            "selected_loop_strategy": self.selected_loop_strategy,
            "agent_tool_policy": self.tool_policy.value,
            "agent_allowed_tool_count": len(self.allowed_tool_names),
            "selected_model_profile": self.selected_model_profile,
            "model_profile": self.selected_model_profile,
            "request_plan_status": self.request_plan_status,
            "request_plan_reason_code": self.request_plan_reason_code,
        }
        if self.allowed_tool_names:
            metadata["agent_allowed_tool_names"] = list(self.allowed_tool_names)
        return metadata


@dataclass(frozen=True)
class RuntimeRequestMetadataResolution:
    metadata: dict[str, Any]
    selection_request: LoopSelectionRequest
    decision: LoopSelectionDecision
    agent_request_plan: AgentRequestPlan



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
    intent_classifier: Any | None = None,
) -> RuntimeRequestMetadataResolution:
    del event_log, intent_classifier
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
    decision = _agent_loop_decision(selection_request)
    if budget_error := _selected_loop_budget_error(
        decision.selected_loop_strategy,
        settings,
        requested_mode=decision.requested_mode,
    ):
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
    agent_request_plan = await agent_request_plan_from_decision(
        decision,
        selection_request=selection_request,
        model_profile=model_profile,
        routing_registry=routing_registry,
        settings=settings,
        policy=policy,
    )
    if decision.requested_mode is LoopSelectionMode.TOOLS and (
        agent_request_plan.tool_policy is AgentToolPolicy.DISABLED
    ):
        raise LoopSelectionError(
            _required_tools_unavailable_message(settings),
            selection_request=selection_request,
            decision=_required_tools_unavailable_decision(
                decision,
                reason_code=_required_tools_unavailable_reason(settings),
            ),
        )
    return RuntimeRequestMetadataResolution(
        metadata=metadata_from_decision(
            decision,
            body=body,
            model_profile=model_profile,
            routing_registry=routing_registry,
            agent_request_plan=agent_request_plan,
        ),
        selection_request=selection_request,
        decision=decision,
        agent_request_plan=agent_request_plan,
    )


async def emit_loop_selection_success(
    event_log: EventLogPort | None,
    resolution: RuntimeRequestMetadataResolution,
) -> None:
    await _emit_loop_selection_started(event_log, request=resolution.selection_request)
    await _emit_loop_selection_completed(
        event_log,
        request=resolution.selection_request,
        resolution=resolution,
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
    agent_request_plan: AgentRequestPlan | None = None,
) -> dict[str, Any]:
    selected_loop_strategy = _loop_strategy_value(decision.selected_loop_strategy)
    metadata = (
        agent_request_plan.redacted_metadata()
        if agent_request_plan is not None
        else {
            "requested_loop_mode": decision.requested_mode.value,
            "selected_loop_strategy": selected_loop_strategy,
            "selected_model_profile": model_profile,
            "model_profile": model_profile,
        }
    )
    metadata.update(
        {
            "requested_loop_mode": decision.requested_mode.value,
            "selected_loop_strategy": selected_loop_strategy,
            "loop_strategy": selected_loop_strategy,
            "selected_model_profile": model_profile,
            "model_profile": model_profile,
        }
    )
    metadata.update(static_request_metadata(body))
    return metadata


def _agent_loop_decision(request: LoopSelectionRequest) -> LoopSelectionDecision:
    return LoopSelectionDecision(
        requested_mode=request.requested_mode,
        selected_loop_strategy=LoopStrategyName.TOOL_REACT_LOOP,
        selected_model_profile=None,
        intent_family="unknown",
        reason_code=_agent_loop_reason_code(request.requested_mode),
        confidence=1.0,
        candidate_capabilities=(),
        requires_tools=request.requested_mode is LoopSelectionMode.TOOLS,
        requires_live_state=False,
        policy_outcome=None,
        approval_possible=False,
        fallback_behavior="fail_unavailable",
        decision_status=SelectionDecisionStatus.SELECTED,
        classification_source="request_plan",
    )


def _agent_loop_reason_code(requested_mode: LoopSelectionMode) -> str:
    if requested_mode is LoopSelectionMode.CHAT:
        return "request_plan_chat_agent_loop"
    if requested_mode is LoopSelectionMode.TOOLS:
        return "request_plan_tools_agent_loop"
    return "request_plan_auto_agent_loop"


async def agent_request_plan_from_decision(
    decision: LoopSelectionDecision,
    *,
    selection_request: LoopSelectionRequest,
    model_profile: str,
    routing_registry: CapabilityRoutingRegistry,
    settings: Settings,
    policy: PolicyPort | None,
) -> AgentRequestPlan:
    selected_loop_strategy = _loop_strategy_value(decision.selected_loop_strategy)
    if selected_loop_strategy is None:
        selected_loop_strategy = ""
    allowed_tool_names = await _agent_allowed_tool_names(
        decision,
        selection_request,
        routing_registry,
        settings=settings,
        policy=policy,
    )
    return AgentRequestPlan(
        requested_mode=decision.requested_mode.value,
        selected_loop_strategy=selected_loop_strategy,
        tool_policy=_agent_tool_policy(decision, allowed_tool_names=allowed_tool_names),
        allowed_tool_names=allowed_tool_names,
        selected_model_profile=model_profile,
        request_plan_status=decision.decision_status.value,
        request_plan_reason_code=_request_plan_reason_code(decision),
    )


def _tool_surface_available(settings: Settings) -> bool:
    if not settings.policy.tools_enabled:
        return False
    budget = settings.runtime_budgets.get(LoopStrategyName.TOOL_REACT_LOOP.value)
    if budget is None:
        return False
    return budget.allow_tools and budget.max_tool_calls > 0


def _agent_tool_policy(
    decision: LoopSelectionDecision,
    *,
    allowed_tool_names: tuple[str, ...],
) -> AgentToolPolicy:
    if decision.requested_mode is LoopSelectionMode.CHAT or not allowed_tool_names:
        return AgentToolPolicy.DISABLED
    if decision.requested_mode is LoopSelectionMode.TOOLS:
        return AgentToolPolicy.REQUIRED
    return AgentToolPolicy.AVAILABLE


async def _agent_allowed_tool_names(
    decision: LoopSelectionDecision,
    selection_request: LoopSelectionRequest,
    routing_registry: CapabilityRoutingRegistry,
    *,
    settings: Settings,
    policy: PolicyPort | None,
) -> tuple[str, ...]:
    if (
        decision.requested_mode is LoopSelectionMode.CHAT
        or not _tool_surface_available(settings)
        or policy is None
    ):
        return ()
    names: list[str] = []
    for item in routing_registry.available_tools_summary():
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name or tool_name in names:
            continue
        descriptor = routing_registry.descriptor(tool_name)
        if descriptor is None:
            continue
        if (
            SENSITIVITY_ORDER[selection_request.current_message_sensitivity]
            > SENSITIVITY_ORDER[descriptor.sensitivity_ceiling]
        ):
            continue
        policy_decision = await policy.evaluate_capability_request(
            _tool_policy_request(selection_request, descriptor)
        )
        if policy_decision.outcome is PolicyDecisionOutcome.DENY:
            continue
        names.append(tool_name)
    return tuple(names)


def _tool_policy_request(
    selection_request: LoopSelectionRequest,
    descriptor: Any,
) -> CapabilityPolicyRequest:
    return CapabilityPolicyRequest(
        capability=descriptor.capability,
        risk_classes=descriptor.risk_classes,
        sensitivity=selection_request.current_message_sensitivity,
        permission_mode=selection_request.permission_mode,
        user_id=selection_request.user_id,
        conversation_id=selection_request.conversation_id,
        request_id=selection_request.request_id,
        project_namespace=selection_request.active_project_namespace,
        working_directory=selection_request.working_directory,
        tool_name=descriptor.tool_name,
        scope={
            "intent_families": sorted(intent.value for intent in descriptor.intent_families),
        },
    )


def _request_plan_reason_code(decision: LoopSelectionDecision) -> str:
    if decision.reason_code.startswith("request_plan_"):
        return decision.reason_code
    if decision.reason_code in {
        "model_profile_unavailable",
        "model_profile_invalid_for_selected_loop",
    }:
        return f"request_plan_{decision.reason_code}"
    if decision.decision_status is SelectionDecisionStatus.SELECTED:
        return "request_plan_agent_loop_selected"
    if decision.decision_status is SelectionDecisionStatus.FALLBACK_CHAT:
        return "request_plan_chat_fallback"
    if decision.decision_status is SelectionDecisionStatus.CLARIFICATION_REQUIRED:
        return "request_plan_clarification_required"
    if decision.decision_status is SelectionDecisionStatus.REJECTED_BY_POLICY:
        return "request_plan_policy_rejected"
    if decision.decision_status is SelectionDecisionStatus.TOOLS_UNAVAILABLE:
        return "request_plan_tools_unavailable"
    if decision.decision_status is SelectionDecisionStatus.INVALID_OVERRIDE:
        return "request_plan_invalid_override"
    if decision.decision_status is SelectionDecisionStatus.CLASSIFIER_UNAVAILABLE:
        return "request_plan_intent_resolver_unavailable"
    return "request_plan_unknown"


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


def _required_tools_unavailable_decision(
    decision: LoopSelectionDecision,
    *,
    reason_code: str,
) -> LoopSelectionDecision:
    return replace(
        decision,
        selected_loop_strategy=None,
        selected_model_profile=None,
        reason_code=reason_code,
        policy_outcome=PolicyDecisionOutcome.DENY,
        decision_status=SelectionDecisionStatus.TOOLS_UNAVAILABLE,
    )


def _required_tools_unavailable_reason(settings: Settings) -> str:
    if not settings.policy.tools_enabled:
        return "tools_disabled_for_required_tool_policy"
    return "required_tools_unavailable_for_request_policy"


def _required_tools_unavailable_message(settings: Settings) -> str:
    if not settings.policy.tools_enabled:
        return "tool loop is disabled by policy"
    return "tool loop is rejected by policy"


def _selected_loop_budget_error(
    loop_strategy: LoopStrategyName,
    settings: Settings,
    *,
    requested_mode: LoopSelectionMode,
) -> tuple[str, str] | None:
    budget = settings.runtime_budgets.get(loop_strategy.value)
    if budget is None:
        return "loop strategy is not configured", "selected_loop_budget_unavailable"
    if (
        requested_mode is LoopSelectionMode.TOOLS
        and loop_strategy is LoopStrategyName.TOOL_REACT_LOOP
        and (not budget.allow_tools or budget.max_tool_calls <= 0)
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
    del loop_strategy
    return "local_main"


def required_model_profile_purpose(loop_strategy: LoopStrategyName) -> str:
    del loop_strategy
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
    resolution: RuntimeRequestMetadataResolution,
) -> None:
    if event_log is None:
        return
    await event_log.append(
        _loop_selection_event(
            event_type=EventType.LOOP_SELECTION_COMPLETED,
            request=request,
            payload=_request_plan_event_payload(
                request=request,
                decision=resolution.decision,
                agent_request_plan=resolution.agent_request_plan,
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
            payload=_request_plan_event_payload(
                request=request,
                decision=decision,
            ),
        ),
    )


def _request_plan_event_payload(
    *,
    request: LoopSelectionRequest,
    decision: LoopSelectionDecision,
    agent_request_plan: AgentRequestPlan | None = None,
) -> dict[str, Any]:
    selected_loop_strategy = _loop_strategy_value(decision.selected_loop_strategy)
    payload: dict[str, Any] = {
        "request_id": request.request_id,
        "conversation_id": request.conversation_id,
        "requested_mode": decision.requested_mode.value,
        "selected_loop_strategy": selected_loop_strategy,
        "selected_model_profile": decision.selected_model_profile,
        "request_plan_status": decision.decision_status.value,
        "request_plan_reason_code": _request_plan_reason_code(decision),
        "agent_tool_policy": (
            agent_request_plan.tool_policy.value if agent_request_plan is not None else None
        ),
        "agent_allowed_tool_count": (
            len(agent_request_plan.allowed_tool_names)
            if agent_request_plan is not None
            else None
        ),
    }
    if agent_request_plan is not None and agent_request_plan.allowed_tool_names:
        payload["agent_allowed_tool_names"] = list(agent_request_plan.allowed_tool_names)
    return payload


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
    if decision.decision_status is SelectionDecisionStatus.CLARIFICATION_REQUIRED:
        return "request needs clarification"
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
