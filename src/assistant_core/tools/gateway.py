from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from assistant_core.domain.events import EventType
from assistant_core.domain.policy import CapabilityPolicyRequest, PolicyDecisionOutcome
from assistant_core.domain.tools import (
    SENSITIVITY_ORDER,
    ToolCallRequest,
    ToolObservation,
    ToolObservationStatus,
    ToolSpec,
)
from assistant_core.ports.approvals import ApprovalStorePort
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.policy import PolicyPort
from assistant_core.tools.registry import (
    ToolAdapter,
    ToolClassificationResult,
    ToolExecutionDenied,
    ToolRegistry,
)
from assistant_core.tools.approval_coordination import ToolApprovalCoordinator
from assistant_core.tools.authorization import (
    validate_arguments as _validate_arguments,
    with_effective_working_directory as _with_effective_working_directory,
)
from assistant_core.tools.audit import ToolInvocationAuditRecorder
from assistant_core.tools.events import (
    audited_tool_event_payload as _audited_tool_event_payload,
    classified_event_type as _classified_event_type,
    denied_event_type as _denied_event_type,
    is_shell_spec as _is_shell_spec,
    is_system_diagnostics_spec as _is_system_diagnostics_spec,
    tool_event_payload as _tool_event_payload,
    tool_output_sensitivity as _tool_output_sensitivity,
)
from assistant_core.tools.execution import execute_adapter as _execute_adapter
from assistant_core.tools.results import (
    empty_observation as _empty_observation,
)


class ToolGateway:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: PolicyPort,
        event_log: EventLogPort,
        approval_store: ApprovalStorePort | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._audit = ToolInvocationAuditRecorder(event_log)
        self._approval = ToolApprovalCoordinator(approval_store)

    async def list_tools(self) -> list[ToolSpec]:
        return self._registry.list_specs()

    async def get_tool(self, tool_name: str) -> ToolSpec | None:
        return self._registry.get_spec(tool_name)

    async def invoke(self, request: ToolCallRequest) -> ToolObservation:
        started_at = datetime.now(UTC)
        tool_call_id = str(uuid4())
        adapter = self._registry.get_adapter(request.tool_name, include_disabled=True)
        await self._audit.record_event(
            EventType.TOOL_CALL_REQUESTED,
            request,
            tool_call_id=tool_call_id,
            payload={"tool_name": adapter.spec.name if adapter else "<redacted>"},
        )
        if adapter is None:
            observation = _empty_observation(
                request,
                ToolObservationStatus.FAILED,
                started_at,
                tool_call_id=tool_call_id,
                tool_name="<redacted>",
                error={"code": "unknown_tool", "message": "tool is not registered"},
            )
            await self._audit.record_event(
                EventType.TOOL_CALL_FAILED,
                request,
                tool_call_id=tool_call_id,
                payload={"tool_name": "<redacted>", "error_code": "unknown_tool"},
            )
            await self._audit.record_observation(request, observation, policy_decision_id=None)
            return observation

        spec = adapter.spec
        if not spec.enabled:
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": "tool_disabled", "message": "tool is disabled"},
            )
            await self._audit.record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(spec, error_code="tool_disabled"),
            )
            await self._audit.record_observation(request, observation, policy_decision_id=None)
            return observation

        if SENSITIVITY_ORDER[request.sensitivity] > SENSITIVITY_ORDER[spec.sensitivity_ceiling]:
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": "sensitivity_ceiling_exceeded",
                    "message": "tool sensitivity ceiling is lower than request sensitivity",
                },
            )
            await self._audit.record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(spec, error_code="sensitivity_ceiling_exceeded"),
            )
            await self._audit.record_observation(request, observation, policy_decision_id=None)
            return observation

        request = _with_effective_working_directory(request, spec)
        validation_error = _validate_arguments(spec, request.arguments)
        if validation_error is not None:
            observation = _empty_observation(
                request,
                ToolObservationStatus.FAILED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": "invalid_arguments",
                    "message": "tool arguments failed validation",
                },
            )
            await self._audit.record_event(
                EventType.TOOL_CALL_FAILED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(spec, error_code="invalid_arguments"),
            )
            await self._audit.record_observation(request, observation, policy_decision_id=None)
            return observation

        if _requires_caller_working_directory(spec) and request.working_directory is None:
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": "working_directory_required",
                    "message": "working directory is required",
                },
                sensitivity=_tool_output_sensitivity(request, spec),
            )
            await self._audit.record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    spec,
                    error_code="working_directory_required",
                ),
            )
            await self._audit.record_observation(request, observation, policy_decision_id=None)
            return observation

        tool_classification = await self._classify_tool_if_supported(adapter, request)
        classified_event_type = _classified_event_type(spec)
        if tool_classification is not None and classified_event_type is not None:
            await self._audit.record_event(
                classified_event_type,
                request,
                tool_call_id=tool_call_id,
                payload=_audited_tool_event_payload(
                    spec,
                    tool_classification.metadata,
                    error_code=None if tool_classification.allowed else tool_classification.code,
                ),
                sensitivity=_tool_output_sensitivity(request, spec),
            )

        decision = await self._policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=spec.capability,
                risk_classes=spec.risk_classes,
                sensitivity=request.sensitivity,
                permission_mode=request.permission_mode,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                task_id=request.step_id,
                project_namespace=request.project_namespace,
                working_directory=request.working_directory,
                tool_name=spec.name,
                redacted_payload={
                    "tool_name": spec.name,
                    "argument_keys": sorted(request.arguments),
                },
            ),
        )

        if decision.outcome == PolicyDecisionOutcome.DENY:
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": decision.code, "message": decision.reason},
                sensitivity=_tool_output_sensitivity(request, spec),
            )
            denied_event_type = _denied_event_type(spec)
            if tool_classification is not None and denied_event_type is not None:
                await self._audit.record_event(
                    denied_event_type,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_audited_tool_event_payload(
                        spec,
                        tool_classification.metadata,
                        policy_decision_id=decision.decision_id,
                        error_code=decision.code,
                        policy_outcome=decision.outcome.value,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, spec),
                )
            await self._audit.record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    spec,
                    policy_decision_id=decision.decision_id,
                    error_code=decision.code,
                    policy_outcome=decision.outcome.value,
                ),
            )
            await self._audit.record_observation(
                request,
                observation,
                policy_decision_id=decision.decision_id,
            )
            return observation

        if tool_classification is not None and not tool_classification.allowed:
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": tool_classification.code,
                    "message": tool_classification.reason,
                },
                metadata=tool_classification.metadata,
                sensitivity=_tool_output_sensitivity(request, spec),
            )
            denied_event_type = _denied_event_type(spec)
            if denied_event_type is not None:
                await self._audit.record_event(
                    denied_event_type,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_audited_tool_event_payload(
                        spec,
                        tool_classification.metadata,
                        policy_decision_id=decision.decision_id,
                        error_code=tool_classification.code,
                        policy_outcome=decision.outcome.value,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, spec),
                )
            await self._audit.record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    spec,
                    policy_decision_id=decision.decision_id,
                    error_code=tool_classification.code,
                    policy_outcome=decision.outcome.value,
                ),
            )
            await self._audit.record_observation(
                request,
                observation,
                policy_decision_id=decision.decision_id,
            )
            return observation

        if decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED and request.approval_id:
            approved = await self._approval.validate_approval(
                request,
                spec,
                tool_call_id=tool_call_id,
            )
            if isinstance(approved, ToolObservation):
                await self._audit.record_observation(
                    request,
                    approved,
                    policy_decision_id=decision.decision_id,
                )
                return approved
            await self._audit.record_event(
                EventType.TOOL_CALL_APPROVED,
                request,
                tool_call_id=tool_call_id,
                payload={
                    **_tool_event_payload(spec, policy_decision_id=decision.decision_id),
                    "approval_id": request.approval_id,
                },
            )
            await self._audit.record_event(
                EventType.TOOL_CALL_STARTED,
                request,
                tool_call_id=tool_call_id,
                payload={
                    **_tool_event_payload(spec, policy_decision_id=decision.decision_id),
                    "approval_id": request.approval_id,
                },
            )
            return await _execute_adapter(
                request=request,
                adapter=adapter,
                tool_call_id=tool_call_id,
                started_at=started_at,
                policy_decision_id=decision.decision_id,
                record_event=self._audit.record_event,
                record_observation=self._audit.record_observation,
                tool_metadata=(
                    tool_classification.metadata
                    if tool_classification is not None
                    else None
                ),
                policy_outcome=decision.outcome.value,
            )

        if decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED:
            approval_metadata = await self._approval.create_metadata(
                request,
                spec,
                started_at=started_at,
                policy_decision_id=decision.decision_id,
            )
            observation = _empty_observation(
                request,
                ToolObservationStatus.APPROVAL_REQUIRED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": decision.code, "message": decision.reason},
                metadata=approval_metadata,
                sensitivity=_tool_output_sensitivity(request, spec),
            )
            await self._audit.record_observation(
                request,
                observation,
                policy_decision_id=decision.decision_id,
            )
            return observation

        await self._audit.record_event(
            EventType.TOOL_CALL_STARTED,
            request,
            tool_call_id=tool_call_id,
            payload=_tool_event_payload(spec, policy_decision_id=decision.decision_id),
        )
        return await _execute_adapter(
            request=request,
            adapter=adapter,
            tool_call_id=tool_call_id,
            started_at=started_at,
            policy_decision_id=decision.decision_id,
            record_event=self._audit.record_event,
            record_observation=self._audit.record_observation,
            tool_metadata=(
                tool_classification.metadata if tool_classification is not None else None
            ),
            policy_outcome=decision.outcome.value,
        )

    async def _classify_tool_if_supported(
        self,
        adapter: ToolAdapter,
        request: ToolCallRequest,
    ) -> ToolClassificationResult | None:
        classifier = getattr(adapter, "classify", None)
        if classifier is None:
            return None
        result = classifier(request.arguments)
        if asyncio.iscoroutine(result):
            result = await result
        if not isinstance(result, ToolClassificationResult):
            raise TypeError("tool classifier must return ToolClassificationResult")
        return result


def _requires_caller_working_directory(spec: ToolSpec) -> bool:
    return _is_shell_spec(spec) or _is_system_diagnostics_spec(spec)
