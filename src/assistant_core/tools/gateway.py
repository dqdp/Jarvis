from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from assistant_core.domain.approvals import (
    ApprovalConflict,
    ApprovalNotFound,
    ApprovalScope,
    CreateApprovalCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.policy import Capability, CapabilityPolicyRequest, PolicyDecisionOutcome
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import (
    SENSITIVITY_ORDER,
    ToolCallRequest,
    ToolInvocationResult,
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
        self._event_log = event_log
        self._approval_store = approval_store

    async def list_tools(self) -> list[ToolSpec]:
        return self._registry.list_specs()

    async def get_tool(self, tool_name: str) -> ToolSpec | None:
        return self._registry.get_spec(tool_name)

    async def invoke(self, request: ToolCallRequest) -> ToolObservation:
        started_at = datetime.now(UTC)
        tool_call_id = str(uuid4())
        adapter = self._registry.get_adapter(request.tool_name, include_disabled=True)
        await self._record_event(
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
            await self._record_event(
                EventType.TOOL_CALL_FAILED,
                request,
                tool_call_id=tool_call_id,
                payload={"tool_name": "<redacted>", "error_code": "unknown_tool"},
            )
            await self._record_observation(request, observation, policy_decision_id=None)
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
            await self._record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(spec, error_code="tool_disabled"),
            )
            await self._record_observation(request, observation, policy_decision_id=None)
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
            await self._record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(spec, error_code="sensitivity_ceiling_exceeded"),
            )
            await self._record_observation(request, observation, policy_decision_id=None)
            return observation

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
            await self._record_event(
                EventType.TOOL_CALL_FAILED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(spec, error_code="invalid_arguments"),
            )
            await self._record_observation(request, observation, policy_decision_id=None)
            return observation

        request = _with_effective_working_directory(request, spec)
        shell_classification = await self._classify_tool_if_supported(adapter, request)
        if shell_classification is not None:
            await self._record_event(
                EventType.TOOL_SHELL_CLASSIFIED,
                request,
                tool_call_id=tool_call_id,
                payload=_shell_event_payload(
                    spec,
                    shell_classification.metadata,
                    error_code=None if shell_classification.allowed else shell_classification.code,
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
            if shell_classification is not None:
                await self._record_event(
                    EventType.TOOL_SHELL_DENIED,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_shell_event_payload(
                        spec,
                        shell_classification.metadata,
                        policy_decision_id=decision.decision_id,
                        error_code=decision.code,
                        policy_outcome=decision.outcome.value,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, spec),
                )
            await self._record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    spec,
                    policy_decision_id=decision.decision_id,
                ),
            )
            await self._record_observation(
                request,
                observation,
                policy_decision_id=decision.decision_id,
            )
            return observation

        if shell_classification is not None and not shell_classification.allowed:
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": shell_classification.code,
                    "message": shell_classification.reason,
                },
                metadata=shell_classification.metadata,
                sensitivity=_tool_output_sensitivity(request, spec),
            )
            await self._record_event(
                EventType.TOOL_SHELL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_shell_event_payload(
                    spec,
                    shell_classification.metadata,
                    policy_decision_id=decision.decision_id,
                    error_code=shell_classification.code,
                    policy_outcome=decision.outcome.value,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=_tool_output_sensitivity(request, spec),
            )
            await self._record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    spec,
                    policy_decision_id=decision.decision_id,
                    error_code=shell_classification.code,
                ),
            )
            await self._record_observation(
                request,
                observation,
                policy_decision_id=decision.decision_id,
            )
            return observation

        if decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED and request.approval_id:
            approved = await self._validate_approval(
                request,
                spec,
                tool_call_id=tool_call_id,
                policy_decision_id=decision.decision_id,
            )
            if isinstance(approved, ToolObservation):
                await self._record_observation(
                    request,
                    approved,
                    policy_decision_id=decision.decision_id,
                )
                return approved
            await self._record_event(
                EventType.TOOL_CALL_APPROVED,
                request,
                tool_call_id=tool_call_id,
                payload={
                    **_tool_event_payload(spec, policy_decision_id=decision.decision_id),
                    "approval_id": request.approval_id,
                },
            )
            await self._record_event(
                EventType.TOOL_CALL_STARTED,
                request,
                tool_call_id=tool_call_id,
                payload={
                    **_tool_event_payload(spec, policy_decision_id=decision.decision_id),
                    "approval_id": request.approval_id,
                },
            )
            return await self._execute_adapter(
                request,
                adapter,
                tool_call_id,
                started_at,
                policy_decision_id=decision.decision_id,
                shell_metadata=(
                    shell_classification.metadata
                    if shell_classification is not None
                    else None
                ),
                policy_outcome=decision.outcome.value,
            )

        if decision.outcome == PolicyDecisionOutcome.APPROVAL_REQUIRED:
            approval_metadata = await self._create_approval_metadata(
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
            await self._record_observation(
                request,
                observation,
                policy_decision_id=decision.decision_id,
            )
            return observation

        await self._record_event(
            EventType.TOOL_CALL_STARTED,
            request,
            tool_call_id=tool_call_id,
            payload=_tool_event_payload(spec, policy_decision_id=decision.decision_id),
        )
        return await self._execute_adapter(
            request,
            adapter,
            tool_call_id,
            started_at,
            policy_decision_id=decision.decision_id,
            shell_metadata=(
                shell_classification.metadata if shell_classification is not None else None
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

    async def _create_approval_metadata(
        self,
        request: ToolCallRequest,
        spec: ToolSpec,
        *,
        started_at: datetime,
        policy_decision_id: str,
    ) -> dict[str, Any]:
        if self._approval_store is None:
            return {}
        approval = await self._approval_store.create_approval(
            CreateApprovalCommand(
                scope=_approval_scope(spec, request),
                redacted_payload=_approval_payload(spec, request),
                requested_by=request.user_id,
                created_at=started_at,
                metadata={
                    "causation_event_id": request.causation_event_id,
                    "policy_decision_id": policy_decision_id,
                },
            ),
        )
        return {
            "approval_id": approval.approval_id,
            "status": approval.status.value,
            "expires_at": approval.expires_at.isoformat(),
        }

    async def _validate_approval(
        self,
        request: ToolCallRequest,
        spec: ToolSpec,
        *,
        tool_call_id: str,
        policy_decision_id: str,
    ) -> ToolObservation | None:
        started_at = datetime.now(UTC)
        if self._approval_store is None:
            return _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": "approval_store_unavailable",
                    "message": "approval store is not configured",
                },
                sensitivity=_tool_output_sensitivity(request, spec),
            )
        try:
            await self._approval_store.consume_granted_approval(
                request.approval_id or "",
                scope=_approval_scope(spec, request),
            )
        except ApprovalNotFound:
            return _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": "approval_not_found", "message": "approval not found"},
                sensitivity=_tool_output_sensitivity(request, spec),
            )
        except ApprovalConflict as exc:
            return _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": exc.code, "message": str(exc)},
                sensitivity=_tool_output_sensitivity(request, spec),
            )
        return None

    async def _execute_adapter(
        self,
        request: ToolCallRequest,
        adapter: ToolAdapter,
        tool_call_id: str,
        started_at: datetime,
        policy_decision_id: str,
        shell_metadata: dict[str, Any] | None = None,
        policy_outcome: str | None = None,
    ) -> ToolObservation:
        timeout_seconds = _effective_timeout(request, adapter.spec)
        max_output_bytes = _effective_max_output(request, adapter.spec)
        try:
            if shell_metadata is not None:
                await self._record_event(
                    EventType.TOOL_SHELL_STARTED,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_shell_event_payload(
                        adapter.spec,
                        shell_metadata,
                        policy_decision_id=policy_decision_id,
                        policy_outcome=policy_outcome,
                    ),
                    sensitivity=_tool_output_sensitivity(request, adapter.spec),
                )
            result = await asyncio.wait_for(
                adapter.invoke(request.arguments),
                timeout=timeout_seconds,
            )
        except ToolExecutionDenied as exc:
            metadata = shell_metadata or exc.metadata
            observation = _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": exc.code, "message": exc.message},
                metadata=metadata,
                sensitivity=_tool_output_sensitivity(request, adapter.spec),
            )
            if _is_shell_spec(adapter.spec):
                await self._record_event(
                    EventType.TOOL_SHELL_DENIED,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_shell_event_payload(
                        adapter.spec,
                        metadata,
                        policy_decision_id=policy_decision_id,
                        error_code=exc.code,
                        policy_outcome=policy_outcome,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, adapter.spec),
                )
            await self._record_event(
                EventType.TOOL_CALL_DENIED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    adapter.spec,
                    policy_decision_id=policy_decision_id,
                    error_code=exc.code,
                ),
            )
            await self._record_observation(
                request,
                observation,
                policy_decision_id=policy_decision_id,
            )
            return observation
        except TimeoutError:
            observation = _empty_observation(
                request,
                ToolObservationStatus.TIMEOUT,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": "tool_timeout", "message": "tool execution timed out"},
                sensitivity=_tool_output_sensitivity(request, adapter.spec),
            )
            await self._record_event(
                EventType.TOOL_CALL_TIMEOUT,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    adapter.spec,
                    policy_decision_id=policy_decision_id,
                ),
            )
            if _is_shell_spec(adapter.spec):
                await self._record_event(
                    EventType.TOOL_SHELL_TIMEOUT,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_shell_event_payload(
                        adapter.spec,
                        shell_metadata or {},
                        policy_decision_id=policy_decision_id,
                        error_code="tool_timeout",
                        policy_outcome=policy_outcome,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, adapter.spec),
                )
            await self._record_observation(
                request,
                observation,
                policy_decision_id=policy_decision_id,
            )
            return observation
        except Exception as exc:  # noqa: BLE001 - normalize adapter failures.
            observation = _empty_observation(
                request,
                ToolObservationStatus.FAILED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": "tool_failed", "message": "tool execution failed"},
                sensitivity=_tool_output_sensitivity(request, adapter.spec),
            )
            await self._record_event(
                EventType.TOOL_CALL_FAILED,
                request,
                tool_call_id=tool_call_id,
                payload=_tool_event_payload(
                    adapter.spec,
                    policy_decision_id=policy_decision_id,
                    error_code="tool_failed",
                ),
            )
            if _is_shell_spec(adapter.spec):
                await self._record_event(
                    EventType.TOOL_SHELL_FAILED,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_shell_event_payload(
                        adapter.spec,
                        shell_metadata or {},
                        policy_decision_id=policy_decision_id,
                        error_code="tool_failed",
                        policy_outcome=policy_outcome,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, adapter.spec),
                )
            await self._record_observation(
                request,
                observation,
                policy_decision_id=policy_decision_id,
            )
            return observation

        observation = _completed_observation(
            request=request,
            adapter=adapter,
            tool_call_id=tool_call_id,
            started_at=started_at,
            result=result,
            max_output_bytes=max_output_bytes,
        )
        await self._record_event(
            EventType.TOOL_CALL_COMPLETED,
            request,
            tool_call_id=tool_call_id,
            payload={
                **_tool_event_payload(adapter.spec, policy_decision_id=policy_decision_id),
                "truncated": observation.truncated,
                "output_bytes": observation.output_bytes,
            },
        )
        if _is_shell_spec(adapter.spec):
            await self._record_event(
                EventType.TOOL_SHELL_COMPLETED,
                request,
                tool_call_id=tool_call_id,
                payload=_shell_event_payload(
                    adapter.spec,
                    {
                        **(shell_metadata or {}),
                        **observation.metadata,
                    },
                    policy_decision_id=policy_decision_id,
                    policy_outcome=policy_outcome,
                    observation=observation,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=_tool_output_sensitivity(request, adapter.spec),
            )
            if observation.truncated:
                await self._record_event(
                    EventType.TOOL_SHELL_OUTPUT_TRUNCATED,
                    request,
                    tool_call_id=tool_call_id,
                    payload=_shell_event_payload(
                        adapter.spec,
                        {
                            **(shell_metadata or {}),
                            **observation.metadata,
                        },
                        policy_decision_id=policy_decision_id,
                        policy_outcome=policy_outcome,
                        observation=observation,
                        duration_ms=observation.duration_ms,
                    ),
                    sensitivity=_tool_output_sensitivity(request, adapter.spec),
                )
        await self._record_observation(
            request,
            observation,
            policy_decision_id=policy_decision_id,
        )
        return observation

    async def _record_observation(
        self,
        request: ToolCallRequest,
        observation: ToolObservation,
        *,
        policy_decision_id: str | None,
    ) -> None:
        await self._record_event(
            EventType.TOOL_OBSERVATION_RECORDED,
            request,
            tool_call_id=observation.tool_call_id,
            payload={
                "tool_name": observation.tool_name,
                "policy_decision_id": policy_decision_id,
                "status": observation.status.value,
                "truncated": observation.truncated,
                "output_bytes": observation.output_bytes,
                "error_code": observation.error["code"] if observation.error else None,
            },
            sensitivity=observation.sensitivity,
        )

    async def _record_event(
        self,
        event_type: EventType,
        request: ToolCallRequest,
        *,
        tool_call_id: str,
        payload: dict[str, Any],
        sensitivity: Sensitivity | None = None,
    ) -> None:
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=event_type,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                correlation_id=request.correlation_id or request.request_id,
                causation_id=request.causation_event_id,
                parent_event_id=None,
                actor_type=ActorType.TOOL,
                actor_id=request.user_id,
                source_component="tool_gateway",
                source_node=None,
                sensitivity=sensitivity or request.sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=request.idempotency_key,
                payload={"tool_call_id": tool_call_id, "step_id": request.step_id, **payload},
                metadata={},
            ),
        )


def _empty_observation(
    request: ToolCallRequest,
    status: ToolObservationStatus,
    started_at: datetime,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    sensitivity: Sensitivity | None = None,
) -> ToolObservation:
    completed_at = datetime.now(UTC)
    return ToolObservation(
        tool_call_id=tool_call_id or str(uuid4()),
        tool_name=tool_name or request.tool_name,
        status=status,
        content="",
        content_type="text/plain",
        sensitivity=sensitivity or request.sensitivity,
        truncated=False,
        output_bytes=0,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
        error=error,
        metadata=metadata or {},
    )


def _completed_observation(
    *,
    request: ToolCallRequest,
    adapter: ToolAdapter,
    tool_call_id: str,
    started_at: datetime,
    result: Any,
    max_output_bytes: int,
) -> ToolObservation:
    content, content_type, metadata, result_truncated, result_output_bytes = _serialize_content(
        result,
        adapter.content_type,
    )
    content = _redact_content(content, content_type=content_type)
    encoded = content.encode("utf-8")
    output_bytes = result_output_bytes if result_output_bytes is not None else len(encoded)
    gateway_truncated = len(encoded) > max_output_bytes
    truncated = result_truncated or gateway_truncated
    if gateway_truncated:
        content = encoded[:max_output_bytes].decode("utf-8", errors="ignore")
    completed_at = datetime.now(UTC)
    return ToolObservation(
        tool_call_id=tool_call_id,
        tool_name=request.tool_name,
        status=ToolObservationStatus.COMPLETED,
        content=content,
        content_type=content_type,
        sensitivity=_tool_output_sensitivity(request, adapter.spec),
        truncated=truncated,
        output_bytes=output_bytes,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
        error=None,
        metadata=metadata,
    )


def _redact_content(content: str, *, content_type: str) -> str:
    if not _looks_sensitive(content):
        return content
    if content_type == "application/json":
        return json.dumps({"redacted": True}, sort_keys=True)
    return "<redacted>"


def _serialize_content(
    result: Any,
    content_type: str,
) -> tuple[str, str, dict[str, Any], bool, int | None]:
    if isinstance(result, ToolInvocationResult):
        return (
            result.content,
            result.content_type,
            result.metadata,
            result.truncated,
            result.output_bytes,
        )
    if isinstance(result, str):
        return result, content_type, {}, False, None
    return json.dumps(result, sort_keys=True), "application/json", {}, False, None


def _effective_timeout(request: ToolCallRequest, spec: ToolSpec) -> float:
    if request.timeout_seconds is None:
        return spec.default_timeout_seconds
    return min(request.timeout_seconds, spec.default_timeout_seconds)


def _effective_max_output(request: ToolCallRequest, spec: ToolSpec) -> int:
    if request.max_output_bytes is None:
        return spec.max_output_bytes
    return min(request.max_output_bytes, spec.max_output_bytes)


def _tool_event_payload(
    spec: ToolSpec,
    *,
    policy_decision_id: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    payload = {
        "tool_name": spec.name,
        "capability": spec.capability.value,
        "risk_classes": sorted(risk.value for risk in spec.risk_classes),
        "policy_decision_id": policy_decision_id,
    }
    if error_code is not None:
        payload["error_code"] = error_code
    return payload


def _shell_event_payload(
    spec: ToolSpec,
    shell_metadata: dict[str, Any],
    *,
    policy_decision_id: str | None = None,
    error_code: str | None = None,
    policy_outcome: str | None = None,
    duration_ms: int | None = None,
    observation: ToolObservation | None = None,
) -> dict[str, Any]:
    safe_metadata = {
        key: value
        for key, value in shell_metadata.items()
        if key
        in {
            "argv",
            "cwd",
            "exit_code",
            "family",
            "raw_stderr_bytes",
            "raw_stdout_bytes",
            "stderr_truncated",
            "stdout_truncated",
        }
    }
    payload: dict[str, Any] = {
        **_tool_event_payload(
            spec,
            policy_decision_id=policy_decision_id,
            error_code=error_code,
        ),
        **safe_metadata,
    }
    if observation is not None:
        payload["truncated"] = observation.truncated
        payload["output_bytes"] = observation.output_bytes
    if policy_outcome is not None:
        payload["policy_outcome"] = policy_outcome
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return payload


def _tool_output_sensitivity(request: ToolCallRequest, spec: ToolSpec) -> Sensitivity:
    return _max_sensitivity(request.sensitivity, _tool_output_sensitivity_floor(spec))


def _tool_output_sensitivity_floor(spec: ToolSpec) -> Sensitivity:
    if spec.capability == Capability.TOOL_SHELL_READ:
        return Sensitivity.PROJECT
    return Sensitivity.PUBLIC


def _max_sensitivity(first: Sensitivity, second: Sensitivity) -> Sensitivity:
    return first if SENSITIVITY_ORDER[first] >= SENSITIVITY_ORDER[second] else second


def _is_shell_spec(spec: ToolSpec) -> bool:
    return spec.capability == Capability.TOOL_SHELL_READ


def _with_effective_working_directory(
    request: ToolCallRequest,
    spec: ToolSpec,
) -> ToolCallRequest:
    if request.working_directory is not None or spec.capability != Capability.TOOL_SHELL_READ:
        return request
    cwd = request.arguments.get("cwd")
    if isinstance(cwd, str) and cwd:
        return replace(request, working_directory=cwd)
    return request


def _approval_scope(spec: ToolSpec, request: ToolCallRequest) -> ApprovalScope:
    return ApprovalScope(
        capability=spec.capability,
        risk_classes=spec.risk_classes,
        tool_name=spec.name,
        user_id=request.user_id,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        step_id=request.step_id,
        project_namespace=request.project_namespace,
        working_directory=request.working_directory,
        sensitivity=request.sensitivity,
        permission_mode=request.permission_mode,
        argument_keys=tuple(sorted(request.arguments)),
        arguments_hash=_arguments_hash(request.arguments),
    )


def _approval_payload(spec: ToolSpec, request: ToolCallRequest) -> dict[str, Any]:
    return {
        "tool_name": spec.name,
        "capability": spec.capability.value,
        "risk_classes": sorted(risk.value for risk in spec.risk_classes),
        "argument_keys": sorted(request.arguments),
    }


def _arguments_hash(arguments: dict[str, Any]) -> str:
    encoded = json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _safe_tool_name(tool_name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")
    if (
        tool_name
        and len(tool_name) <= 128
        and all(character in allowed for character in tool_name)
        and not _looks_sensitive(tool_name)
    ):
        return tool_name
    return "<redacted>"


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "api_key",
            "apikey",
            "authorization",
            "credential",
            ".env",
            ".ssh",
            "ghp_",
            "github_pat_",
            "akia",
            "id_ed25519",
            "id_rsa",
            "known_hosts",
            "password",
            "pat_",
            ".crt",
            ".key",
            ".pem",
            "-----begin",
            "openssh",
            "private key",
            "private_key",
            "prompt",
            "secret",
            "sk-",
            "sk_",
            "token",
        )
    )


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
    schema = spec.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = sorted(required - set(arguments))
    if missing:
        return f"missing required arguments: {missing}"
    if not schema.get("additionalProperties", True):
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return f"unexpected arguments: {unexpected}"
    for key, value in arguments.items():
        property_schema = properties.get(key, {})
        expected = property_schema.get("type")
        if expected is not None and not _matches_type(value, expected):
            return f"invalid type for argument: {key}"
        if expected == "array":
            min_items = property_schema.get("minItems")
            if isinstance(min_items, int) and len(value) < min_items:
                return f"array argument has too few items: {key}"
            max_items = property_schema.get("maxItems")
            if isinstance(max_items, int) and len(value) > max_items:
                return f"array argument has too many items: {key}"
            item_schema = property_schema.get("items", {})
            item_type = item_schema.get("type")
            if isinstance(item_type, str) and not all(
                _matches_type(item, item_type) for item in value
            ):
                return f"invalid array item type for argument: {key}"
            item_max_length = item_schema.get("maxLength", property_schema.get("maxLength"))
            if isinstance(item_max_length, int) and any(
                isinstance(item, str) and len(item) > item_max_length for item in value
            ):
                return f"array item is too long: {key}"
        max_length = property_schema.get("maxLength", schema.get("maxLength"))
        if isinstance(value, str) and isinstance(max_length, int) and len(value) > max_length:
            return f"argument is too long: {key}"
    return None


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False
