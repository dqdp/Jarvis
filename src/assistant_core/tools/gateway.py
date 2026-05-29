from __future__ import annotations

import asyncio
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
from assistant_core.tools.registry import ToolAdapter, ToolRegistry


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
        )

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
            )
        except ApprovalConflict as exc:
            return _empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": exc.code, "message": str(exc)},
            )
        return None

    async def _execute_adapter(
        self,
        request: ToolCallRequest,
        adapter: ToolAdapter,
        tool_call_id: str,
        started_at: datetime,
        policy_decision_id: str,
    ) -> ToolObservation:
        timeout_seconds = _effective_timeout(request, adapter.spec)
        max_output_bytes = _effective_max_output(request, adapter.spec)
        try:
            result = await asyncio.wait_for(
                adapter.invoke(request.arguments),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            observation = _empty_observation(
                request,
                ToolObservationStatus.TIMEOUT,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": "tool_timeout", "message": "tool execution timed out"},
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
        )

    async def _record_event(
        self,
        event_type: EventType,
        request: ToolCallRequest,
        *,
        tool_call_id: str,
        payload: dict[str, Any],
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
                sensitivity=request.sensitivity,
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
) -> ToolObservation:
    completed_at = datetime.now(UTC)
    return ToolObservation(
        tool_call_id=tool_call_id or str(uuid4()),
        tool_name=tool_name or request.tool_name,
        status=status,
        content="",
        content_type="text/plain",
        sensitivity=request.sensitivity,
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
    content, content_type = _serialize_content(result, adapter.content_type)
    content = _redact_content(content)
    encoded = content.encode("utf-8")
    output_bytes = len(encoded)
    truncated = output_bytes > max_output_bytes
    if truncated:
        content = encoded[:max_output_bytes].decode("utf-8", errors="ignore")
    completed_at = datetime.now(UTC)
    return ToolObservation(
        tool_call_id=tool_call_id,
        tool_name=request.tool_name,
        status=ToolObservationStatus.COMPLETED,
        content=content,
        content_type=content_type,
        sensitivity=request.sensitivity,
        truncated=truncated,
        output_bytes=output_bytes,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
        error=None,
    )


def _redact_content(content: str) -> str:
    return "<redacted>" if _looks_sensitive(content) else content


def _serialize_content(result: Any, content_type: str) -> tuple[str, str]:
    if isinstance(result, str):
        return result, content_type
    return json.dumps(result, sort_keys=True), "application/json"


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
            "ghp_",
            "github_pat_",
            "akia",
            "password",
            "pat_",
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
