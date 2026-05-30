from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable

from assistant_core.domain.events import EventType
from assistant_core.domain.tools import ToolCallRequest, ToolObservation, ToolObservationStatus
from assistant_core.tools.events import (
    audited_tool_event_payload,
    completed_event_type,
    denied_event_type,
    failed_event_type,
    is_system_diagnostics_spec,
    output_truncated_event_type,
    started_event_type,
    timeout_event_type,
    tool_event_payload,
    tool_output_sensitivity,
)
from assistant_core.tools.registry import ToolAdapter, ToolExecutionDenied
from assistant_core.tools.results import (
    completed_observation,
    effective_max_output,
    effective_timeout,
    empty_observation,
)


RecordEvent = Callable[..., Awaitable[None]]
RecordObservation = Callable[[ToolCallRequest, ToolObservation], Awaitable[None]]


async def execute_adapter(
    *,
    request: ToolCallRequest,
    adapter: ToolAdapter,
    tool_call_id: str,
    started_at: datetime,
    policy_decision_id: str,
    record_event: RecordEvent,
    record_observation: RecordEvent,
    tool_metadata: dict[str, Any] | None = None,
    policy_outcome: str | None = None,
) -> ToolObservation:
    timeout_seconds = effective_timeout(request, adapter.spec)
    max_output_bytes = effective_max_output(request, adapter.spec)
    try:
        started_event = started_event_type(adapter.spec)
        if tool_metadata is not None and started_event is not None:
            await record_event(
                started_event,
                request,
                tool_call_id=tool_call_id,
                payload=audited_tool_event_payload(
                    adapter.spec,
                    tool_metadata,
                    policy_decision_id=policy_decision_id,
                    policy_outcome=policy_outcome,
                ),
                sensitivity=tool_output_sensitivity(request, adapter.spec),
            )
        result = await asyncio.wait_for(
            adapter.invoke(request.arguments),
            timeout=timeout_seconds,
        )
    except ToolExecutionDenied as exc:
        metadata = tool_metadata or exc.metadata
        observation = empty_observation(
            request,
            ToolObservationStatus.DENIED,
            started_at,
            tool_call_id=tool_call_id,
            error={"code": exc.code, "message": exc.message},
            metadata=metadata,
            sensitivity=tool_output_sensitivity(request, adapter.spec),
        )
        denied_event = denied_event_type(adapter.spec)
        if denied_event is not None:
            await record_event(
                denied_event,
                request,
                tool_call_id=tool_call_id,
                payload=audited_tool_event_payload(
                    adapter.spec,
                    metadata,
                    policy_decision_id=policy_decision_id,
                    error_code=exc.code,
                    policy_outcome=policy_outcome,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=tool_output_sensitivity(request, adapter.spec),
            )
        await record_event(
            EventType.TOOL_CALL_DENIED,
            request,
            tool_call_id=tool_call_id,
            payload=tool_event_payload(
                adapter.spec,
                policy_decision_id=policy_decision_id,
                error_code=exc.code,
            ),
        )
        await record_observation(
            request,
            observation,
            policy_decision_id=policy_decision_id,
        )
        return observation
    except TimeoutError:
        observation = empty_observation(
            request,
            ToolObservationStatus.TIMEOUT,
            started_at,
            tool_call_id=tool_call_id,
            error={"code": "tool_timeout", "message": "tool execution timed out"},
            sensitivity=tool_output_sensitivity(request, adapter.spec),
        )
        await record_event(
            EventType.TOOL_CALL_TIMEOUT,
            request,
            tool_call_id=tool_call_id,
            payload=tool_event_payload(
                adapter.spec,
                policy_decision_id=policy_decision_id,
            ),
        )
        timeout_event = timeout_event_type(adapter.spec)
        if timeout_event is not None:
            await record_event(
                timeout_event,
                request,
                tool_call_id=tool_call_id,
                payload=audited_tool_event_payload(
                    adapter.spec,
                    tool_metadata or {},
                    policy_decision_id=policy_decision_id,
                    error_code="tool_timeout",
                    policy_outcome=policy_outcome,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=tool_output_sensitivity(request, adapter.spec),
            )
        await record_observation(
            request,
            observation,
            policy_decision_id=policy_decision_id,
        )
        return observation
    except Exception:  # noqa: BLE001 - normalize adapter failures.
        observation = empty_observation(
            request,
            ToolObservationStatus.FAILED,
            started_at,
            tool_call_id=tool_call_id,
            error={"code": "tool_failed", "message": "tool execution failed"},
            sensitivity=tool_output_sensitivity(request, adapter.spec),
        )
        await record_event(
            EventType.TOOL_CALL_FAILED,
            request,
            tool_call_id=tool_call_id,
            payload=tool_event_payload(
                adapter.spec,
                policy_decision_id=policy_decision_id,
                error_code="tool_failed",
            ),
        )
        failed_event = failed_event_type(adapter.spec)
        if failed_event is not None:
            await record_event(
                failed_event,
                request,
                tool_call_id=tool_call_id,
                payload=audited_tool_event_payload(
                    adapter.spec,
                    tool_metadata or {},
                    policy_decision_id=policy_decision_id,
                    error_code="tool_failed",
                    policy_outcome=policy_outcome,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=tool_output_sensitivity(request, adapter.spec),
            )
        await record_observation(
            request,
            observation,
            policy_decision_id=policy_decision_id,
        )
        return observation

    observation = completed_observation(
        request=request,
        adapter=adapter,
        tool_call_id=tool_call_id,
        started_at=started_at,
        result=result,
        max_output_bytes=max_output_bytes,
    )
    await record_event(
        EventType.TOOL_CALL_COMPLETED,
        request,
        tool_call_id=tool_call_id,
        payload={
            **tool_event_payload(adapter.spec, policy_decision_id=policy_decision_id),
            "truncated": observation.truncated,
            "output_bytes": observation.output_bytes,
        },
    )
    completed_event = completed_event_type(adapter.spec)
    if completed_event is not None:
        await record_event(
            completed_event,
            request,
            tool_call_id=tool_call_id,
            payload=audited_tool_event_payload(
                adapter.spec,
                {
                    **(tool_metadata or {}),
                    **observation.metadata,
                },
                policy_decision_id=policy_decision_id,
                policy_outcome=policy_outcome,
                observation=observation,
                duration_ms=observation.duration_ms,
            ),
            sensitivity=tool_output_sensitivity(request, adapter.spec),
        )
        if is_system_diagnostics_spec(adapter.spec) and observation.metadata.get("unavailable") is True:
            await record_event(
                EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE,
                request,
                tool_call_id=tool_call_id,
                payload=audited_tool_event_payload(
                    adapter.spec,
                    {
                        **(tool_metadata or {}),
                        **observation.metadata,
                    },
                    policy_decision_id=policy_decision_id,
                    policy_outcome=policy_outcome,
                    observation=observation,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=tool_output_sensitivity(request, adapter.spec),
            )
        if observation.truncated:
            truncated_event = output_truncated_event_type(adapter.spec)
            assert truncated_event is not None
            await record_event(
                truncated_event,
                request,
                tool_call_id=tool_call_id,
                payload=audited_tool_event_payload(
                    adapter.spec,
                    {
                        **(tool_metadata or {}),
                        **observation.metadata,
                    },
                    policy_decision_id=policy_decision_id,
                    policy_outcome=policy_outcome,
                    observation=observation,
                    duration_ms=observation.duration_ms,
                ),
                sensitivity=tool_output_sensitivity(request, adapter.spec),
            )
    await record_observation(
        request,
        observation,
        policy_decision_id=policy_decision_id,
    )
    return observation


__all__ = ["execute_adapter"]
