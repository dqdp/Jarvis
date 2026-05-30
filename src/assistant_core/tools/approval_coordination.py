from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from assistant_core.domain.approvals import (
    ApprovalConflict,
    ApprovalNotFound,
    CreateApprovalCommand,
)
from assistant_core.domain.tools import ToolCallRequest, ToolObservation, ToolObservationStatus, ToolSpec
from assistant_core.ports.approvals import ApprovalStorePort
from assistant_core.tools.approval_flow import (
    approval_payload,
    approval_scope,
)
from assistant_core.tools.events import tool_output_sensitivity
from assistant_core.tools.results import empty_observation


class ToolApprovalCoordinator:
    def __init__(self, approval_store: ApprovalStorePort | None) -> None:
        self._approval_store = approval_store

    async def create_metadata(
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
                scope=approval_scope(spec, request),
                redacted_payload=approval_payload(spec, request),
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

    async def validate_approval(
        self,
        request: ToolCallRequest,
        spec: ToolSpec,
        *,
        tool_call_id: str,
    ) -> ToolObservation | None:
        started_at = datetime.now(UTC)
        if self._approval_store is None:
            return empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={
                    "code": "approval_store_unavailable",
                    "message": "approval store is not configured",
                },
                sensitivity=tool_output_sensitivity(request, spec),
            )
        try:
            await self._approval_store.consume_granted_approval(
                request.approval_id or "",
                scope=approval_scope(spec, request),
            )
        except ApprovalNotFound:
            return empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": "approval_not_found", "message": "approval not found"},
                sensitivity=tool_output_sensitivity(request, spec),
            )
        except ApprovalConflict as exc:
            return empty_observation(
                request,
                ToolObservationStatus.DENIED,
                started_at,
                tool_call_id=tool_call_id,
                error={"code": exc.code, "message": str(exc)},
                sensitivity=tool_output_sensitivity(request, spec),
            )
        return None
