from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from assistant_core.domain.approvals import (
    ApprovalConflict,
    ApprovalNotFound,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStatus,
    CreateApprovalCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventLogPort


class InMemoryApprovalStore:
    def __init__(self, *, event_log: EventLogPort | None = None) -> None:
        self._event_log = event_log
        self._approvals: dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()

    async def create_approval(self, command: CreateApprovalCommand) -> ApprovalRequest:
        approval = command.to_request()
        async with self._lock:
            self._approvals[approval.approval_id] = approval
        await self._emit(EventType.APPROVAL_REQUIRED, approval)
        return approval

    async def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        async with self._lock:
            return self._approvals.get(approval_id)

    async def grant_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        expired: ApprovalRequest | None = None
        async with self._lock:
            approval = self._required(approval_id)
            if approval.status == ApprovalStatus.PENDING and datetime.now(UTC) >= approval.expires_at:
                updated = approval.expire()
                self._approvals[approval_id] = updated
                expired = updated
            else:
                updated = approval.grant(actor_id=actor_id, reason=reason)
                self._approvals[approval_id] = updated
        if expired is not None:
            await self._emit(EventType.APPROVAL_EXPIRED, expired)
            raise ApprovalConflict("approval expired", code="approval_expired")
        await self._emit(EventType.APPROVAL_GRANTED, updated)
        return updated

    async def deny_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        expired: ApprovalRequest | None = None
        async with self._lock:
            approval = self._required(approval_id)
            if approval.status == ApprovalStatus.PENDING and datetime.now(UTC) >= approval.expires_at:
                updated = approval.expire()
                self._approvals[approval_id] = updated
                expired = updated
            else:
                updated = approval.deny(actor_id=actor_id, reason=reason)
                self._approvals[approval_id] = updated
        if expired is not None:
            await self._emit(EventType.APPROVAL_EXPIRED, expired)
            raise ApprovalConflict("approval expired", code="approval_expired")
        await self._emit(EventType.APPROVAL_DENIED, updated)
        return updated

    async def cancel_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        async with self._lock:
            approval = self._required(approval_id)
            updated = approval.cancel(actor_id=actor_id, reason=reason)
            self._approvals[approval_id] = updated
        await self._emit(EventType.APPROVAL_CANCELLED, updated)
        return updated

    async def cancel_pending_for_request(
        self,
        request_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> list[ApprovalRequest]:
        cancelled: list[ApprovalRequest] = []
        async with self._lock:
            for approval_id, approval in list(self._approvals.items()):
                if (
                    approval.scope.request_id == request_id
                    and approval.status == ApprovalStatus.PENDING
                ):
                    updated = approval.cancel(actor_id=actor_id, reason=reason)
                    self._approvals[approval_id] = updated
                    cancelled.append(updated)
        for approval in cancelled:
            await self._emit(EventType.APPROVAL_CANCELLED, approval)
        return cancelled

    async def expire_stale(self, *, now: datetime | None = None) -> list[ApprovalRequest]:
        current_time = now or datetime.now(UTC)
        expired: list[ApprovalRequest] = []
        async with self._lock:
            for approval_id, approval in list(self._approvals.items()):
                if (
                    approval.status in {ApprovalStatus.PENDING, ApprovalStatus.GRANTED}
                    and current_time >= approval.expires_at
                ):
                    updated = approval.expire(now=current_time)
                    self._approvals[approval_id] = updated
                    expired.append(updated)
        for approval in expired:
            await self._emit(EventType.APPROVAL_EXPIRED, approval)
        return expired

    async def consume_granted_approval(
        self,
        approval_id: str,
        *,
        scope: ApprovalScope,
    ) -> ApprovalRequest:
        expired: ApprovalRequest | None = None
        async with self._lock:
            approval = self._required(approval_id)
            if (
                approval.status in {ApprovalStatus.PENDING, ApprovalStatus.GRANTED}
                and datetime.now(UTC) >= approval.expires_at
            ):
                updated = approval.expire()
                self._approvals[approval_id] = updated
                expired = updated
            if expired is None:
                updated = approval.consume(scope=scope)
                self._approvals[approval_id] = updated
                return updated
        await self._emit(EventType.APPROVAL_EXPIRED, expired)
        raise ApprovalConflict("approval expired", code="approval_expired")

    def _required(self, approval_id: str) -> ApprovalRequest:
        approval = self._approvals.get(approval_id)
        if approval is None:
            raise ApprovalNotFound(f"approval not found: {approval_id}")
        return approval

    async def _emit(self, event_type: EventType, approval: ApprovalRequest) -> None:
        if self._event_log is None:
            return
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=event_type,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=approval.scope.conversation_id,
                request_id=approval.scope.request_id,
                correlation_id=approval.scope.request_id,
                causation_id=approval.metadata.get("causation_event_id"),
                parent_event_id=None,
                actor_type=ActorType.USER if event_type in _USER_EVENTS else ActorType.SYSTEM,
                actor_id=approval.decision_actor_id or approval.requested_by,
                source_component="approval_store",
                source_node=None,
                sensitivity=Sensitivity.PROJECT,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload=_approval_event_payload(approval),
                metadata={},
            ),
        )


_USER_EVENTS = {
    EventType.APPROVAL_GRANTED,
    EventType.APPROVAL_DENIED,
    EventType.APPROVAL_CANCELLED,
}


def _approval_event_payload(approval: ApprovalRequest) -> dict[str, object]:
    return {
        "approval_id": approval.approval_id,
        "status": approval.status.value,
        "capability": approval.capability.value,
        "risk_classes": sorted(risk.value for risk in approval.risk_classes),
        "tool_name": approval.scope.tool_name,
        "step_id": approval.scope.step_id,
        "expires_at": approval.expires_at.isoformat(),
        "redacted_summary": _redacted_summary(approval),
        "redacted_payload": approval.redacted_payload,
        "policy_decision_id": approval.metadata.get("policy_decision_id"),
    }


def _redacted_summary(approval: ApprovalRequest) -> str:
    keys = ", ".join(approval.scope.argument_keys)
    return f"{approval.scope.tool_name}({keys})"
