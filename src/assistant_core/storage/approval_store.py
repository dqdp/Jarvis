from __future__ import annotations

from datetime import UTC, datetime
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from assistant_core.domain.approvals import (
    ApprovalConflict,
    ApprovalNotFound,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStatus,
    CreateApprovalCommand,
)
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventLogPort


_metadata = sa.MetaData()

_approvals = sa.Table(
    "approvals",
    _metadata,
    sa.Column("approval_id", sa.Text(), primary_key=True),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("capability", sa.Text(), nullable=False),
    sa.Column("risk_classes", postgresql.JSONB(), nullable=False),
    sa.Column("tool_name", sa.Text(), nullable=False),
    sa.Column("request_id", sa.Text(), nullable=True),
    sa.Column("conversation_id", sa.Text(), nullable=True),
    sa.Column("step_id", sa.Text(), nullable=True),
    sa.Column("project_namespace", sa.Text(), nullable=True),
    sa.Column("working_directory", sa.Text(), nullable=True),
    sa.Column("argument_keys", postgresql.JSONB(), nullable=False),
    sa.Column("arguments_hash", sa.Text(), nullable=False),
    sa.Column("scope", postgresql.JSONB(), nullable=False),
    sa.Column("redacted_payload", postgresql.JSONB(), nullable=False),
    sa.Column("requested_by", sa.Text(), nullable=True),
    sa.Column("decision_actor_id", sa.Text(), nullable=True),
    sa.Column("decision_reason", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("granted_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("denied_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)


class PostgresApprovalStore:
    def __init__(self, engine: AsyncEngine, *, event_log: EventLogPort | None = None) -> None:
        self.engine = engine
        self._event_log = event_log

    async def create_approval(self, command: CreateApprovalCommand) -> ApprovalRequest:
        approval = command.to_request()
        async with self.engine.begin() as connection:
            row = (
                await connection.execute(
                    sa.insert(_approvals).values(_approval_values(approval)).returning(*_approvals.c),
                )
            ).mappings().one()
        created = _row_to_approval(row)
        await self._emit(EventType.APPROVAL_REQUIRED, created)
        return created

    async def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        async with self.engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.select(_approvals).where(_approvals.c.approval_id == approval_id),
                )
            ).mappings().first()
        if row is None:
            return None
        return _row_to_approval(row)

    async def grant_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        expired_approval: ApprovalRequest | None = None
        async with self.engine.begin() as connection:
            approval = await self._required(connection, approval_id)
            if approval.status == ApprovalStatus.PENDING and datetime.now(UTC) >= approval.expires_at:
                expired_approval = await _save(connection, approval.expire())
            else:
                updated = approval.grant(actor_id=actor_id, reason=reason)
                saved = await _save(connection, updated)
        if expired_approval is not None:
            await self._emit(EventType.APPROVAL_EXPIRED, expired_approval)
            raise ApprovalConflict("approval expired", code="approval_expired")
        await self._emit(EventType.APPROVAL_GRANTED, saved)
        return saved

    async def deny_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        expired_approval: ApprovalRequest | None = None
        async with self.engine.begin() as connection:
            approval = await self._required(connection, approval_id)
            if approval.status == ApprovalStatus.PENDING and datetime.now(UTC) >= approval.expires_at:
                expired_approval = await _save(connection, approval.expire())
            else:
                updated = approval.deny(actor_id=actor_id, reason=reason)
                saved = await _save(connection, updated)
        if expired_approval is not None:
            await self._emit(EventType.APPROVAL_EXPIRED, expired_approval)
            raise ApprovalConflict("approval expired", code="approval_expired")
        await self._emit(EventType.APPROVAL_DENIED, saved)
        return saved

    async def cancel_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest:
        async with self.engine.begin() as connection:
            approval = await self._required(connection, approval_id)
            updated = approval.cancel(actor_id=actor_id, reason=reason)
            saved = await _save(connection, updated)
        await self._emit(EventType.APPROVAL_CANCELLED, saved)
        return saved

    async def cancel_pending_for_request(
        self,
        request_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> list[ApprovalRequest]:
        cancelled: list[ApprovalRequest] = []
        async with self.engine.begin() as connection:
            rows = (
                await connection.execute(
                    sa.select(_approvals)
                    .where(_approvals.c.request_id == request_id)
                    .where(_approvals.c.status == ApprovalStatus.PENDING.value)
                    .with_for_update(),
                )
            ).mappings().all()
            for row in rows:
                approval = _row_to_approval(row)
                updated = approval.cancel(actor_id=actor_id, reason=reason)
                cancelled.append(await _save(connection, updated))
        for approval in cancelled:
            await self._emit(EventType.APPROVAL_CANCELLED, approval)
        return cancelled

    async def expire_stale(self, *, now: datetime | None = None) -> list[ApprovalRequest]:
        current_time = now or datetime.now(UTC)
        expired: list[ApprovalRequest] = []
        async with self.engine.begin() as connection:
            rows = (
                await connection.execute(
                    sa.select(_approvals)
                    .where(
                        _approvals.c.status.in_(
                            [ApprovalStatus.PENDING.value, ApprovalStatus.GRANTED.value],
                        ),
                    )
                    .where(_approvals.c.expires_at <= current_time)
                    .order_by(_approvals.c.expires_at)
                    .with_for_update(),
                )
            ).mappings().all()
            for row in rows:
                approval = _row_to_approval(row)
                updated = approval.expire(now=current_time)
                expired.append(await _save(connection, updated))
        for approval in expired:
            await self._emit(EventType.APPROVAL_EXPIRED, approval)
        return expired

    async def consume_granted_approval(
        self,
        approval_id: str,
        *,
        scope: ApprovalScope,
    ) -> ApprovalRequest:
        expired_approval: ApprovalRequest | None = None
        async with self.engine.begin() as connection:
            approval = await self._required(connection, approval_id)
            if (
                approval.status in {ApprovalStatus.PENDING, ApprovalStatus.GRANTED}
                and datetime.now(UTC) >= approval.expires_at
            ):
                expired_approval = await _save(connection, approval.expire())
            else:
                updated = approval.consume(scope=scope)
                return await _save(connection, updated)
        if expired_approval is not None:
            await self._emit(EventType.APPROVAL_EXPIRED, expired_approval)
            raise ApprovalConflict("approval expired", code="approval_expired")
        raise ApprovalNotFound(f"approval not found: {approval_id}")

    async def _required(self, connection: AsyncConnection, approval_id: str) -> ApprovalRequest:
        row = (
            await connection.execute(
                sa.select(_approvals)
                .where(_approvals.c.approval_id == approval_id)
                .with_for_update(),
            )
        ).mappings().first()
        if row is None:
            raise ApprovalNotFound(f"approval not found: {approval_id}")
        return _row_to_approval(row)

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


async def _save(connection: AsyncConnection, approval: ApprovalRequest) -> ApprovalRequest:
    values = _approval_values(approval)
    values["updated_at"] = datetime.now(UTC)
    row = (
        await connection.execute(
            sa.update(_approvals)
            .where(_approvals.c.approval_id == approval.approval_id)
            .values(values)
            .returning(*_approvals.c),
        )
    ).mappings().one()
    return _row_to_approval(row)


def _approval_values(approval: ApprovalRequest) -> dict[str, Any]:
    scope = approval.scope.payload()
    return {
        "approval_id": approval.approval_id,
        "status": approval.status.value,
        "capability": approval.capability.value,
        "risk_classes": sorted(risk.value for risk in approval.risk_classes),
        "tool_name": approval.scope.tool_name,
        "request_id": approval.scope.request_id,
        "conversation_id": approval.scope.conversation_id,
        "step_id": approval.scope.step_id,
        "project_namespace": approval.scope.project_namespace,
        "working_directory": approval.scope.working_directory,
        "argument_keys": list(approval.scope.argument_keys),
        "arguments_hash": approval.scope.arguments_hash,
        "scope": scope,
        "redacted_payload": approval.redacted_payload,
        "requested_by": approval.requested_by,
        "decision_actor_id": approval.decision_actor_id,
        "decision_reason": approval.decision_reason,
        "created_at": approval.created_at,
        "updated_at": datetime.now(UTC),
        "expires_at": approval.expires_at,
        "granted_at": approval.granted_at,
        "denied_at": approval.denied_at,
        "cancelled_at": approval.cancelled_at,
        "used_at": approval.used_at,
        "metadata": approval.metadata,
    }


def _row_to_approval(row: Mapping[str, Any]) -> ApprovalRequest:
    scope = ApprovalScope.from_payload(dict(row["scope"]))
    return ApprovalRequest(
        approval_id=row["approval_id"],
        status=ApprovalStatus(row["status"]),
        capability=Capability(row["capability"]),
        risk_classes=frozenset(RiskClass(value) for value in row["risk_classes"]),
        scope=scope,
        redacted_payload=dict(row["redacted_payload"]),
        requested_by=row["requested_by"],
        created_at=_datetime(row["created_at"]),
        expires_at=_datetime(row["expires_at"]),
        decision_actor_id=row["decision_actor_id"],
        decision_reason=row["decision_reason"],
        granted_at=_optional_datetime(row["granted_at"]),
        denied_at=_optional_datetime(row["denied_at"]),
        cancelled_at=_optional_datetime(row["cancelled_at"]),
        used_at=_optional_datetime(row["used_at"]),
        metadata=dict(row["metadata"]),
    )


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime, got {type(value).__name__}")


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)


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
