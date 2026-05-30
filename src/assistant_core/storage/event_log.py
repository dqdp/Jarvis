from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from assistant_core.domain.events import (
    ActorType,
    EventEnvelope,
    EventType,
    EventVisibility,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import (
    EventFilter,
    sanitize_event_envelope,
    validate_event_envelope,
)


_metadata = sa.MetaData()

_events = sa.Table(
    "events",
    _metadata,
    sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("event_seq", sa.BigInteger(), sa.Identity(), nullable=False),
    sa.Column("event_type", sa.Text(), nullable=False),
    sa.Column("event_version", sa.Integer(), nullable=False),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("causation_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("parent_event_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("actor_type", sa.Text(), nullable=False),
    sa.Column("actor_id", sa.Text(), nullable=True),
    sa.Column("source_component", sa.Text(), nullable=False),
    sa.Column("source_node", sa.Text(), nullable=True),
    sa.Column("sensitivity", sa.Text(), nullable=False),
    sa.Column("visibility", sa.Text(), nullable=False),
    sa.Column("idempotency_key", sa.Text(), nullable=True),
    sa.Column("payload", postgresql.JSONB(), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)


class PostgresEventLog:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def append(self, event: EventEnvelope) -> EventEnvelope:
        async with self._engine.begin() as connection:
            return await insert_event(connection, event)

    async def query(self, event_filter: EventFilter) -> list[EventEnvelope]:
        statement = sa.select(_events).order_by(_events.c.event_seq)

        conditions = []
        if event_filter.request_id is not None:
            conditions.append(_events.c.request_id == _uuid(event_filter.request_id))
        if event_filter.conversation_id is not None:
            conditions.append(_events.c.conversation_id == _uuid(event_filter.conversation_id))
        if event_filter.correlation_id is not None:
            conditions.append(_events.c.correlation_id == _uuid(event_filter.correlation_id))
        if conditions:
            statement = statement.where(*conditions)

        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()

        return [_row_to_event(row) for row in rows]


async def insert_event(connection: AsyncConnection, event: EventEnvelope) -> EventEnvelope:
    event = sanitize_event_envelope(event)
    validate_event_envelope(event)
    statement = (
        sa.insert(_events)
        .values(_event_values(event))
        .returning(*_events.c)
    )
    row = (await connection.execute(statement)).mappings().one()
    return _row_to_event(row)


def _event_values(event: EventEnvelope) -> dict[str, Any]:
    return {
        "event_id": _uuid(event.event_id),
        "event_type": event.event_type.value,
        "event_version": event.event_version,
        "occurred_at": event.occurred_at,
        "recorded_at": event.recorded_at,
        "conversation_id": _optional_uuid(event.conversation_id),
        "request_id": _optional_uuid(event.request_id),
        "correlation_id": _optional_uuid(event.correlation_id),
        "causation_id": _optional_uuid(event.causation_id),
        "parent_event_id": _optional_uuid(event.parent_event_id),
        "actor_type": event.actor_type.value,
        "actor_id": event.actor_id,
        "source_component": event.source_component,
        "source_node": event.source_node,
        "sensitivity": event.sensitivity.value,
        "visibility": event.visibility.value,
        "idempotency_key": event.idempotency_key,
        "payload": event.payload,
        "metadata": event.metadata,
    }


def _row_to_event(row: Mapping[str, Any]) -> EventEnvelope:
    return EventEnvelope(
        event_id=str(row["event_id"]),
        event_seq=row["event_seq"],
        event_type=EventType(row["event_type"]),
        event_version=row["event_version"],
        occurred_at=_datetime(row["occurred_at"]),
        recorded_at=_datetime(row["recorded_at"]),
        conversation_id=_optional_string(row["conversation_id"]),
        request_id=_optional_string(row["request_id"]),
        correlation_id=_optional_string(row["correlation_id"]),
        causation_id=_optional_string(row["causation_id"]),
        parent_event_id=_optional_string(row["parent_event_id"]),
        actor_type=ActorType(row["actor_type"]),
        actor_id=row["actor_id"],
        source_component=row["source_component"],
        source_node=row["source_node"],
        sensitivity=Sensitivity(row["sensitivity"]),
        visibility=EventVisibility(row["visibility"]),
        idempotency_key=row["idempotency_key"],
        payload=dict(row["payload"]),
        metadata=dict(row["metadata"]),
    )


def _uuid(value: str) -> UUID:
    return UUID(value)


def _optional_uuid(value: str | None) -> UUID | None:
    if value is None:
        return None
    return _uuid(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime, got {type(value).__name__}")
