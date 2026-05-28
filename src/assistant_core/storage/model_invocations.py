from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncEngine

from assistant_core.domain.model_invocations import ModelInvocationRecord
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.model_invocations import (
    FinishModelInvocationCommand,
    StartModelInvocationCommand,
)


_metadata = sa.MetaData()

_model_invocations = sa.Table(
    "model_invocations",
    _metadata,
    sa.Column("model_invocation_id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("profile", sa.Text(), nullable=False),
    sa.Column("provider", sa.Text(), nullable=False),
    sa.Column("model", sa.Text(), nullable=False),
    sa.Column("purpose", sa.Text(), nullable=False),
    sa.Column("sensitivity", sa.Text(), nullable=False),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("latency_ms", sa.Integer(), nullable=True),
    sa.Column("input_token_estimate", sa.Integer(), nullable=True),
    sa.Column("input_tokens_reported", sa.Integer(), nullable=True),
    sa.Column("output_tokens_reported", sa.Integer(), nullable=True),
    sa.Column("streaming", sa.Boolean(), nullable=False),
    sa.Column("error_type", sa.Text(), nullable=True),
    sa.Column("error_message", sa.Text(), nullable=True),
    sa.Column("context_manifest_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("metadata", postgresql.JSONB(), nullable=False),
)


class PostgresModelInvocationRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def start(
        self,
        command: StartModelInvocationCommand,
    ) -> ModelInvocationRecord:
        statement = (
            sa.insert(_model_invocations)
            .values(
                {
                    "model_invocation_id": uuid4(),
                    "request_id": _optional_uuid(command.request_id),
                    "conversation_id": _optional_uuid(command.conversation_id),
                    "profile": command.profile,
                    "provider": command.provider,
                    "model": command.model,
                    "purpose": command.purpose,
                    "sensitivity": command.sensitivity.value,
                    "status": "running",
                    "started_at": _now(),
                    "finished_at": None,
                    "latency_ms": None,
                    "input_token_estimate": command.input_token_estimate,
                    "input_tokens_reported": None,
                    "output_tokens_reported": None,
                    "streaming": command.streaming,
                    "error_type": None,
                    "error_message": None,
                    "context_manifest_id": _optional_uuid(command.context_manifest_id),
                    "metadata": command.metadata,
                },
            )
            .returning(*_model_invocations.c)
        )
        async with self._engine.begin() as connection:
            row = (await connection.execute(statement)).mappings().one()
        return _row_to_invocation(row)

    async def finish(
        self,
        command: FinishModelInvocationCommand,
    ) -> ModelInvocationRecord:
        async with self._engine.begin() as connection:
            current = (
                await connection.execute(
                    sa.select(_model_invocations).where(
                        _model_invocations.c.model_invocation_id
                        == _uuid(command.model_invocation_id),
                    ),
                )
            ).mappings().one()
            finished_at = _now()
            latency_ms = int(
                (finished_at - _datetime(current["started_at"])).total_seconds() * 1000,
            )
            metadata = dict(current["metadata"])
            if command.metadata:
                metadata.update(command.metadata)
            row = (
                await connection.execute(
                    sa.update(_model_invocations)
                    .where(
                        _model_invocations.c.model_invocation_id
                        == _uuid(command.model_invocation_id),
                    )
                    .values(
                        {
                            "status": command.status,
                            "finished_at": finished_at,
                            "latency_ms": latency_ms,
                            "input_tokens_reported": command.input_tokens_reported,
                            "output_tokens_reported": command.output_tokens_reported,
                            "error_type": command.error_type,
                            "error_message": command.error_message,
                            "metadata": metadata,
                        },
                    )
                    .returning(*_model_invocations.c),
                )
            ).mappings().one()
        return _row_to_invocation(row)

    async def list_recent(self, limit: int) -> list[ModelInvocationRecord]:
        statement = (
            sa.select(_model_invocations)
            .order_by(_model_invocations.c.started_at, _model_invocations.c.model_invocation_id)
            .limit(limit)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return [_row_to_invocation(row) for row in rows]


def _row_to_invocation(row: Mapping[str, Any]) -> ModelInvocationRecord:
    return ModelInvocationRecord(
        model_invocation_id=str(row["model_invocation_id"]),
        request_id=_optional_string(row["request_id"]),
        conversation_id=_optional_string(row["conversation_id"]),
        profile=row["profile"],
        provider=row["provider"],
        model=row["model"],
        purpose=row["purpose"],
        sensitivity=Sensitivity(row["sensitivity"]),
        status=row["status"],
        started_at=_datetime(row["started_at"]),
        finished_at=_optional_datetime(row["finished_at"]),
        latency_ms=row["latency_ms"],
        input_token_estimate=row["input_token_estimate"],
        input_tokens_reported=row["input_tokens_reported"],
        output_tokens_reported=row["output_tokens_reported"],
        streaming=row["streaming"],
        error_type=row["error_type"],
        error_message=row["error_message"],
        context_manifest_id=_optional_string(row["context_manifest_id"]),
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


def _now() -> datetime:
    return datetime.now(UTC)


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime, got {type(value).__name__}")


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)
