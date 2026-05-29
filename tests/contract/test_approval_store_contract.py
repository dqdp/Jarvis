from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import os
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy import text

from assistant_core.domain.approvals import (
    ApprovalConflict,
    ApprovalScope,
    ApprovalStatus,
    CreateApprovalCommand,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventFilter
from assistant_core.storage.approval_store import PostgresApprovalStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.migrations import run_migrations


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_approvals(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "truncate table approvals, events restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


@pytest.fixture
def store_parts():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_approvals(database_url))
    engine = create_database_engine(database_url)
    event_log = PostgresEventLog(engine)
    try:
        yield PostgresApprovalStore(engine, event_log=event_log), event_log
    finally:
        asyncio.run(engine.dispose())


def _scope(*, tool_name: str = "fake.echo", arguments_hash: str = "sha256:args") -> ApprovalScope:
    return ApprovalScope(
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        tool_name=tool_name,
        user_id="user-1",
        request_id=_id("request"),
        conversation_id=_id("conversation"),
        step_id="step-1",
        project_namespace="project.personal_assistant",
        working_directory=None,
        sensitivity=Sensitivity.PROJECT,
        permission_mode="developer_local",
        argument_keys=("message",),
        arguments_hash=arguments_hash,
    )


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-approval-store:{label}"))


def _command(
    *,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> CreateApprovalCommand:
    now = created_at or datetime.now(UTC)
    return CreateApprovalCommand(
        scope=_scope(),
        redacted_payload={"tool_name": "fake.echo", "argument_keys": ["message"]},
        requested_by="user-1",
        created_at=now,
        expires_at=expires_at or now + timedelta(minutes=5),
    )


def test_approval_store_creates_pending_approval(store_parts) -> None:
    store, event_log = store_parts

    async def scenario():
        approval = await store.create_approval(_command())
        events = await event_log.query(EventFilter(request_id=_id("request")))
        return approval, events

    approval, events = asyncio.run(scenario())

    assert approval.status == ApprovalStatus.PENDING
    assert approval.redacted_payload == {"tool_name": "fake.echo", "argument_keys": ["message"]}
    assert [event.event_type for event in events] == [EventType.APPROVAL_REQUIRED]


def test_approval_store_gets_approval_by_id(store_parts) -> None:
    store, _event_log = store_parts

    async def scenario():
        created = await store.create_approval(_command())
        return created, await store.get_approval(created.approval_id)

    created, loaded = asyncio.run(scenario())

    assert loaded == created


def test_approval_store_grants_pending_approval(store_parts) -> None:
    store, event_log = store_parts

    async def scenario():
        created = await store.create_approval(_command())
        granted = await store.grant_approval(created.approval_id, actor_id="user-1", reason="ok")
        events = await event_log.query(EventFilter(request_id=_id("request")))
        return granted, events

    granted, events = asyncio.run(scenario())

    assert granted.status == ApprovalStatus.GRANTED
    assert EventType.APPROVAL_GRANTED in [event.event_type for event in events]


def test_approval_store_denies_pending_approval(store_parts) -> None:
    store, event_log = store_parts

    async def scenario():
        created = await store.create_approval(_command())
        denied = await store.deny_approval(created.approval_id, actor_id="user-1", reason="no")
        events = await event_log.query(EventFilter(request_id=_id("request")))
        return denied, events

    denied, events = asyncio.run(scenario())

    assert denied.status == ApprovalStatus.DENIED
    assert EventType.APPROVAL_DENIED in [event.event_type for event in events]


def test_approval_store_expires_stale_approvals(store_parts) -> None:
    store, event_log = store_parts

    async def scenario():
        created = await store.create_approval(
            _command(
                created_at=datetime(2026, 5, 29, 10, 0, tzinfo=UTC),
                expires_at=datetime(2026, 5, 29, 10, 1, tzinfo=UTC),
            ),
        )
        expired = await store.expire_stale(now=datetime(2026, 5, 29, 10, 2, tzinfo=UTC))
        loaded = await store.get_approval(created.approval_id)
        events = await event_log.query(EventFilter(request_id=_id("request")))
        return expired, loaded, events

    expired, loaded, events = asyncio.run(scenario())

    assert expired == [loaded]
    assert loaded.status == ApprovalStatus.EXPIRED
    assert EventType.APPROVAL_EXPIRED in [event.event_type for event in events]


def test_approval_store_consumes_granted_approval_once(store_parts) -> None:
    store, _event_log = store_parts

    async def scenario():
        created = await store.create_approval(_command())
        granted = await store.grant_approval(created.approval_id, actor_id="user-1")
        consumed = await store.consume_granted_approval(granted.approval_id, scope=_scope())
        with pytest.raises(ApprovalConflict, match="used"):
            await store.consume_granted_approval(granted.approval_id, scope=_scope())
        return consumed

    consumed = asyncio.run(scenario())
    assert consumed.status == ApprovalStatus.GRANTED
    assert consumed.used_at is not None


def test_approval_store_expires_granted_unused_approval_on_consume(store_parts) -> None:
    store, event_log = store_parts

    async def scenario():
        now = datetime.now(UTC)
        created = await store.create_approval(
            _command(
                created_at=now,
                expires_at=now + timedelta(milliseconds=200),
            ),
        )
        granted = await store.grant_approval(
            created.approval_id,
            actor_id="user-1",
            reason="ok",
        )
        await asyncio.sleep(0.25)
        with pytest.raises(ApprovalConflict, match="expired"):
            await store.consume_granted_approval(granted.approval_id, scope=_scope())
        loaded = await store.get_approval(granted.approval_id)
        events = await event_log.query(EventFilter(request_id=_id("request")))
        return loaded, events

    loaded, events = asyncio.run(scenario())

    assert loaded.status == ApprovalStatus.EXPIRED
    assert EventType.APPROVAL_EXPIRED in [event.event_type for event in events]
