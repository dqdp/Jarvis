from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy import text

from assistant_core.api.app import create_app
from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.approvals import ApprovalScope, CreateApprovalCommand
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.fake_provider import FakeEmbeddingProvider
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.storage.approval_store import PostgresApprovalStore
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations


pytestmark = [pytest.mark.contract, pytest.mark.db]


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


async def _truncate_api(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("set local jarvis.allow_events_truncate = 'on'"))
            await connection.execute(
                text(
                    "truncate table approvals, content_embeddings, content_chunks, content_sources, "
                    "memory_embeddings, memory_candidates, memories, "
                    "model_invocations, assistant_requests, messages, conversations, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


@pytest.fixture
def approval_app_parts():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_api(database_url))
    engine = create_database_engine(database_url)
    settings = ConfigLoader(Path("config")).load("test")
    approval_store = PostgresApprovalStore(engine)
    app = create_app(
        conversation_store=PostgresConversationStore(engine),
        memory_store=PostgresMemoryStore(
            engine=engine,
            settings=settings,
            policy=ConfigPolicyEngine(settings),
            embedding_port=FakeEmbeddingProvider(),
        ),
        settings=settings,
        approval_store=approval_store,
    )
    app.state.approval_store = approval_store
    app.state.engine = engine
    assert isinstance(app, FastAPI)
    try:
        yield app, approval_store
    finally:
        asyncio.run(engine.dispose())


def _scope() -> ApprovalScope:
    return ApprovalScope(
        capability=Capability.TOOL_SAFE,
        risk_classes=frozenset({RiskClass.SAFE}),
        tool_name="fake.echo",
        user_id="local_user",
        request_id=_id("request"),
        conversation_id=_id("conversation"),
        step_id="step-1",
        project_namespace="project.personal_assistant",
        working_directory=None,
        sensitivity=Sensitivity.PROJECT,
        permission_mode="developer_local",
        argument_keys=("message",),
        arguments_hash="sha256:args",
    )


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"jarvis-approval-api:{label}"))


async def _create_approval(
    store: PostgresApprovalStore,
    *,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
):
    now = created_at or datetime.now(UTC)
    return await store.create_approval(
        CreateApprovalCommand(
            scope=_scope(),
            redacted_payload={
                "tool_name": "fake.echo",
                "argument_keys": ["message"],
                "secret_token": "<redacted>",
            },
            requested_by="local_user",
            created_at=now,
            expires_at=expires_at or now + timedelta(minutes=5),
        ),
    )


async def _request(app, method: str, path: str, body: dict[str, Any] | None = None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        kwargs: dict[str, Any] = {"json": body} if body is not None else {}
        response = await client.request(method, path, **kwargs)
    payload = response.json() if response.content else None
    return response.status_code, payload


def test_get_approval_returns_redacted_payload(approval_app_parts) -> None:
    app, store = approval_app_parts

    async def scenario():
        approval = await _create_approval(store)
        status, payload = await _request(app, "GET", f"/v1/approvals/{approval.approval_id}")
        return approval, status, payload

    approval, status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["approval_id"] == approval.approval_id
    assert payload["status"] == "pending"
    assert payload["redacted_payload"]["secret_token"] == "<redacted>"
    assert "raw" not in str(payload).lower()


def test_grant_pending_approval(approval_app_parts) -> None:
    app, store = approval_app_parts

    async def scenario():
        approval = await _create_approval(store)
        return await _request(app, "POST", f"/v1/approvals/{approval.approval_id}/grant", {})

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["status"] == "granted"


def test_deny_pending_approval(approval_app_parts) -> None:
    app, store = approval_app_parts

    async def scenario():
        approval = await _create_approval(store)
        return await _request(app, "POST", f"/v1/approvals/{approval.approval_id}/deny", {})

    status, payload = asyncio.run(scenario())

    assert status == 200
    assert payload["status"] == "denied"


def test_grant_expired_approval_returns_conflict(approval_app_parts) -> None:
    app, store = approval_app_parts

    async def scenario():
        approval = await _create_approval(
            store,
            created_at=datetime(2026, 5, 29, 10, 0, tzinfo=UTC),
            expires_at=datetime(2026, 5, 29, 10, 0, 1, tzinfo=UTC),
        )
        await store.expire_stale(now=datetime(2026, 5, 29, 10, 1, tzinfo=UTC))
        return await _request(app, "POST", f"/v1/approvals/{approval.approval_id}/grant", {})

    status, payload = asyncio.run(scenario())

    assert status == 409
    assert payload["error"]["code"] == "approval_expired"


def test_deny_already_granted_approval_returns_conflict(approval_app_parts) -> None:
    app, store = approval_app_parts

    async def scenario():
        approval = await _create_approval(store)
        await store.grant_approval(approval.approval_id, actor_id="user-1")
        return await _request(app, "POST", f"/v1/approvals/{approval.approval_id}/deny", {})

    status, payload = asyncio.run(scenario())

    assert status == 409
    assert payload["error"]["code"] == "approval_granted"
