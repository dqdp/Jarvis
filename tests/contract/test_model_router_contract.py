from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy import text

from assistant_core.config.settings import ConfigLoader, Settings
from assistant_core.domain.messages import ChatMessage, MessageRole, TextPart
from assistant_core.domain.models import ChatModelRequest, EmbeddingRequest, StructuredModelRequest
from assistant_core.domain.policy import (
    ContextPolicyRequest,
    MemoryWritePolicyRequest,
    ModelPolicyRequest,
    PolicyDecision,
)
from assistant_core.domain.events import EventType
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.models.fake_provider import FakeEmbeddingProvider, FakeModelProvider
from assistant_core.models.router import ModelPolicyDenied, ModelProviderError, ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.event_log import EventFilter
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository


pytestmark = pytest.mark.contract


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def _settings() -> Settings:
    return ConfigLoader(Path("config")).load("test")


async def _truncate_model_invocations(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("truncate table model_invocations"))
    finally:
        await engine.dispose()


@pytest.fixture
def model_repo():
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)
    asyncio.run(_truncate_model_invocations(database_url))
    engine = create_database_engine(database_url)
    repository = PostgresModelInvocationRepository(engine)
    try:
        yield repository
    finally:
        asyncio.run(engine.dispose())


class RecordingPolicy:
    def __init__(self, call_log: list[str]) -> None:
        self.call_log = call_log

    async def evaluate_model_request(self, request: ModelPolicyRequest) -> PolicyDecision:
        self.call_log.append("policy.model")
        return PolicyDecision(allowed=True, code="allowed", reason="test")

    async def evaluate_memory_write(self, request: MemoryWritePolicyRequest) -> PolicyDecision:
        return PolicyDecision(allowed=True, code="allowed", reason="test")

    async def evaluate_context_inclusion(self, request: ContextPolicyRequest) -> PolicyDecision:
        return PolicyDecision(allowed=True, code="allowed", reason="test")


def _chat_request(
    *,
    profile: str = "local_main",
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> ChatModelRequest:
    return ChatModelRequest(
        profile=profile,
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content=[TextPart(text="hello")],
                sensitivity=sensitivity,
            ),
        ],
        sensitivity=sensitivity,
        request_id="11111111-1111-1111-1111-111111111111",
        conversation_id="22222222-2222-2222-2222-222222222222",
    )


def _structured_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        profile="local_structured",
        messages=_chat_request().messages,
        schema={"type": "object"},
        sensitivity=Sensitivity.PROJECT,
    )


def _embedding_request() -> EmbeddingRequest:
    return EmbeddingRequest(
        profile="local_embedding",
        texts=["hello"],
        sensitivity=Sensitivity.PROJECT,
    )


def _router(
    model_repo,
    *,
    policy=None,
    model_provider: FakeModelProvider | None = None,
    embedding_provider: FakeEmbeddingProvider | None = None,
    extra_providers: dict[str, object] | None = None,
    event_log=None,
) -> ModelRouter:
    settings = _settings()
    providers: dict[str, object] = {
        "local_openai_compatible": model_provider or FakeModelProvider(),
        "local_embedding": embedding_provider or FakeEmbeddingProvider(),
    }
    providers.update(extra_providers or {})
    return ModelRouter(
        settings=settings,
        policy=policy or ConfigPolicyEngine(settings),
        invocation_repository=model_repo,
        providers=providers,
        event_log=event_log or InMemoryEventLog(),
    )


def test_model_router_calls_policy_before_provider(model_repo) -> None:
    async def scenario() -> list[str]:
        call_log: list[str] = []
        provider = FakeModelProvider(chat_response="ok", call_log=call_log)
        router = _router(
            model_repo,
            policy=RecordingPolicy(call_log),
            model_provider=provider,
        )
        await router.chat(_chat_request())
        return call_log

    assert asyncio.run(scenario()) == ["policy.model", "provider.chat"]


def test_cloud_reasoning_denied(model_repo) -> None:
    async def scenario() -> FakeModelProvider:
        provider = FakeModelProvider()
        router = _router(model_repo, model_provider=provider)
        with pytest.raises(ModelPolicyDenied):
            await router.chat(
                _chat_request(profile="cloud_reasoning", sensitivity=Sensitivity.PUBLIC),
            )
        return provider

    provider = asyncio.run(scenario())

    assert provider.chat_calls == 0


def test_secret_sensitivity_denied(model_repo) -> None:
    async def scenario() -> FakeModelProvider:
        provider = FakeModelProvider()
        router = _router(model_repo, model_provider=provider)
        with pytest.raises(ModelPolicyDenied):
            await router.chat(_chat_request(sensitivity=Sensitivity.SECRET))
        return provider

    provider = asyncio.run(scenario())

    assert provider.chat_calls == 0


def test_model_policy_denial_records_policy_decision(model_repo) -> None:
    async def scenario():
        provider = FakeModelProvider()
        event_log = InMemoryEventLog()
        router = _router(model_repo, model_provider=provider, event_log=event_log)
        with pytest.raises(ModelPolicyDenied):
            await router.chat(_chat_request(sensitivity=Sensitivity.SECRET))
        return provider, await event_log.query(EventFilter(request_id=_chat_request().request_id))

    provider, events = asyncio.run(scenario())

    assert provider.chat_calls == 0
    policy_event = next(
        event for event in events if event.event_type == EventType.POLICY_DECISION_RECORDED
    )
    assert policy_event.payload["allowed"] is False
    assert policy_event.payload["source_ref"] == "model_request:local_main"
    denied_event = next(event for event in events if event.event_type == EventType.MODEL_REQUEST_DENIED)
    assert denied_event.payload["code"] == "sensitivity_denied"


def test_model_policy_allow_records_policy_decision(model_repo) -> None:
    async def scenario():
        event_log = InMemoryEventLog()
        router = _router(
            model_repo,
            model_provider=FakeModelProvider(chat_response="allowed"),
            event_log=event_log,
        )
        await router.chat(_chat_request())
        return await event_log.query(EventFilter(request_id=_chat_request().request_id))

    events = asyncio.run(scenario())

    policy_event = next(
        event for event in events if event.event_type == EventType.POLICY_DECISION_RECORDED
    )
    assert policy_event.payload["allowed"] is True
    assert policy_event.payload["source_ref"] == "model_request:local_main"


def test_chat_creates_model_invocation(model_repo) -> None:
    async def scenario():
        router = _router(model_repo, model_provider=FakeModelProvider(chat_response="done"))
        response = await router.chat(_chat_request())
        invocations = await model_repo.list_recent(limit=10)
        return response, invocations

    response, invocations = asyncio.run(scenario())

    assert response.text == "done"
    assert len(invocations) == 1
    assert invocations[0].profile == "local_main"
    assert invocations[0].purpose == "chat"
    assert invocations[0].status == "completed"
    assert invocations[0].streaming is False


def test_profile_specific_provider_overrides_shared_provider(model_repo) -> None:
    async def scenario():
        shared_provider = FakeModelProvider(chat_response="shared")
        profile_provider = FakeModelProvider(chat_response="profile")
        router = _router(
            model_repo,
            model_provider=shared_provider,
            extra_providers={"local_main": profile_provider},
        )
        response = await router.chat(_chat_request(profile="local_main"))
        return response, shared_provider, profile_provider

    response, shared_provider, profile_provider = asyncio.run(scenario())

    assert response.text == "profile"
    assert shared_provider.chat_calls == 0
    assert profile_provider.chat_calls == 1


def test_stream_chat_emits_normalized_tokens(model_repo) -> None:
    async def scenario():
        provider = FakeModelProvider(stream_tokens=["A", "B"])
        router = _router(model_repo, model_provider=provider)
        return [event async for event in router.stream_chat(_chat_request())]

    events = asyncio.run(scenario())

    assert [(event.event_type, event.delta) for event in events] == [
        ("token", "A"),
        ("token", "B"),
    ]


def test_chat_retry_zero(model_repo) -> None:
    async def scenario() -> FakeModelProvider:
        provider = FakeModelProvider(fail_chat_times=1)
        router = _router(model_repo, model_provider=provider)
        with pytest.raises(ModelProviderError):
            await router.chat(_chat_request())
        return provider

    provider = asyncio.run(scenario())

    assert provider.chat_calls == 1


def test_structured_invalid_json_retries_once(model_repo) -> None:
    async def scenario():
        provider = FakeModelProvider(structured_text_responses=["not json", '{"ok": true}'])
        router = _router(model_repo, model_provider=provider)
        response = await router.structured(_structured_request())
        return response, provider

    response, provider = asyncio.run(scenario())

    assert response.value == {"ok": True}
    assert provider.structured_calls == 2


def test_embedding_retries_once(model_repo) -> None:
    async def scenario():
        provider = FakeEmbeddingProvider(fail_embed_times=1, vectors=[[0.1, 0.2]])
        router = _router(model_repo, embedding_provider=provider)
        response = await router.embed(_embedding_request())
        return response, provider

    response, provider = asyncio.run(scenario())

    assert response.vectors == [[0.1, 0.2]]
    assert provider.embed_calls == 2


def test_no_automatic_fallback(model_repo) -> None:
    async def scenario():
        local_provider = FakeModelProvider(fail_chat_times=1)
        cloud_provider = FakeModelProvider(chat_response="cloud")
        router = _router(
            model_repo,
            model_provider=local_provider,
            extra_providers={"openai": cloud_provider},
        )
        with pytest.raises(ModelProviderError):
            await router.chat(_chat_request())
        return local_provider, cloud_provider

    local_provider, cloud_provider = asyncio.run(scenario())

    assert local_provider.chat_calls == 1
    assert cloud_provider.chat_calls == 0
