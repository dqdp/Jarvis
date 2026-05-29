from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy import text

from assistant_core.api.app import create_app
from assistant_core.config.settings import ConfigLoader
from assistant_core.content_retrieval.project_docs import (
    MarkdownChunker,
    ProjectDocsIngestionService,
    ProjectDocsSourceScanner,
)
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.domain.events import EventType
from assistant_core.domain.models import EmbeddingResponse
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.models.fake_provider import FakeModelProvider
from assistant_core.models.router import ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.ports.embedding import GenerateEmbeddingCommand
from assistant_core.ports.event_log import EventFilter
from assistant_core.runtime.agent_runtime import AgentRuntime
from assistant_core.storage.content_store import PostgresContentStore
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.database import assert_test_database_url, create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository


pytestmark = pytest.mark.e2e


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


class RecordingEmbeddingPort:
    def __init__(self) -> None:
        self.calls: list[GenerateEmbeddingCommand] = []

    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse:
        self.calls.append(command)
        return EmbeddingResponse(vectors=[_vector(text) for text in command.texts])


async def _request(app, method: str, path: str, body: dict[str, Any] | None = None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        kwargs: dict[str, Any] = {"json": body} if body is not None else {}
        response = await client.request(method, path, **kwargs)
    return response.status_code, response.text


async def _truncate_e2e(database_url: str) -> None:
    assert_test_database_url(database_url)
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "truncate table content_embeddings, content_chunks, content_sources, "
                    "memory_embeddings, memory_candidates, memories, model_invocations, "
                    "assistant_requests, messages, conversations, events "
                    "restart identity cascade",
                ),
            )
    finally:
        await engine.dispose()


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float(lowered.count("project")),
        float(lowered.count("docs")),
        float(lowered.count("citation")),
        1.0,
    ]


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_project_docs_turn(tmp_path: Path, *, message_sensitivity: str = "project"):
    database_url = _database_url()
    assert_test_database_url(database_url)
    run_migrations(database_url)

    async def scenario():
        await _truncate_e2e(database_url)
        _write(tmp_path, "docs/guide.md", "# Guide\nproject docs require citation\n")
        engine = create_database_engine(database_url)
        settings = ConfigLoader(Path("config")).load("test")
        policy = ConfigPolicyEngine(settings)
        event_log = PostgresEventLog(engine)
        conversation_store = PostgresConversationStore(engine)
        embedding_port = RecordingEmbeddingPort()
        memory_store = PostgresMemoryStore(
            engine=engine,
            settings=settings,
            policy=policy,
            embedding_port=embedding_port,
        )
        content_store = PostgresContentStore(
            engine=engine,
            embedding_port=embedding_port,
        )
        invocations = PostgresModelInvocationRepository(engine)
        router = ModelRouter(
            settings=settings,
            policy=policy,
            invocation_repository=invocations,
            event_log=event_log,
            providers={
                "local_openai_compatible": FakeModelProvider(
                    chat_response="Use docs/guide.md:1-2",
                    stream_tokens=["Use docs/guide.md:1-2"],
                ),
            },
        )
        await ProjectDocsIngestionService(
            store=content_store,
            scanner=ProjectDocsSourceScanner(project_root=tmp_path),
            chunker=MarkdownChunker(max_chars=160),
        ).ingest()
        runtime = AgentRuntime(
            conversation_store=conversation_store,
            context_assembler=DeterministicContextAssembler(
                conversation_store=conversation_store,
                memory_read=memory_store,
                content_retrieval=content_store,
                event_log=event_log,
                policy=policy,
            ),
            model_router=router,
            event_log=event_log,
            settings=settings,
        )
        app = create_app(
            conversation_store=conversation_store,
            memory_store=memory_store,
            content_store=content_store,
            settings=settings,
            runtime=runtime,
            event_log=event_log,
        )
        assert isinstance(app, FastAPI)
        try:
            _, conversation_raw = await _request(
                app,
                "POST",
                "/v1/conversations",
                {"title": "rag e2e", "active_project_namespace": "project.personal_assistant"},
            )
            conversation = json.loads(conversation_raw)
            _, message_raw = await _request(
                app,
                "POST",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
                {
                    "client_message_id": "client-rag-e2e",
                    "content": "What do project docs say about citation?",
                    "sensitivity": message_sensitivity,
                },
            )
            submitted = json.loads(message_raw)
            for _ in range(100):
                _, request_raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}")
                if json.loads(request_raw)["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
            _, request_raw = await _request(app, "GET", f"/v1/requests/{submitted['request_id']}")
            _, messages_raw = await _request(
                app,
                "GET",
                f"/v1/conversations/{conversation['conversation_id']}/messages",
            )
            events = await event_log.query(EventFilter(request_id=submitted["request_id"]))
            return json.loads(request_raw), json.loads(messages_raw), events, embedding_port.calls
        finally:
            await engine.dispose()

    return asyncio.run(scenario())


def test_agent_answers_from_project_doc_content_hit_with_citation(tmp_path: Path) -> None:
    request_status, messages, events, embedding_calls = _run_project_docs_turn(tmp_path)

    assert request_status["status"] == "completed"
    assert messages["messages"][-1]["content"] == "Use docs/guide.md:1-2"
    context_event = next(event for event in events if event.event_type == EventType.CONTEXT_ASSEMBLED)
    retrieved_event = next(event for event in events if event.event_type == EventType.CONTENT_RETRIEVED)
    assert context_event.payload["used_content_refs"][0]["citation"] == "docs/guide.md:1-2"
    assert retrieved_event.payload["retrieved_content_refs"][0]["citation"] == "docs/guide.md:1-2"
    assert "used_content_refs" not in retrieved_event.payload
    assert retrieved_event.request_id == context_event.request_id
    assert retrieved_event.conversation_id == context_event.conversation_id
    assert retrieved_event.correlation_id == context_event.request_id
    assert [call.texts for call in embedding_calls] == [
        ["# Guide\nproject docs require citation"],
        ["What do project docs say about citation?"],
    ]


def test_agent_does_not_treat_content_hit_as_memory(tmp_path: Path) -> None:
    _, _, events, _ = _run_project_docs_turn(tmp_path)

    context_event = next(event for event in events if event.event_type == EventType.CONTEXT_ASSEMBLED)
    assert context_event.payload["used_memory_ids"] == []
    assert context_event.payload["used_content_refs"]
    assert "chunk_id" in context_event.payload["used_content_refs"][0]


def test_content_retrieved_event_uses_query_sensitivity(tmp_path: Path) -> None:
    _, _, events, _ = _run_project_docs_turn(tmp_path, message_sensitivity="personal")

    retrieved_event = next(event for event in events if event.event_type == EventType.CONTENT_RETRIEVED)
    assert retrieved_event.sensitivity is Sensitivity.PERSONAL
