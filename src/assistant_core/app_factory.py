from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI

from assistant_core.api.app import create_app
from assistant_core.config.settings import ConfigLoader, Settings
from assistant_core.content_retrieval.project_docs import (
    MarkdownChunker,
    ProjectDocsIngestionService,
    ProjectDocsSourceScanner,
)
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.models.embedding_port import ModelRouterEmbeddingPort
from assistant_core.models.local_openai import (
    HttpxOpenAICompatibleTransport,
    LocalOpenAICompatibleProviderAdapter,
    OpenAICompatibleTransport,
)
from assistant_core.models.ollama import (
    HttpxOllamaTransport,
    OllamaProviderAdapter,
    OllamaTransport,
)
from assistant_core.models.router import ModelRouter
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.runtime.agent_runtime import AgentRuntime
from assistant_core.runtime.loops import LoopStrategyRegistry, MemoryAugmentedAnswerLoop
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.approval_store import PostgresApprovalStore
from assistant_core.storage.content_store import PostgresContentStore
from assistant_core.storage.database import create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository
from assistant_core.tools.builtin import calculator_tool, daemon_status_tool, datetime_now_tool
from assistant_core.tools.fake import fake_echo_tool, fake_fail_tool, fake_timeout_tool
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry
from assistant_core.tools.shell_read import project_shell_read_tool_from_config
from assistant_core.tools.system_diagnostics import system_diagnostics_tools_from_config


class AsyncDisposable(Protocol):
    async def dispose(self) -> None: ...


@dataclass(frozen=True)
class RuntimeApplication:
    app: FastAPI
    engine: AsyncDisposable
    settings: Settings
    runtime: AgentRuntime

    async def dispose(self) -> None:
        await _shutdown_request_execution_manager(self.app)
        await self.engine.dispose()


def create_runtime_app(
    *,
    database_url: str,
    settings: Settings,
    providers: dict[str, object] | None = None,
    project_root: Path | None = None,
    run_database_migrations: bool = False,
) -> RuntimeApplication:
    if run_database_migrations:
        run_migrations(database_url)

    engine = create_database_engine(database_url)
    conversation_store = PostgresConversationStore(engine)
    event_log = PostgresEventLog(engine)
    approval_store = PostgresApprovalStore(engine, event_log=event_log)
    policy = ConfigPolicyEngine(settings, event_log=event_log)
    invocation_repository = PostgresModelInvocationRepository(engine)
    router = ModelRouter(
        settings=settings,
        policy=policy,
        invocation_repository=invocation_repository,
        providers=providers if providers is not None else build_local_providers(settings),
        event_log=event_log,
    )
    memory_store = PostgresMemoryStore(
        engine=engine,
        settings=settings,
        policy=policy,
        embedding_port=ModelRouterEmbeddingPort(router=router, profile="local_embedding"),
    )
    content_store = PostgresContentStore(
        engine=engine,
        settings=settings,
        embedding_port=ModelRouterEmbeddingPort(router=router, profile="local_embedding"),
    )
    content_project_root = (project_root or Path(os.environ.get("JARVIS_PROJECT_ROOT", "."))).resolve()
    content_ingestion = ProjectDocsIngestionService(
        store=content_store,
        scanner=ProjectDocsSourceScanner(project_root=content_project_root),
        chunker=MarkdownChunker(),
    )
    context_assembler = DeterministicContextAssembler(
        conversation_store=conversation_store,
        memory_read=memory_store,
        content_retrieval=content_store,
        event_log=event_log,
        policy=policy,
    )
    tool_gateway = ToolGateway(
        registry=ToolRegistry(
            [
                fake_echo_tool(),
                fake_fail_tool(),
                fake_timeout_tool(),
                datetime_now_tool(),
                calculator_tool(),
                daemon_status_tool(),
                project_shell_read_tool_from_config(settings.capabilities),
                *system_diagnostics_tools_from_config(settings.capabilities),
            ],
        ),
        policy=policy,
        event_log=event_log,
        approval_store=approval_store,
    )
    runtime = AgentRuntime(
        conversation_store=conversation_store,
        context_assembler=context_assembler,
        model_router=router,
        event_log=event_log,
        settings=settings,
        loop_strategy_registry=LoopStrategyRegistry(
            [
                MemoryAugmentedAnswerLoop(
                    conversation_store=conversation_store,
                    context_assembler=context_assembler,
                    model_router=router,
                    event_log=event_log,
                ),
                ToolReactLoop(
                    conversation_store=conversation_store,
                    context_assembler=context_assembler,
                    model_router=router,
                    event_log=event_log,
                    tool_gateway=tool_gateway,
                    approval_store=approval_store,
                ),
            ],
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await _shutdown_request_execution_manager(_app)
            await engine.dispose()

    app = create_app(
        conversation_store=conversation_store,
        memory_store=memory_store,
        content_store=content_store,
        settings=settings,
        runtime=runtime,
        event_log=event_log,
        approval_store=approval_store,
        inference_health=router,
        content_ingestion=content_ingestion,
        policy=policy,
        lifespan=lifespan,
    )
    runtime_app = RuntimeApplication(app=app, engine=engine, settings=settings, runtime=runtime)
    app.state.runtime_application = runtime_app

    return runtime_app


async def _shutdown_request_execution_manager(app: FastAPI) -> None:
    manager = getattr(app.state, "request_execution_manager", None)
    shutdown = getattr(manager, "shutdown", None)
    if shutdown is not None:
        await shutdown()


def build_local_providers(
    settings: Settings,
    *,
    transport: OpenAICompatibleTransport | None = None,
    ollama_transport: OllamaTransport | None = None,
) -> dict[str, object]:
    local_transport = transport or HttpxOpenAICompatibleTransport()
    local_ollama_transport = ollama_transport or HttpxOllamaTransport()
    providers: dict[str, object] = {}
    for profile_name, profile in settings.model_profiles.items():
        if not profile.enabled or profile.cloud:
            continue
        if profile.provider in {"local_openai_compatible", "local_embedding"}:
            providers[profile_name] = LocalOpenAICompatibleProviderAdapter(
                profile=profile,
                transport=local_transport,
            )
        elif profile.provider == "ollama":
            providers[profile_name] = OllamaProviderAdapter(
                profile=profile,
                transport=local_ollama_transport,
            )
    return providers


def create_asgi_app() -> FastAPI:
    config_dir = Path(os.environ.get("JARVIS_CONFIG_DIR", "config"))
    profile = os.environ.get("JARVIS_CONFIG_PROFILE", "default")
    database_url = os.environ["DATABASE_URL"]
    settings = ConfigLoader(config_dir).load(profile)
    run_startup_migrations = _env_bool(os.environ.get("JARVIS_RUN_MIGRATIONS_ON_STARTUP"))
    return create_runtime_app(
        database_url=database_url,
        settings=settings,
        run_database_migrations=run_startup_migrations,
    ).app


def _env_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
