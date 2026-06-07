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
from assistant_core.content_retrieval.indexing_service import ContentIndexingService
from assistant_core.content_retrieval.retrieval_service import ContentRetrievalService
from assistant_core.context_assembly.deterministic import DeterministicContextAssembler
from assistant_core.memory.write_service import MemoryWriteService
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
from assistant_core.ports.model_provider import ModelProviderPort
from assistant_core.policy.engine import ConfigPolicyEngine
from assistant_core.runtime.agent_runtime import AgentRuntime
from assistant_core.runtime.loops import LoopStrategyRegistry, MemoryAugmentedAnswerLoop
from assistant_core.runtime.loops.tool_react import ToolReactLoop
from assistant_core.runtime.routing import CapabilityRoutingRegistry
from assistant_core.storage.conversation_store import PostgresConversationStore
from assistant_core.storage.approval_store import PostgresApprovalStore
from assistant_core.storage.content_store import PostgresContentStore
from assistant_core.storage.database import create_database_engine
from assistant_core.storage.event_log import PostgresEventLog
from assistant_core.storage.memory_store import PostgresMemoryStore
from assistant_core.storage.migrations import run_migrations
from assistant_core.storage.model_invocations import PostgresModelInvocationRepository
from assistant_core.tools.builtin import (
    calendar_diff_tool,
    calculator_tool,
    daemon_status_tool,
    datetime_diff_tool,
    datetime_now_tool,
    datetime_until_tool,
)
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
    providers: dict[str, ModelProviderPort] | None = None,
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
    memory_write = MemoryWriteService(memory_store)
    content_store = PostgresContentStore(
        engine=engine,
        settings=settings,
        embedding_port=ModelRouterEmbeddingPort(router=router, profile="local_embedding"),
    )
    content_indexing = ContentIndexingService(content_store)
    content_retrieval = ContentRetrievalService(content_store)
    content_project_root = (project_root or Path(os.environ.get("JARVIS_PROJECT_ROOT", "."))).resolve()
    content_ingestion = ProjectDocsIngestionService(
        store=content_indexing,
        scanner=ProjectDocsSourceScanner(project_root=content_project_root),
        chunker=MarkdownChunker(),
    )
    context_assembler = DeterministicContextAssembler(
        conversation_store=conversation_store,
        memory_read=memory_store,
        content_retrieval=content_retrieval,
        event_log=event_log,
        policy=policy,
        settings=settings,
    )
    tool_registry = ToolRegistry(
        [
            fake_echo_tool(),
            fake_fail_tool(),
            fake_timeout_tool(),
            calendar_diff_tool(),
            datetime_diff_tool(),
            datetime_now_tool(),
            datetime_until_tool(),
            calculator_tool(),
            daemon_status_tool(),
            project_shell_read_tool_from_config(settings.capabilities),
            *system_diagnostics_tools_from_config(settings.capabilities),
        ],
    )
    _validate_request_plan_tool_surface(settings, tool_registry)
    tool_gateway = ToolGateway(
        registry=tool_registry,
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
        memory_write=memory_write,
        content_store=content_store,
        content_retrieval=content_retrieval,
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


def _validate_request_plan_tool_surface(settings: Settings, registry: ToolRegistry) -> None:
    request_plan_tools = {
        item["tool_name"]: item
        for item in CapabilityRoutingRegistry.from_settings(settings).available_tools_summary()
        if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
    }
    gateway_tools = {spec.name: spec for spec in registry.list_specs()}
    missing = sorted(set(request_plan_tools) - set(gateway_tools))
    if missing:
        raise RuntimeError(
            "request-plan tool is not registered in ToolGateway: " + ", ".join(missing)
        )
    drifted = [
        name
        for name, request_plan_tool in request_plan_tools.items()
        if _request_plan_tool_shape(request_plan_tool) != _gateway_tool_shape(gateway_tools[name])
    ]
    if drifted:
        raise RuntimeError(
            "request-plan tool metadata differs from ToolGateway: " + ", ".join(sorted(drifted))
        )


def _request_plan_tool_shape(tool_summary: dict) -> tuple[str | None, tuple[str, ...], str | None]:
    risk_classes = tool_summary.get("risk_classes")
    return (
        tool_summary.get("capability") if isinstance(tool_summary.get("capability"), str) else None,
        tuple(sorted(item for item in risk_classes if isinstance(item, str)))
        if isinstance(risk_classes, list)
        else (),
        (
            tool_summary.get("sensitivity_ceiling")
            if isinstance(tool_summary.get("sensitivity_ceiling"), str)
            else None
        ),
    )


def _gateway_tool_shape(spec) -> tuple[str, tuple[str, ...], str]:
    return (
        spec.capability.value,
        tuple(sorted(risk.value for risk in spec.risk_classes)),
        spec.sensitivity_ceiling.value,
    )


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
) -> dict[str, ModelProviderPort]:
    local_transport = transport or HttpxOpenAICompatibleTransport()
    local_ollama_transport = ollama_transport or HttpxOllamaTransport()
    providers: dict[str, ModelProviderPort] = {}
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
