from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "assistant_core"
pytestmark = pytest.mark.architecture


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _python_files(package: str) -> list[Path]:
    root = SRC_ROOT / package
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _assert_no_import_prefixes(files: list[Path], forbidden_prefixes: set[str]) -> None:
    violations: list[str] = []
    for path in files:
        for module in _imported_modules(path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            ):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {module}")

    assert violations == []


def test_domain_does_not_import_adapters() -> None:
    _assert_no_import_prefixes(
        _python_files("domain"),
        {
            "assistant_core.adapters",
            "assistant_core.storage",
            "sqlalchemy",
            "pgvector",
        },
    )


def test_domain_does_not_import_api() -> None:
    _assert_no_import_prefixes(
        _python_files("domain"),
        {"assistant_core.api"},
    )


def test_domain_does_not_import_runtime() -> None:
    _assert_no_import_prefixes(
        _python_files("domain"),
        {"assistant_core.runtime"},
    )


def test_ports_do_not_expose_adapter_types() -> None:
    _assert_no_import_prefixes(
        _python_files("ports"),
        {
            "assistant_core.adapters",
            "assistant_core.storage",
            "sqlalchemy",
            "pgvector",
            "openai",
            "ollama",
            "vllm",
        },
    )


def test_runtime_does_not_import_storage_adapters() -> None:
    runtime_files = _python_files("runtime")
    if not runtime_files:
        return

    _assert_no_import_prefixes(
        runtime_files,
        {
            "assistant_core.storage",
            "sqlalchemy",
            "pgvector",
        },
    )


def test_runtime_does_not_import_provider_clients() -> None:
    _assert_no_import_prefixes(
        _python_files("runtime"),
        {
            "openai",
            "ollama",
            "vllm",
            "httpx",
            "urllib.request",
        },
    )


def test_context_assembler_does_not_import_provider_clients() -> None:
    _assert_no_import_prefixes(
        _python_files("context_assembly"),
        {
            "openai",
            "ollama",
            "vllm",
            "httpx",
            "urllib.request",
            "assistant_core.models",
        },
    )


def test_only_storage_imports_sqlalchemy() -> None:
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.relative_to(SRC_ROOT).parts[0] == "storage":
            continue
        for module in _imported_modules(path):
            if module == "sqlalchemy" or module.startswith("sqlalchemy."):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {module}")

    assert offenders == []


def test_contract_tests_guard_database_url_before_migrations() -> None:
    offenders: list[str] = []
    guarded_test_roots = [
        PROJECT_ROOT / "tests" / "contract",
        PROJECT_ROOT / "tests" / "e2e",
        PROJECT_ROOT / "tests" / "integration",
    ]
    for root in guarded_test_roots:
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            )
            for function in functions:
                guarded_names: set[str] = set()
                guarded_any = False
                calls = sorted(
                    (
                        node
                        for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                    ),
                    key=lambda node: (node.lineno, node.col_offset),
                )
                for call in calls:
                    call_name = _call_name(call)
                    first_arg_name = _first_arg_name(call)
                    if call_name == "assert_test_database_url":
                        if first_arg_name is not None:
                            guarded_names.add(first_arg_name)
                        guarded_any = True
                        continue
                    if call_name == "run_migrations":
                        if first_arg_name is None or first_arg_name not in guarded_names:
                            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{call.lineno}")
                    if call_name == "command.upgrade" and not guarded_any:
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{call.lineno}")

    assert offenders == []


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts = [call.func.attr]
        value = call.func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _first_arg_name(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Name):
        return first.id
    return None


def test_runtime_does_not_construct_prompt_from_context_sections() -> None:
    runtime_path = SRC_ROOT / "runtime" / "agent_runtime.py"
    source = runtime_path.read_text(encoding="utf-8")

    assert "MODEL_PROMPT_SECTION_NAMES" not in source
    assert "_model_prompt_messages" not in source
    assert "context.sections" not in source


def test_no_pgvector_import_outside_storage_or_memory_adapter() -> None:
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        top_package = path.relative_to(SRC_ROOT).parts[0]
        if top_package in {"storage", "memory"}:
            continue
        if "pgvector" in _imported_modules(path):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_model_router_uses_policy_port() -> None:
    router_path = SRC_ROOT / "models" / "router.py"
    imported = _imported_modules(router_path)
    source = router_path.read_text(encoding="utf-8")

    assert "assistant_core.ports.policy" in imported
    assert "evaluate_model_request" in source


def test_no_raw_prompt_logging_by_default() -> None:
    from assistant_core.config.settings import ConfigLoader

    settings = ConfigLoader(PROJECT_ROOT / "config").load("test")

    assert settings.privacy.raw_prompt_logging is False
    assert settings.context_assembly.full_prompt_logging is False
    assert settings.observability.log_raw_prompts is False


def test_no_mvp_scope_creep_packages_exist() -> None:
    forbidden_packages = {"mcp", "rag", "react", "planner", "voice"}

    assert sorted(
        package
        for package in forbidden_packages
        if (SRC_ROOT / package).exists()
    ) == []


def test_toolgateway_package_exists_for_pm02() -> None:
    assert (SRC_ROOT / "tools").exists()
    assert (SRC_ROOT / "ports" / "tools.py").exists()


def test_project_shell_read_adapter_exists_without_write_shell_for_pm06a() -> None:
    assert (SRC_ROOT / "tools" / "shell_read.py").exists()
    assert not (SRC_ROOT / "tools" / "shell_write.py").exists()
    assert not (SRC_ROOT / "tools" / "shell_write").exists()


def test_runtime_does_not_import_tool_or_shell_adapters() -> None:
    _assert_no_import_prefixes(
        [
            path
            for path in _python_files("runtime")
            if path.name != "tool_react.py"
        ],
        {
            "assistant_core.toolgateway",
            "assistant_core.tools",
            "assistant_core.adapters.shell",
            "subprocess",
        },
    )


def test_model_router_does_not_execute_tools() -> None:
    _assert_no_import_prefixes(
        _python_files("models"),
        {
            "assistant_core.tools",
            "assistant_core.ports.tools",
        },
    )


def test_model_provider_adapters_do_not_import_router_facade() -> None:
    adapter_paths = [
        SRC_ROOT / "models" / "fake_provider.py",
        SRC_ROOT / "models" / "local_openai.py",
        SRC_ROOT / "models" / "ollama.py",
    ]

    _assert_no_import_prefixes(adapter_paths, {"assistant_core.models.router"})
    assert (SRC_ROOT / "ports" / "model_provider.py").is_file()


def test_model_router_uses_model_provider_port_contract() -> None:
    source = (SRC_ROOT / "models" / "router.py").read_text(encoding="utf-8")

    assert "ModelProviderPort" in source
    assert "providers: dict[str, object]" not in source
    assert "type: ignore[attr-defined]" not in source


def test_context_assembler_does_not_execute_tools() -> None:
    _assert_no_import_prefixes(
        _python_files("context_assembly"),
        {
            "assistant_core.tools",
            "assistant_core.ports.tools",
        },
    )


def test_cli_api_do_not_import_concrete_tool_adapters() -> None:
    _assert_no_import_prefixes(
        _python_files("api") + [SRC_ROOT / "cli.py"],
        {
            "assistant_core.tools.fake",
            "assistant_core.tools.builtin",
            "assistant_core.tools.gateway",
            "assistant_core.tools.registry",
        },
    )


def test_api_facade_does_not_own_request_execution_runtime() -> None:
    app_path = SRC_ROOT / "api" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert "class _RequestExecutionManager" not in source
    assert "class RequestExecutionManager" not in source
    assert "def _resolve_loop_strategy" not in source
    assert "def _resolve_model_profile" not in source
    assert (SRC_ROOT / "runtime" / "request_execution.py").is_file()
    assert (SRC_ROOT / "runtime" / "request_metadata.py").is_file()
    assert (SRC_ROOT / "runtime" / "request_streaming.py").is_file()
    assert (SRC_ROOT / "runtime" / "request_stream_buffer.py").is_file()
    assert (SRC_ROOT / "runtime" / "request_command.py").is_file()
    assert (SRC_ROOT / "runtime" / "request_lifecycle.py").is_file()


def test_request_execution_manager_delegates_stateful_subsystems() -> None:
    source = (SRC_ROOT / "runtime" / "request_execution.py").read_text(encoding="utf-8")

    assert "class RequestStreamBuffer" not in source
    assert "class RuntimeTurnCommandBuilder" not in source
    assert "class RequestLifecycleService" not in source
    assert "self._events:" not in source
    assert "self._conditions:" not in source
    assert "self._active_streams:" not in source
    assert "from assistant_core.runtime.agent_runtime import RuntimeTurnCommand" not in source


def test_api_has_split_transport_components() -> None:
    expected_modules = [
        SRC_ROOT / "api" / "errors.py",
        SRC_ROOT / "api" / "health.py",
        SRC_ROOT / "api" / "presenters.py",
        SRC_ROOT / "api" / "sse.py",
    ]

    assert [path.name for path in expected_modules if not path.is_file()] == []


def test_cli_entrypoint_delegates_to_cli_app_components() -> None:
    entrypoint = SRC_ROOT / "cli.py"
    source = entrypoint.read_text(encoding="utf-8")
    expected_modules = [
        SRC_ROOT / "cli_app" / "client.py",
        SRC_ROOT / "cli_app" / "config.py",
        SRC_ROOT / "cli_app" / "commands.py",
        SRC_ROOT / "cli_app" / "chat_flow.py",
        SRC_ROOT / "cli_app" / "interactive.py",
        SRC_ROOT / "cli_app" / "line_reader.py",
        SRC_ROOT / "cli_app" / "renderers.py",
        SRC_ROOT / "cli_app" / "sse.py",
        SRC_ROOT / "cli_app" / "utils.py",
    ]

    assert [path.name for path in expected_modules if not path.is_file()] == []
    assert "class HttpJarvisClient" not in source
    assert "class TerminalInteractiveLineReader" not in source
    assert "async def run_interactive_chat" not in source

    assert not (SRC_ROOT / "cli_app" / "core.py").exists()


def test_toolgateway_does_not_import_loop_strategies() -> None:
    if not (SRC_ROOT / "tools").exists():
        pytest.fail("PM-02 requires the tools package")
    _assert_no_import_prefixes(
        _python_files("tools"),
        {
            "assistant_core.runtime.loops",
            "assistant_core.loops",
        },
    )


def test_toolgateway_is_thin_facade_over_internal_services() -> None:
    expected_modules = [
        SRC_ROOT / "tools" / "authorization.py",
        SRC_ROOT / "tools" / "approval_flow.py",
        SRC_ROOT / "tools" / "approval_coordination.py",
        SRC_ROOT / "tools" / "audit.py",
        SRC_ROOT / "tools" / "execution.py",
        SRC_ROOT / "tools" / "events.py",
        SRC_ROOT / "tools" / "results.py",
    ]

    assert [path.name for path in expected_modules if not path.is_file()] == []
    gateway_source = (SRC_ROOT / "tools" / "gateway.py").read_text(encoding="utf-8")
    execution_source = (SRC_ROOT / "tools" / "execution.py").read_text(encoding="utf-8")

    assert "async def _execute_adapter" not in gateway_source
    assert "async def execute_adapter" in execution_source
    assert "CreateApprovalCommand" not in gateway_source
    assert "EventEnvelope(" not in gateway_source


def test_tool_react_loop_delegates_approval_and_proposal_execution() -> None:
    loop_source = (SRC_ROOT / "runtime" / "loops" / "tool_react.py").read_text(encoding="utf-8")

    assert (SRC_ROOT / "runtime" / "loops" / "tool_approval.py").is_file()
    assert (SRC_ROOT / "runtime" / "loops" / "tool_proposal_executor.py").is_file()
    assert "async def _wait_for_approval" not in loop_source
    assert "async def _execute_tool_proposal" not in loop_source


def test_loop_strategies_do_not_import_storage_adapters() -> None:
    loop_files = _python_files("runtime/loops")
    if not loop_files:
        pytest.fail("PM-03 requires runtime loop strategies")
    _assert_no_import_prefixes(
        loop_files,
        {
            "assistant_core.storage",
            "sqlalchemy",
            "pgvector",
        },
    )


def test_loop_strategies_do_not_import_provider_clients() -> None:
    loop_files = _python_files("runtime/loops")
    if not loop_files:
        pytest.fail("PM-03 requires runtime loop strategies")
    _assert_no_import_prefixes(
        loop_files,
        {
            "openai",
            "ollama",
            "vllm",
            "httpx",
            "urllib.request",
        },
    )


def test_memory_augmented_answer_loop_does_not_import_toolgateway() -> None:
    loop_path = SRC_ROOT / "runtime" / "loops" / "memory_augmented_answer.py"
    assert loop_path.is_file()
    _assert_no_import_prefixes(
        [loop_path],
        {
            "assistant_core.tools",
            "assistant_core.ports.tools",
        },
    )


def test_tool_react_loop_uses_toolgateway_port_not_adapters() -> None:
    loop_path = SRC_ROOT / "runtime" / "loops" / "tool_react.py"
    assert loop_path.is_file()
    imported = _imported_modules(loop_path)

    assert "assistant_core.ports.tools" in imported
    _assert_no_import_prefixes(
        [loop_path],
        {
            "assistant_core.tools.fake",
            "assistant_core.tools.builtin",
            "assistant_core.tools.gateway",
            "assistant_core.tools.registry",
        },
    )


def test_tool_react_loop_does_not_import_shell_mcp_or_integration_adapters() -> None:
    loop_path = SRC_ROOT / "runtime" / "loops" / "tool_react.py"
    assert loop_path.is_file()
    _assert_no_import_prefixes(
        [loop_path],
        {
            "assistant_core.tools.shell",
            "assistant_core.tools.shell_read",
            "assistant_core.mcp",
            "assistant_core.integrations",
            "subprocess",
        },
    )


def test_agent_runtime_does_not_import_diagnostics_adapters() -> None:
    _assert_no_import_prefixes(
        [SRC_ROOT / "runtime" / "agent_runtime.py"],
        {"assistant_core.tools.system_diagnostics"},
    )


def test_loop_strategies_do_not_import_diagnostics_adapters() -> None:
    _assert_no_import_prefixes(
        _python_files("runtime/loops"),
        {"assistant_core.tools.system_diagnostics"},
    )


def test_agent_runtime_does_not_import_subprocess() -> None:
    _assert_no_import_prefixes([SRC_ROOT / "runtime" / "agent_runtime.py"], {"subprocess"})


def test_loop_strategies_do_not_import_subprocess() -> None:
    _assert_no_import_prefixes(_python_files("runtime/loops"), {"subprocess"})


def test_only_shell_or_diagnostics_adapter_executes_subprocess() -> None:
    offenders: list[str] = []
    allowed = {
        SRC_ROOT / "tools" / "shell_read.py",
        SRC_ROOT / "tools" / "system_diagnostics.py",
    }
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        imported = _imported_modules(path)
        uses_subprocess = (
            "subprocess" in imported
            or "asyncio.subprocess" in imported
            or "create_subprocess_exec" in source
        )
        if uses_subprocess and path not in allowed:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_toolgateway_consults_policy_before_shell_execution() -> None:
    gateway_path = SRC_ROOT / "tools" / "gateway.py"
    source = gateway_path.read_text(encoding="utf-8")

    policy_index = source.index("evaluate_capability_request")
    execute_index = source.index("_execute_adapter(")

    assert policy_index < execute_index


def test_toolgateway_consults_policy_before_diagnostics_execution() -> None:
    gateway_path = SRC_ROOT / "tools" / "gateway.py"
    source = gateway_path.read_text(encoding="utf-8")

    policy_index = source.index("evaluate_capability_request")
    execute_index = source.index("_execute_adapter(")

    assert policy_index < execute_index


def test_memory_subsystem_does_not_import_content_retrieval_storage() -> None:
    memory_paths = [
        SRC_ROOT / "domain" / "memory.py",
        SRC_ROOT / "ports" / "memory.py",
        SRC_ROOT / "storage" / "memory_store.py",
    ]

    for path in memory_paths:
        source = path.read_text(encoding="utf-8")
        assert "storage.content_store" not in source
        assert "content_retrieval" not in source


def test_storage_adapters_do_not_import_content_application_services() -> None:
    _assert_no_import_prefixes(
        [SRC_ROOT / "storage" / "content_store.py"],
        {"assistant_core.content_retrieval.project_docs"},
    )


def test_storage_workflows_have_application_service_boundaries() -> None:
    expected_modules = [
        SRC_ROOT / "content_retrieval" / "indexing_service.py",
        SRC_ROOT / "content_retrieval" / "retrieval_service.py",
        SRC_ROOT / "memory" / "write_service.py",
    ]

    assert [path.name for path in expected_modules if not path.is_file()] == []


def test_application_service_boundaries_do_not_contain_placeholders() -> None:
    boundary_modules = [
        SRC_ROOT / "content_retrieval" / "indexing_service.py",
        SRC_ROOT / "content_retrieval" / "retrieval_service.py",
        SRC_ROOT / "memory" / "write_service.py",
    ]

    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in boundary_modules
        if "NotImplementedError" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_runtime_factory_wires_application_workflow_services() -> None:
    source = (SRC_ROOT / "app_factory.py").read_text(encoding="utf-8")

    assert "ContentIndexingService" in source
    assert "ContentRetrievalService" in source
    assert "MemoryWriteService" in source
    assert "content_retrieval=content_retrieval" in source
    assert "memory_write=memory_write" in source


def test_agent_runtime_does_not_import_content_storage_adapters() -> None:
    _assert_no_import_prefixes(
        [SRC_ROOT / "runtime" / "agent_runtime.py"],
        {
            "assistant_core.storage.content_store",
            "assistant_core.storage",
            "sqlalchemy",
            "pgvector",
        },
    )


def test_context_assembler_does_not_import_content_sqlalchemy_models() -> None:
    _assert_no_import_prefixes(
        _python_files("context_assembly"),
        {
            "assistant_core.storage.content_store",
            "assistant_core.storage",
            "sqlalchemy",
            "pgvector",
        },
    )


def test_content_retrieval_does_not_write_memory_tables() -> None:
    forbidden_fragments = [
        "insert into memories",
        "update memories",
        "delete from memories",
        "insert into memory_embeddings",
        "update memory_embeddings",
        "delete from memory_embeddings",
        "_memories",
        "_memory_embeddings",
    ]

    content_paths = [
        *(SRC_ROOT / "content_retrieval").rglob("*.py"),
        SRC_ROOT / "domain" / "content_retrieval.py",
        SRC_ROOT / "storage" / "content_store.py",
    ]
    for path in content_paths:
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        imported = _imported_modules(path)
        assert "assistant_core.storage.memory_store" not in imported
        assert "assistant_core.ports.memory" not in imported
        for fragment in forbidden_fragments:
            assert fragment not in lowered


def test_context_assembler_has_internal_component_boundaries() -> None:
    expected_modules = [
        SRC_ROOT / "context_assembly" / "retrieval.py",
        SRC_ROOT / "context_assembly" / "policy_filter.py",
        SRC_ROOT / "context_assembly" / "rendering.py",
        SRC_ROOT / "context_assembly" / "manifest.py",
        SRC_ROOT / "context_assembly" / "audit.py",
        SRC_ROOT / "context_assembly" / "trimming.py",
    ]

    assert [path.name for path in expected_modules if not path.is_file()] == []


def test_context_assembler_audit_and_trimming_are_real_boundaries() -> None:
    deterministic_source = (
        SRC_ROOT / "context_assembly" / "deterministic.py"
    ).read_text(encoding="utf-8")
    audit_source = (SRC_ROOT / "context_assembly" / "audit.py").read_text(encoding="utf-8")
    trimming_source = (
        SRC_ROOT / "context_assembly" / "trimming.py"
    ).read_text(encoding="utf-8")

    assert "class ContextAssemblyAuditRecorder" in audit_source
    assert "def apply_token_budget" in trimming_source
    assert "def apply_message_count_limit" in trimming_source
    assert "EventEnvelope(" not in deterministic_source
    assert "def _apply_token_budget" not in deterministic_source
    assert "def _apply_message_count_limit" not in deterministic_source


def test_known_facades_stay_below_god_module_size() -> None:
    max_lines_by_path = {
        SRC_ROOT / "api" / "app.py": 650,
        SRC_ROOT / "cli.py": 250,
        SRC_ROOT / "cli_app" / "client.py": 320,
        SRC_ROOT / "cli_app" / "commands.py": 250,
        SRC_ROOT / "cli_app" / "chat_flow.py": 420,
        SRC_ROOT / "cli_app" / "line_reader.py": 300,
        SRC_ROOT / "cli_app" / "renderers.py": 220,
        SRC_ROOT / "tools" / "gateway.py": 850,
        SRC_ROOT / "context_assembly" / "deterministic.py": 780,
    }
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)} has {len(path.read_text(encoding='utf-8').splitlines())} lines"
        for path, max_lines in max_lines_by_path.items()
        if path.is_file() and len(path.read_text(encoding="utf-8").splitlines()) > max_lines
    ]

    assert offenders == []
