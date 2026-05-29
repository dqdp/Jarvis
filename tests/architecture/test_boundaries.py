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


def test_agent_runtime_does_not_import_subprocess() -> None:
    _assert_no_import_prefixes([SRC_ROOT / "runtime" / "agent_runtime.py"], {"subprocess"})


def test_loop_strategies_do_not_import_subprocess() -> None:
    _assert_no_import_prefixes(_python_files("runtime/loops"), {"subprocess"})


def test_only_shell_adapter_executes_subprocess() -> None:
    offenders: list[str] = []
    allowed = SRC_ROOT / "tools" / "shell_read.py"
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        imported = _imported_modules(path)
        uses_subprocess = (
            "subprocess" in imported
            or "asyncio.subprocess" in imported
            or "create_subprocess_exec" in source
        )
        if uses_subprocess and path != allowed:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_toolgateway_consults_policy_before_shell_execution() -> None:
    gateway_path = SRC_ROOT / "tools" / "gateway.py"
    source = gateway_path.read_text(encoding="utf-8")

    policy_index = source.index("evaluate_capability_request")
    execute_index = source.index("_execute_adapter(")

    assert policy_index < execute_index
