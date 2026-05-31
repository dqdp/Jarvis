from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dev" / "jarvis_runtime.py"
COMPOSE_PATH = PROJECT_ROOT / "infra" / "compose" / "jarvis-postgres.yml"
pytestmark = pytest.mark.architecture


def test_jarvis_runtime_compose_uses_persistent_database_not_test_database() -> None:
    assert COMPOSE_PATH.is_file(), "infra/compose/jarvis-postgres.yml must exist"
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))

    service = compose["services"]["postgres-jarvis"]

    assert service["environment"]["POSTGRES_DB"] == "jarvis_local"
    assert service["ports"] == ["55433:5432"]
    assert "jarvis_test" not in COMPOSE_PATH.read_text(encoding="utf-8")
    assert "55432" not in COMPOSE_PATH.read_text(encoding="utf-8")
    assert "jarvis-postgres-data:/var/lib/postgresql/data" in service["volumes"]
    assert "jarvis-postgres-data" in compose["volumes"]


def test_makefile_jarvis_targets_delegate_to_single_startup_script() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "JARVIS_RUNTIME ?= $(PYTHON) scripts/dev/jarvis_runtime.py" in makefile
    for target in [
        "jarvis-bootstrap",
        "jarvis-up",
        "jarvis-cli",
        "jarvis-status",
        "jarvis-logs",
        "jarvis-down",
        "jarvis-reset",
    ]:
        start = makefile.index(f"{target}:")
        end = makefile.find("\n\n", start)
        block = makefile[start:] if end == -1 else makefile[start:end]
        assert "$(JARVIS_RUNTIME)" in block, target

    assert "dogfood-" not in makefile


def test_jarvis_startup_script_is_operational_glue_not_runtime_composition() -> None:
    assert SCRIPT_PATH.is_file(), "scripts/dev/jarvis_runtime.py must exist"
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "assistant_core.app_factory",
        "assistant_core.runtime",
        "assistant_core.storage",
        "assistant_core.models",
        "assistant_core.tools",
        "openai",
        "ollama",
        "vllm",
    }
    violations = [
        module
        for module in sorted(imported)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    ]

    assert violations == []


def test_jarvis_startup_script_has_no_secret_defaults() -> None:
    assert SCRIPT_PATH.is_file(), "scripts/dev/jarvis_runtime.py must exist"
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()

    assert "openai_api_key" not in source
    assert "sk-" not in source
    assert "api_key=" not in source
    assert "password" not in source.replace("postgresql+asyncpg://jarvis:jarvis", "")
