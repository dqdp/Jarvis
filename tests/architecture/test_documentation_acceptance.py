from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"
README = PROJECT_ROOT / "README.md"
pytestmark = pytest.mark.architecture


def _readme_paths(prefix: str) -> set[str]:
    paths: set[str] = set()
    for raw_line in README.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if prefix in {"ADR-", "docs/adr/"} and line.startswith("ADR-") and line.endswith(".md"):
            paths.add(f"docs/adr/{line}")
        elif line.startswith(prefix) and line.endswith(".md"):
            paths.add(line)
        elif prefix == "docs/" and line.endswith(".md") and line[:2].isdigit():
            paths.add(f"docs/{line}")
    return paths


def test_docs_referenced_adrs_exist() -> None:
    documented_docs = _readme_paths("docs/")
    documented_adrs = _readme_paths("ADR-")
    documented_adrs.update(_readme_paths("docs/adr/"))

    for relative_path in documented_docs:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path
    for relative_path in documented_adrs:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path

    actual_docs = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in DOCS_ROOT.glob("*.md")
        if path.is_file()
    }
    actual_adrs = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (DOCS_ROOT / "adr").glob("*.md")
        if path.is_file()
    }

    assert actual_docs - documented_docs == set()
    assert actual_adrs - documented_adrs == set()


def test_config_matches_documented_defaults() -> None:
    from assistant_core.config.settings import ConfigLoader

    settings = ConfigLoader(PROJECT_ROOT / "config").load("default")

    assert set(settings.model_profiles) >= {
        "local_main",
        "local_structured",
        "local_embedding",
        "cloud_reasoning",
    }
    assert settings.model_profiles["cloud_reasoning"].enabled is False
    assert settings.policy.cloud_models_enabled is False
    assert settings.policy.tools_enabled is False
    assert settings.privacy.raw_prompt_logging is False
    assert settings.context_assembly.full_prompt_logging is False
    assert settings.observability.log_raw_prompts is False

    assert settings.memory.retrieval.max_hits_total == 8
    assert settings.memory.retrieval.max_hits_per_namespace == 4
    assert settings.context_assembly.conversation_window["max_messages"] == 12
    assert settings.context_assembly.conversation_window["max_tokens"] == 3000

    budget = settings.runtime_budgets["memory_augmented_answer"]
    assert budget.max_model_calls == 1
    assert budget.max_tool_calls == 0
    assert budget.allow_cloud is False
    assert budget.allow_tools is False
    assert budget.allow_autonomous_memory_write is False


def test_all_required_ports_have_contract_tests() -> None:
    expected_contract_tests = {
        "tests/contract/test_agent_runtime_contract.py",
        "tests/contract/test_api_lifecycle_contract.py",
        "tests/contract/test_conversation_store_contract.py",
        "tests/contract/test_event_log_contract.py",
        "tests/contract/test_memory_embedding_contract.py",
        "tests/contract/test_memory_read_contract.py",
        "tests/contract/test_memory_write_contract.py",
        "tests/contract/test_model_router_contract.py",
        "tests/contract/test_sse_stream_contract.py",
    }

    missing = {
        relative_path
        for relative_path in expected_contract_tests
        if not (PROJECT_ROOT / relative_path).is_file()
    }

    assert missing == set()


def test_mvp_acceptance_checklist_complete() -> None:
    checklist = DOCS_ROOT / "28_mvp_acceptance_checklist.md"
    unchecked = [
        line
        for line in checklist.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- [ ]")
    ]

    assert unchecked == []


def test_mvp_implementation_archive_exists() -> None:
    archive = DOCS_ROOT / "30_mvp_implementation_archive.md"

    assert archive.is_file()
    archive_text = archive.read_text(encoding="utf-8")
    assert "make test" in archive_text
    assert "Slice 19" in archive_text
    assert "cancellation side effects" in archive_text
    assert "namespace default sensitivity" in archive_text
    assert "policy allow/deny decisions" in archive_text


def test_required_make_targets_are_self_contained() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "test-e2e: test-db-up" in makefile
    assert "DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST) -m e2e tests/e2e" in makefile
    assert "migrate:" in makefile
    assert (
        "PYTHONPATH=$(APP_PYTHONPATH) DATABASE_URL=$(DATABASE_URL) "
        "$(PYTHON) -m assistant_core.storage.migrations"
    ) in makefile
    assert "run:" in makefile
    assert "PYTHONPATH=$(APP_PYTHONPATH) DATABASE_URL=$(DATABASE_URL)" in makefile
    assert "assistant_core.app_factory:create_asgi_app --factory" in makefile
    assert "models-list:" in makefile
    assert "models-pull:" in makefile
    assert "cli:" in makefile
    assert "PYTHONPATH=$(APP_PYTHONPATH) $(PYTHON) -m assistant_core.cli $(ARGS)" in makefile
    assert "ALLOW_EMPTY" not in makefile


def test_native_ollama_provider_is_documented_as_phase_1_adapter() -> None:
    from assistant_core.config.settings import ConfigLoader

    settings = ConfigLoader(PROJECT_ROOT / "config").load("ollama")
    assert settings.model_profiles["local_main"].provider == "ollama"

    model_router_adr = (DOCS_ROOT / "adr" / "ADR-019_model_profiles_and_model_router_baseline.md").read_text(
        encoding="utf-8",
    )
    project_charter = (DOCS_ROOT / "00_project_charter.md").read_text(encoding="utf-8")

    assert "native Ollama adapter" in model_router_adr
    assert "local OpenAI-compatible or native Ollama inference backend" in project_charter


def test_lifecycle_docs_do_not_defer_implemented_cancellation() -> None:
    lifecycle_docs = "\n".join(
        (DOCS_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in [
            "08_data_model_and_storage.md",
            "22_api_shape_and_request_lifecycle.md",
            "23_error_handling_and_runtime_budgets.md",
            "27_tdd_implementation_slices_plan.md",
            "28_mvp_acceptance_checklist.md",
            "adr/ADR-023_api_shape_and_request_lifecycle.md",
            "adr/ADR-024_error_handling_and_runtime_budgets.md",
        ]
    )

    assert "request cancellation implementation" not in lifecycle_docs.lower()
    assert "cancel endpoint is reserved" not in lifecycle_docs.lower()
    assert "websocket\ncancellation" not in lifecycle_docs.lower()
    assert "token replay guarantee\nwebsocket\ncancellation" not in lifecycle_docs.lower()
    assert "running -> cancelled" in lifecycle_docs
    assert "POST /v1/requests/{request_id}/cancel" in lifecycle_docs


def test_docs_do_not_require_graph_runtime_for_mvp() -> None:
    docs_text = "\n".join(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in [
            "README.md",
            "docs/00_project_charter.md",
            "docs/01_target_architecture_overview.md",
            "docs/06_agent_runtime_and_loop_architecture.md",
            "docs/27_tdd_implementation_slices_plan.md",
        ]
    )

    assert "graph-based runtime flow" not in docs_text
    assert "graph flow" not in docs_text
    assert "Runtime: LangGraph" not in docs_text
    assert "Use LangGraph as execution substrate" not in docs_text


def test_testing_docs_use_canonical_runtime_event_chain() -> None:
    testing_doc = (DOCS_ROOT / "26_testing_strategy.md").read_text(encoding="utf-8")

    assert "context.assembly.started" in testing_doc
    assert "memory.retrieved" in testing_doc


def test_runtime_substrate_adr_matches_custom_deterministic_mvp() -> None:
    adr = (DOCS_ROOT / "adr" / "ADR-006_langgraph_as_phase_1_runtime_substrate.md").read_text(
        encoding="utf-8",
    )

    assert "custom deterministic workflow" in adr.lower()
    assert "LangGraph is deferred" in adr
