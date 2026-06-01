from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.architecture


def test_pytest_declares_db_marker_and_opt_in_gate() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    conftest = (PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")

    assert "db: tests requiring the local PostgreSQL test database" in pyproject
    assert "--run-db" in conftest
    assert "JARVIS_RUN_DB_TESTS" in conftest
    assert "pytest.mark.skip" in conftest


def test_pytest_full_suite_uses_package_safe_import_mode() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "--import-mode=importlib" in pyproject


def test_make_db_targets_enable_db_tests_explicitly() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    for target in ["test-contract", "test-integration", "test-e2e"]:
        start = makefile.index(f"{target}:")
        end = makefile.find("\n\n", start)
        block = makefile[start:] if end == -1 else makefile[start:end]
        assert "JARVIS_RUN_DB_TESTS=1" in block, target
        assert "--run-db" in block, target


def test_database_dependent_tests_are_marked_db() -> None:
    missing_marker: list[str] = []
    for path in sorted((PROJECT_ROOT / "tests").glob("**/test_*.py")):
        if path.parent.name == "architecture":
            continue
        text = path.read_text(encoding="utf-8")
        if "create_database_engine(" not in text and "run_migrations(" not in text:
            continue
        if "pytest.mark.db" not in text:
            missing_marker.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert missing_marker == []
