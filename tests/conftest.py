from __future__ import annotations

import os

import pytest


_TRUE_VALUES = {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-db",
        action="store_true",
        default=False,
        help="run tests marked db that require the local PostgreSQL test database",
    )
    parser.addoption(
        "--run-evaluation",
        action="store_true",
        default=False,
        help="run tests marked evaluation that require an opt-in local model",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "db: tests requiring the local PostgreSQL test database",
    )
    config.addinivalue_line(
        "markers",
        "evaluation: opt-in local model evaluation tests",
    )


def _db_tests_enabled(config: pytest.Config) -> bool:
    enabled_by_env = os.environ.get("JARVIS_RUN_DB_TESTS", "").lower() in _TRUE_VALUES
    return bool(config.getoption("--run-db") or enabled_by_env)


def _evaluation_tests_enabled(config: pytest.Config) -> bool:
    enabled_by_env = (
        os.environ.get("JARVIS_RUN_EVALUATION_TESTS", "").lower() in _TRUE_VALUES
    )
    return bool(config.getoption("--run-evaluation") or enabled_by_env)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_db = None
    if not _db_tests_enabled(config):
        skip_db = pytest.mark.skip(
            reason=(
                "requires local PostgreSQL test database; pass --run-db or set "
                "JARVIS_RUN_DB_TESTS=1"
            ),
        )
    skip_evaluation = None
    if not _evaluation_tests_enabled(config):
        skip_evaluation = pytest.mark.skip(
            reason=(
                "requires opt-in local model evaluation; pass --run-evaluation "
                "or set JARVIS_RUN_EVALUATION_TESTS=1"
            ),
        )
    for item in items:
        if skip_db is not None and "db" in item.keywords:
            item.add_marker(skip_db)
        if skip_evaluation is not None and "evaluation" in item.keywords:
            item.add_marker(skip_evaluation)
