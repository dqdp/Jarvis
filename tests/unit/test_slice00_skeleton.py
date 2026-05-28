from __future__ import annotations

import importlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def test_import_assistant_core() -> None:
    module = importlib.import_module("assistant_core")

    assert module.__name__ == "assistant_core"


def test_pytest_runs() -> None:
    assert True


def test_pytest_sees_required_markers(pytestconfig) -> None:
    configured_markers = {
        marker.split(":", 1)[0].strip()
        for marker in pytestconfig.getini("markers")
    }

    assert {
        "unit",
        "contract",
        "integration",
        "golden",
        "architecture",
        "e2e",
    }.issubset(configured_markers)


def test_config_test_environment_loads_minimally() -> None:
    config_path = PROJECT_ROOT / "config" / "test.yaml"

    assert config_path.is_file()
    assert "environment: test" in config_path.read_text(encoding="utf-8")
