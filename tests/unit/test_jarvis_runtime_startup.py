from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dev" / "jarvis_runtime.py"
pytestmark = pytest.mark.unit


def _load_runtime_module():
    assert SCRIPT_PATH.is_file(), "scripts/dev/jarvis_runtime.py must exist"
    spec = importlib.util.spec_from_file_location("jarvis_runtime", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_jarvis_runtime_defaults_use_canonical_local_paths() -> None:
    runtime = _load_runtime_module()

    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    assert config.base_url == "http://127.0.0.1:8080"
    assert config.database_url == (
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55433/jarvis_local"
    )
    assert config.compose_file == PROJECT_ROOT / "infra" / "compose" / "jarvis-postgres.yml"
    assert config.run_dir == PROJECT_ROOT / ".run" / "jarvis"
    assert config.pid_file == PROJECT_ROOT / ".run" / "jarvis" / "daemon.pid"
    assert config.log_file == PROJECT_ROOT / ".run" / "jarvis" / "daemon.log"


def test_jarvis_up_plan_runs_migrations_before_daemon_and_health() -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    assert runtime.up_step_names(config) == [
        "check_dependencies",
        "compose_up",
        "wait_database",
        "migrate",
        "start_daemon",
        "wait_health",
    ]


def test_jarvis_cli_command_uses_canonical_base_url_without_plain_mode() -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    command = runtime.cli_command(config)

    assert command[:4] == [str(config.python), "-m", "assistant_core.cli", "--base-url"]
    assert config.base_url in command
    assert "--plain" not in command


def test_jarvis_status_payload_reports_runtime_contract() -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    payload = runtime.status_payload(
        config,
        pid=1234,
        running=True,
        health={"status": "ready"},
    )

    assert payload == {
        "base_url": "http://127.0.0.1:8080",
        "database_url": "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55433/jarvis_local",
        "profile": "ollama",
        "pid": 1234,
        "running": True,
        "health": {"status": "ready"},
        "log": str(PROJECT_ROOT / ".run" / "jarvis" / "daemon.log"),
    }


def test_jarvis_reset_plan_is_explicit_and_not_part_of_up() -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    assert "reset_database_volume" not in runtime.up_step_names(config)
    assert runtime.reset_step_names(config) == [
        "down",
        "compose_down_with_volumes",
        "remove_runtime_files",
    ]


def test_jarvis_dependency_check_fails_loudly_without_prompt_toolkit(monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(runtime.StartupError, match="prompt_toolkit"):
        runtime.check_dependencies(config)


def test_jarvis_down_reports_permission_denied_without_traceback(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)
    config = runtime.JarvisRuntimeConfig(
        project_root=config.project_root,
        python=config.python,
        compose_file=config.compose_file,
        run_dir=tmp_path,
        pid_file=tmp_path / "daemon.pid",
        log_file=tmp_path / "daemon.log",
        env_file=tmp_path / "runtime.env",
        base_url=config.base_url,
        database_url=config.database_url,
        profile=config.profile,
        host=config.host,
        port=config.port,
        health_timeout_seconds=config.health_timeout_seconds,
    )
    config.pid_file.write_text("1234\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: (_ for _ in ()).throw(PermissionError))

    with pytest.raises(runtime.StartupError, match="permission denied"):
        runtime.down(config)
