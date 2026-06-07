from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.tools.builtin import (
    calendar_diff_tool,
    calculator_tool,
    daemon_status_tool,
    datetime_diff_tool,
    datetime_now_tool,
    datetime_until_tool,
)
from assistant_core.tools.registry import ToolRegistry
from assistant_core.tools.shell_read import project_shell_read_tool_from_config
from assistant_core.tools.system_diagnostics import system_diagnostics_tools_from_config


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


def _config_with_run_dir(runtime, tmp_path):
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)
    return runtime.JarvisRuntimeConfig(
        project_root=config.project_root,
        python=config.python,
        compose_file=config.compose_file,
        run_dir=tmp_path,
        pid_file=tmp_path / "daemon.pid",
        log_file=tmp_path / "daemon.log",
        env_file=tmp_path / "runtime.env",
        lock_file=tmp_path / "daemon.lock",
        base_url=config.base_url,
        database_url=config.database_url,
        profile=config.profile,
        host=config.host,
        port=config.port,
        health_timeout_seconds=config.health_timeout_seconds,
    )


def test_runtime_app_validates_request_plan_tools_match_gateway_registry() -> None:
    from assistant_core.app_factory import _validate_request_plan_tool_surface

    settings = ConfigLoader(PROJECT_ROOT / "config").load("test")

    with pytest.raises(RuntimeError, match="request-plan tool is not registered"):
        _validate_request_plan_tool_surface(settings, ToolRegistry([]))


def test_runtime_app_validates_request_plan_tool_policy_shape_matches_gateway_registry() -> None:
    from assistant_core.app_factory import _validate_request_plan_tool_surface

    settings = ConfigLoader(PROJECT_ROOT / "config").load("test")
    adapter = datetime_now_tool()
    adapter.spec = replace(
        adapter.spec,
        capability=Capability.TOOL_SHELL_READ,
        risk_classes=frozenset({RiskClass.READ_ONLY}),
        sensitivity_ceiling=Sensitivity.INFRA,
    )
    registry = ToolRegistry(
        [
            adapter,
            calendar_diff_tool(),
            calculator_tool(),
            daemon_status_tool(),
            datetime_diff_tool(),
            datetime_until_tool(),
            project_shell_read_tool_from_config(settings.capabilities),
            *system_diagnostics_tools_from_config(settings.capabilities),
        ],
    )

    with pytest.raises(RuntimeError, match="request-plan tool metadata differs"):
        _validate_request_plan_tool_surface(settings, registry)


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
    assert config.lock_file == PROJECT_ROOT / ".run" / "jarvis" / "daemon.lock"


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


def test_jarvis_cli_passthrough_preserves_option_like_cli_args() -> None:
    runtime = _load_runtime_module()

    args = runtime._parse_args(["cli", "--color", "always", "chat", "--developer"])

    assert args.command == "cli"
    assert args.args == ["--color", "always", "chat", "--developer"]


def test_jarvis_runtime_cli_failure_redacts_passthrough_prompt_args(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)

    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2),
    )

    with pytest.raises(runtime.StartupError) as exc:
        runtime.cli(config, ["chat", "secret prompt text"])

    message = str(exc.value)
    assert "secret prompt text" not in message
    assert "[cli args redacted]" in message


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
        "database_url": "postgresql+asyncpg://jarvis:***@127.0.0.1:55433/jarvis_local",
        "profile": "ollama",
        "port": 8080,
        "pid": 1234,
        "running": True,
        "owned": True,
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


def test_jarvis_base_url_follows_configured_host_and_port(monkeypatch) -> None:
    runtime = _load_runtime_module()

    monkeypatch.delenv("JARVIS_RUNTIME_BASE_URL", raising=False)
    monkeypatch.setenv("JARVIS_RUNTIME_HOST", "127.0.0.1")
    monkeypatch.setenv("JARVIS_RUNTIME_PORT", "18080")

    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)

    assert config.base_url == "http://127.0.0.1:18080"


def test_jarvis_base_url_override_must_match_host_and_port(monkeypatch) -> None:
    runtime = _load_runtime_module()

    monkeypatch.setenv("JARVIS_RUNTIME_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("JARVIS_RUNTIME_HOST", "127.0.0.1")
    monkeypatch.setenv("JARVIS_RUNTIME_PORT", "18080")

    with pytest.raises(runtime.StartupError, match="JARVIS_RUNTIME_BASE_URL"):
        runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)


def test_jarvis_down_refuses_unowned_pid_without_signal(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config.pid_file.write_text("1234\n", encoding="utf-8")
    killed: list[tuple[int, int]] = []

    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_process_command", lambda pid: "python -c sleep", raising=False)

    def fake_kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        raise AssertionError("must not signal an unowned pid")

    monkeypatch.setattr(runtime.os, "kill", fake_kill)

    with pytest.raises(runtime.StartupError, match="unowned|owned"):
        runtime.down(config)

    assert killed == []


def test_jarvis_down_refuses_reused_daemon_like_pid_without_runtime_lock(
    tmp_path,
    monkeypatch,
) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config.pid_file.write_text("1234\n", encoding="utf-8")
    config.env_file.write_text(
        "\n".join(
            [
                "PID=1234",
                f"BASE_URL={config.base_url}",
                f"HOST={config.host}",
                f"PORT={config.port}",
                f"JARVIS_CONFIG_PROFILE={config.profile}",
                f"PROJECT_ROOT={config.project_root}",
                f"LOCK_FILE={config.lock_file}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    daemon_like_command = " ".join(
        [
            str(config.python),
            "-m",
            "uvicorn",
            runtime.DAEMON_APP,
            "--factory",
            "--host",
            config.host,
            "--port",
            str(config.port),
        ],
    )
    killed: list[tuple[int, int]] = []

    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_process_command", lambda pid: daemon_like_command)

    def fake_kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        raise AssertionError("must not signal a PID without the Jarvis runtime lock")

    monkeypatch.setattr(runtime.os, "kill", fake_kill)

    with pytest.raises(runtime.StartupError, match="unowned|owned"):
        runtime.down(config)

    assert killed == []


def test_jarvis_owned_pid_allows_unavailable_ps_when_runtime_lock_is_held(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config.pid_file.write_text("1234\n", encoding="utf-8")
    config.env_file.write_text(
        "\n".join(
            [
                "PID=1234",
                f"BASE_URL={config.base_url}",
                f"HOST={config.host}",
                f"PORT={config.port}",
                f"JARVIS_CONFIG_PROFILE={config.profile}",
                f"PROJECT_ROOT={config.project_root}",
                f"LOCK_FILE={config.lock_file}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_process_command", lambda pid: None)
    monkeypatch.setattr(runtime, "_runtime_lock_is_held", lambda config: True)

    assert runtime._owned_daemon_pid(config, 1234) is True


def test_jarvis_owned_pid_survives_current_env_drift_when_runtime_lock_is_held(
    tmp_path,
    monkeypatch,
) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config.pid_file.write_text("1234\n", encoding="utf-8")
    config.env_file.write_text(
        "\n".join(
            [
                "PID=1234",
                "BASE_URL=http://127.0.0.1:19090",
                "HOST=127.0.0.1",
                "PORT=19090",
                "JARVIS_CONFIG_PROFILE=custom-local",
                f"PROJECT_ROOT={config.project_root}",
                f"LOCK_FILE={config.lock_file}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    daemon_command = " ".join(
        [
            str(config.python),
            "-m",
            "uvicorn",
            runtime.DAEMON_APP,
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            "19090",
        ],
    )

    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_process_command", lambda pid: daemon_command)
    monkeypatch.setattr(runtime, "_runtime_lock_is_held", lambda config: True)

    assert runtime._owned_daemon_pid(config, 1234) is True


def test_jarvis_start_does_not_skip_unowned_live_pid(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config.pid_file.write_text("1234\n", encoding="utf-8")
    started: list[list[str]] = []

    class FakeProcess:
        pid = 5678

    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_process_command", lambda pid: "python -c sleep", raising=False)
    monkeypatch.setattr(runtime, "_port_open", lambda host, port: False)

    def fake_popen(command, **kwargs):
        started.append(list(command))
        return FakeProcess()

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)

    runtime._start_daemon(config)

    assert started
    assert config.pid_file.read_text(encoding="utf-8") == "5678\n"


def test_jarvis_start_writes_lock_metadata_and_passes_lock_fd(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    popen_kwargs: dict[str, object] = {}

    monkeypatch.setattr(runtime, "_port_open", lambda host, port: False)

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return SimpleNamespace(pid=5678)

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)

    runtime._start_daemon(config)

    assert "LOCK_FILE=" + str(config.lock_file) in config.env_file.read_text(encoding="utf-8")
    assert popen_kwargs["pass_fds"]


def test_jarvis_runtime_metadata_does_not_persist_raw_database_password(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config = runtime.JarvisRuntimeConfig(
        project_root=config.project_root,
        python=config.python,
        compose_file=config.compose_file,
        run_dir=config.run_dir,
        pid_file=config.pid_file,
        log_file=config.log_file,
        env_file=config.env_file,
        lock_file=config.lock_file,
        base_url=config.base_url,
        database_url=(
            "postgresql+asyncpg://jarvis:sensitive@127.0.0.1:55433/jarvis_local"
            "?sslmode=require&password=querysecret"
        ),
        profile=config.profile,
        host=config.host,
        port=config.port,
        health_timeout_seconds=config.health_timeout_seconds,
    )

    monkeypatch.setattr(runtime, "_port_open", lambda host, port: False)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda command, **kwargs: SimpleNamespace(pid=5678))
    monkeypatch.setattr(
        runtime,
        "_process_command",
        lambda pid: (
            f"{config.python} -m uvicorn {runtime.DAEMON_APP} --factory "
            f"--host {config.host} --port {config.port}"
        ),
    )

    runtime._start_daemon(config)

    metadata = config.env_file.read_text(encoding="utf-8")
    assert "sensitive" not in metadata
    assert "querysecret" not in metadata
    assert "DATABASE_URL=postgresql" not in metadata
    assert "DATABASE_URL_SHA256" not in metadata
    assert "DATABASE_URL_REDACTED=postgresql+asyncpg://jarvis:***@" in metadata


def test_jarvis_wait_pid_exit_skips_force_kill_after_ownership_loss(monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)
    killed: list[tuple[int, int]] = []

    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_owned_daemon_pid", lambda config, pid: False, raising=False)
    monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    runtime._wait_pid_exit(config, 1234, timeout_seconds=0)

    assert killed == []


def test_jarvis_wait_database_uses_canonical_compose_project(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    runtime._wait_database(config)

    assert commands == [
        [
            "docker",
            "compose",
            "-p",
            "jarvis-runtime",
            "-f",
            str(config.compose_file),
            "exec",
            "-T",
            "postgres-jarvis",
            "pg_isready",
            "-U",
            "jarvis",
            "-d",
            "jarvis_local",
        ]
    ]


def test_jarvis_process_command_returns_none_when_ps_is_not_permitted(monkeypatch) -> None:
    runtime = _load_runtime_module()

    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("ps denied")),
    )

    assert runtime._process_command(1234) is None


def test_jarvis_wait_database_pins_compose_project_name(monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = runtime.JarvisRuntimeConfig.from_project_root(PROJECT_ROOT)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    runtime._wait_database(config)

    assert commands
    assert commands[0][:4] == ["docker", "compose", "-p", runtime.JARVIS_COMPOSE_PROJECT]


def test_jarvis_main_reports_config_errors_without_traceback(monkeypatch, capsys) -> None:
    runtime = _load_runtime_module()

    monkeypatch.setenv("JARVIS_RUNTIME_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("JARVIS_RUNTIME_PORT", "18080")

    assert runtime.main(["status"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error> JARVIS_RUNTIME_BASE_URL" in captured.err


def test_jarvis_down_reports_permission_denied_without_traceback(tmp_path, monkeypatch) -> None:
    runtime = _load_runtime_module()
    config = _config_with_run_dir(runtime, tmp_path)
    config.pid_file.write_text("1234\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "_pid_running", lambda pid: True)
    monkeypatch.setattr(runtime, "_owned_daemon_pid", lambda config, pid: True, raising=False)
    monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: (_ for _ in ()).throw(PermissionError))

    with pytest.raises(runtime.StartupError, match="permission denied"):
        runtime.down(config)
