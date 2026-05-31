from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_DATABASE_URL = "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55433/jarvis_local"
DEFAULT_PROFILE = "ollama"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_HEALTH_TIMEOUT_SECONDS = 45


class StartupError(RuntimeError):
    pass


@dataclass(frozen=True)
class JarvisRuntimeConfig:
    project_root: Path
    python: Path
    compose_file: Path
    run_dir: Path
    pid_file: Path
    log_file: Path
    env_file: Path
    base_url: str
    database_url: str
    profile: str
    host: str
    port: int
    health_timeout_seconds: int

    @classmethod
    def from_project_root(cls, project_root: Path) -> "JarvisRuntimeConfig":
        root = project_root.resolve()
        run_dir = root / ".run" / "jarvis"
        return cls(
            project_root=root,
            python=Path(os.environ.get("JARVIS_RUNTIME_PYTHON", root / ".venv" / "bin" / "python")),
            compose_file=root / "infra" / "compose" / "jarvis-postgres.yml",
            run_dir=run_dir,
            pid_file=run_dir / "daemon.pid",
            log_file=run_dir / "daemon.log",
            env_file=run_dir / "runtime.env",
            base_url=os.environ.get("JARVIS_RUNTIME_BASE_URL", DEFAULT_BASE_URL),
            database_url=os.environ.get("JARVIS_RUNTIME_DATABASE_URL", DEFAULT_DATABASE_URL),
            profile=os.environ.get("JARVIS_RUNTIME_PROFILE", DEFAULT_PROFILE),
            host=os.environ.get("JARVIS_RUNTIME_HOST", DEFAULT_HOST),
            port=int(os.environ.get("JARVIS_RUNTIME_PORT", str(DEFAULT_PORT))),
            health_timeout_seconds=int(
                os.environ.get(
                    "JARVIS_RUNTIME_HEALTH_TIMEOUT_SECONDS",
                    str(DEFAULT_HEALTH_TIMEOUT_SECONDS),
                ),
            ),
        )


def up_step_names(_config: JarvisRuntimeConfig) -> list[str]:
    return [
        "check_dependencies",
        "compose_up",
        "wait_database",
        "migrate",
        "start_daemon",
        "wait_health",
    ]


def reset_step_names(_config: JarvisRuntimeConfig) -> list[str]:
    return [
        "down",
        "compose_down_with_volumes",
        "remove_runtime_files",
    ]


def cli_command(config: JarvisRuntimeConfig) -> list[str]:
    return [
        str(config.python),
        "-m",
        "assistant_core.cli",
        "--base-url",
        config.base_url,
    ]


def status_payload(
    config: JarvisRuntimeConfig,
    *,
    pid: int | None,
    running: bool,
    health: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "database_url": config.database_url,
        "profile": config.profile,
        "pid": pid,
        "running": running,
        "health": health,
        "log": str(config.log_file),
    }


def check_dependencies(config: JarvisRuntimeConfig) -> None:
    if not config.python.is_file():
        raise StartupError(f"missing Python runtime: {config.python}; run make venv first")
    if importlib.util.find_spec("prompt_toolkit") is None:
        raise StartupError("missing prompt_toolkit; run make jarvis-bootstrap")


def bootstrap(config: JarvisRuntimeConfig) -> None:
    _run([str(config.python), "-m", "pip", "install", "-e", "."], cwd=config.project_root)
    check_dependencies(config)
    print("jarvis> bootstrap complete")


def up(config: JarvisRuntimeConfig) -> None:
    check_dependencies(config)
    _compose(config, "up", "-d", "postgres-jarvis")
    _wait_database(config)
    _migrate(config)
    _start_daemon(config)
    health = _wait_health(config)
    print("jarvis> ready")
    print(f"base_url> {config.base_url}")
    print(f"health> {health.get('status')}")
    print(f"log> {config.log_file}")
    print("cli> make jarvis-cli")


def cli(config: JarvisRuntimeConfig, passthrough: Sequence[str]) -> None:
    env = _runtime_env(config)
    command = [*cli_command(config), *passthrough]
    _run(command, cwd=config.project_root, env=env)


def status(config: JarvisRuntimeConfig) -> None:
    pid = _read_pid(config)
    running = _pid_running(pid) if pid is not None else False
    health = _health(config)
    print(json.dumps(status_payload(config, pid=pid, running=running, health=health), indent=2))


def logs(config: JarvisRuntimeConfig, *, lines: int) -> None:
    if not config.log_file.is_file():
        print(f"jarvis> log not found: {config.log_file}")
        return
    content = config.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        print(line)


def down(config: JarvisRuntimeConfig) -> None:
    pid = _read_pid(config)
    if pid is None:
        print("jarvis> daemon not running")
        return
    if _pid_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except PermissionError as exc:
            raise StartupError(f"cannot stop daemon pid {pid}: permission denied") from exc
        _wait_pid_exit(pid, timeout_seconds=10)
    _remove_pid_file(config)
    print("jarvis> daemon stopped")


def reset(config: JarvisRuntimeConfig, *, yes: bool) -> None:
    if not yes:
        raise StartupError("jarvis-reset is destructive; pass --yes")
    down(config)
    _compose(config, "down", "-v")
    _remove_runtime_files(config)
    print("jarvis> reset complete")


def _runtime_env(config: JarvisRuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["DATABASE_URL"] = config.database_url
    env["JARVIS_CONFIG_PROFILE"] = config.profile
    return env


def _compose(config: JarvisRuntimeConfig, *args: str) -> None:
    _run(
        ["docker", "compose", "-f", str(config.compose_file), *args],
        cwd=config.project_root,
    )


def _wait_database(config: JarvisRuntimeConfig) -> None:
    deadline = time.monotonic() + 30
    command = [
        "docker",
        "compose",
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
    while time.monotonic() < deadline:
        result = subprocess.run(
            command,
            cwd=config.project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise StartupError("Jarvis database did not become ready")


def _migrate(config: JarvisRuntimeConfig) -> None:
    _run(
        [str(config.python), "-m", "assistant_core.storage.migrations"],
        cwd=config.project_root,
        env=_runtime_env(config),
    )


def _start_daemon(config: JarvisRuntimeConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    existing_pid = _read_pid(config)
    if existing_pid is not None and _pid_running(existing_pid):
        return
    _remove_pid_file(config)
    if _port_open(config.host, config.port):
        raise StartupError(f"port {config.port} is already in use outside Jarvis runtime")

    log_handle = config.log_file.open("a", encoding="utf-8")
    command = [
        str(config.python),
        "-m",
        "uvicorn",
        "assistant_core.app_factory:create_asgi_app",
        "--factory",
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    process = subprocess.Popen(
        command,
        cwd=config.project_root,
        env=_runtime_env(config),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    config.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    config.env_file.write_text(
        "\n".join(
            [
                f"BASE_URL={config.base_url}",
                f"DATABASE_URL={config.database_url}",
                f"JARVIS_CONFIG_PROFILE={config.profile}",
                f"LOG_FILE={config.log_file}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )


def _wait_health(config: JarvisRuntimeConfig) -> dict[str, Any]:
    deadline = time.monotonic() + config.health_timeout_seconds
    last_health: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_health = _health(config)
        if last_health and last_health.get("status") == "ready":
            return last_health
        time.sleep(1)
    raise StartupError(f"Jarvis daemon did not become ready: {last_health}")


def _health(config: JarvisRuntimeConfig) -> dict[str, Any] | None:
    request_url = f"{config.base_url}/v1/health"
    try:
        with urlopen(request_url, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return {"status": f"http_{exc.code}"}
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def _read_pid(config: JarvisRuntimeConfig) -> int | None:
    try:
        return int(config.pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_pid_exit(pid: int, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except PermissionError as exc:
        raise StartupError(f"cannot force-stop daemon pid {pid}: permission denied") from exc


def _remove_pid_file(config: JarvisRuntimeConfig) -> None:
    try:
        config.pid_file.unlink()
    except FileNotFoundError:
        pass


def _remove_runtime_files(config: JarvisRuntimeConfig) -> None:
    for path in [config.pid_file, config.env_file, config.log_file]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise StartupError(f"command failed ({result.returncode}): {' '.join(command)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis_runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("up")
    subparsers.add_parser("cli").add_argument("args", nargs=argparse.REMAINDER)
    subparsers.add_parser("status")
    logs_parser = subparsers.add_parser("logs")
    logs_parser.add_argument("--lines", type=int, default=80)
    subparsers.add_parser("down")
    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = JarvisRuntimeConfig.from_project_root(Path(__file__).resolve().parents[2])
    try:
        if args.command == "bootstrap":
            bootstrap(config)
        elif args.command == "up":
            up(config)
        elif args.command == "cli":
            cli(config, args.args)
        elif args.command == "status":
            status(config)
        elif args.command == "logs":
            logs(config, lines=args.lines)
        elif args.command == "down":
            down(config)
        elif args.command == "reset":
            reset(config, yes=args.yes)
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except StartupError as exc:
        print(f"error> {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
