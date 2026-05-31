from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.system_diagnostics import (
    SensorReading,
    SensorSnapshot,
    SystemDiagnosticsDecision,
    SystemDiagnosticsFamily,
)
from assistant_core.domain.tools import ToolInvocationResult, ToolSpec
from assistant_core.tools.registry import ToolClassificationResult, ToolExecutionDenied
from assistant_core.tools.shell_read import (
    ShellExecutionResult,
    ShellExecutionTimeout,
    ShellExecutorPort,
    SubprocessShellExecutor,
)


_MINIMAL_ENV = {
    "PATH": "/Applications/Codex.app/Contents/Resources:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "LANG": "C",
    "LC_ALL": "C",
}
_SHELL_SYNTAX_MARKERS = ("|", ";", "&&", "||", ">", "<", "`", "$(", "\n", "\r")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SYSTEM_DIAGNOSTICS_TOOL_NAMES = {
    SystemDiagnosticsFamily.PROCESS: "tool.system.read.process",
    SystemDiagnosticsFamily.RESOURCES: "tool.system.read.resources",
    SystemDiagnosticsFamily.HARDWARE: "tool.system.read.hardware",
    SystemDiagnosticsFamily.NETWORK: "tool.system.read.network",
    SystemDiagnosticsFamily.SENSORS: "tool.system.read.sensors",
}
_SYSTEM_DIAGNOSTICS_CAPABILITIES = {
    SystemDiagnosticsFamily.PROCESS: Capability.TOOL_SYSTEM_READ_PROCESS,
    SystemDiagnosticsFamily.RESOURCES: Capability.TOOL_SYSTEM_READ_RESOURCES,
    SystemDiagnosticsFamily.HARDWARE: Capability.TOOL_SYSTEM_READ_HARDWARE,
    SystemDiagnosticsFamily.NETWORK: Capability.TOOL_SYSTEM_READ_NETWORK,
    SystemDiagnosticsFamily.SENSORS: Capability.TOOL_SYSTEM_READ_SENSORS,
}
_KNOWN_COMMANDS = {
    "df",
    "du",
    "free",
    "htop",
    "ifconfig",
    "ip",
    "kill",
    "launchctl",
    "less",
    "lscpu",
    "lshw",
    "lsof",
    "netstat",
    "nvidia-smi",
    "pgrep",
    "pmset",
    "powermetrics",
    "ps",
    "renice",
    "sensors",
    "scutil",
    "ss",
    "sudo",
    "sw_vers",
    "sysctl",
    "systemctl",
    "thermal-sysfs",
    "top",
    "uname",
    "uptime",
    "upower",
    "vim",
    "vm_stat",
    "watch",
}
_INTERACTIVE_COMMANDS = {"htop", "less", "vim", "watch"}
_MUTATING_COMMANDS = {"sudo", "kill", "killall", "renice", "launchctl", "systemctl"}
_NETWORK_CLIENTS = {"curl", "wget", "nc", "ssh", "scp", "telnet", "ftp"}
_SENSOR_MUTATION_COMMANDS = {"tee", "echo", "fanctl"}
_DARWIN_SYSCTL_KEYS = {
    "hw.memsize",
    "hw.ncpu",
    "hw.logicalcpu",
    "hw.physicalcpu",
    "machdep.cpu.brand_string",
}
_CREDENTIAL_URL = re.compile(r"\b(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@[^\s/]+")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)--(?:api[_-]?key|auth|authorization|token|password|secret)(?:=|\s+)[^\s]+"),
    re.compile(r"(?i)\b[A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?KEY|AUTH|TOKEN|PASSWORD|SECRET)[A-Z0-9_]*=[^\s]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|auth|authorization|token|password|secret)=[^\s]+"),
    re.compile(r"(?i)\bauthorization:\s*bearer\s+[^\s]+"),
    re.compile(r"\bsk-[A-Za-z0-9._-]+"),
    re.compile(r"\bghp_[A-Za-z0-9_]+"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]+"),
)


class TemperatureSensorProviderPort(Protocol):
    async def snapshot_temperatures(self) -> SensorSnapshot: ...


@dataclass(frozen=True)
class _LimitedText:
    text: str
    raw_bytes: int
    truncated: bool


class SystemDiagnosticsClassifier:
    def __init__(
        self,
        *,
        allowed_roots: Sequence[str | Path],
        platform: str | None = None,
    ) -> None:
        self._allowed_roots = [_resolve_root(root) for root in allowed_roots]
        self._platform = _normalize_platform(platform or sys.platform)

    def classify(self, argv: Sequence[str], *, cwd: str | Path) -> SystemDiagnosticsDecision:
        normalized = tuple(str(arg) for arg in argv)
        cwd_decision = self._resolve_cwd(cwd, normalized)
        if not cwd_decision.allowed:
            return cwd_decision
        resolved_cwd = Path(cwd_decision.cwd or "")

        syntax_decision = _syntax_decision(normalized, resolved_cwd)
        if syntax_decision is not None:
            return syntax_decision

        if not normalized:
            return _deny("empty_command", "command argv must not be empty", normalized, resolved_cwd)

        command = normalized[0]
        if _command_is_path(command):
            return _deny(
                "command_path_denied",
                "command must be an allowlisted bare executable name",
                normalized,
                resolved_cwd,
            )
        if command in _INTERACTIVE_COMMANDS:
            return _deny(
                "interactive_command_denied",
                "interactive diagnostics commands are denied",
                normalized,
                resolved_cwd,
            )
        if command in _MUTATING_COMMANDS:
            return _deny(
                "mutating_command_denied",
                "mutating or privileged system commands are denied",
                normalized,
                resolved_cwd,
            )
        if command in _NETWORK_CLIENTS:
            return _deny(
                "network_client_denied",
                "network clients are denied in diagnostics tools",
                normalized,
                resolved_cwd,
            )
        if (
            command in _SENSOR_MUTATION_COMMANDS
            or _is_pmset_mutation(normalized)
            or any("/sys/class/thermal" in arg for arg in normalized)
        ):
            return _deny(
                "sensor_mutation_denied",
                "sensor write or fan/power mutation commands are denied",
                normalized,
                resolved_cwd,
            )

        classifier = {
            "ps": self._classify_ps,
            "pgrep": self._classify_pgrep,
            "uptime": self._classify_uptime,
            "df": self._classify_df,
            "du": self._classify_du,
            "top": self._classify_top,
            "vm_stat": self._classify_vm_stat,
            "free": self._classify_free,
            "sysctl": self._classify_sysctl,
            "lscpu": self._classify_lscpu,
            "lshw": self._classify_lshw,
            "uname": self._classify_uname,
            "upower": self._classify_upower,
            "netstat": self._classify_netstat,
            "ifconfig": self._classify_ifconfig,
            "lsof": self._classify_lsof,
            "ss": self._classify_ss,
            "ip": self._classify_ip,
            "sw_vers": self._classify_sw_vers,
            "pmset": self._classify_pmset,
            "scutil": self._classify_scutil,
            "powermetrics": self._classify_powermetrics,
            "sensors": self._classify_sensors,
            "thermal-sysfs": self._classify_thermal_sysfs,
            "nvidia-smi": self._classify_nvidia_smi,
        }.get(command)
        if classifier is None:
            return _deny("unsupported_command", "diagnostics command is not allowlisted", normalized, resolved_cwd)
        return classifier(normalized, resolved_cwd)

    def _resolve_cwd(
        self,
        cwd: str | Path,
        argv: tuple[str, ...],
    ) -> SystemDiagnosticsDecision:
        try:
            resolved = Path(cwd).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return _deny("invalid_working_directory", "working directory is invalid", argv, None)
        if not resolved.is_dir():
            return _deny("invalid_working_directory", "working directory must be a directory", argv, resolved)
        if _secret_like_path(str(resolved)):
            return _deny("secret_path_denied", "working directory is secret-like", argv, resolved)
        if not _is_inside_any_root(resolved, self._allowed_roots):
            return _deny(
                "path_outside_workspace",
                "working directory is outside allowed roots",
                argv,
                resolved,
            )
        return SystemDiagnosticsDecision(
            allowed=True,
            code="allowed",
            reason="working directory is allowed",
            argv=argv,
            cwd=str(resolved),
            metadata={"platform": self._platform},
        )

    def _classify_ps(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if argv in {("ps", "aux"), ("ps", "-ef")}:
            return _allow(SystemDiagnosticsFamily.PROCESS, argv, cwd, self._platform)
        if len(argv) == 3 and argv[1] == "-Ao" and argv[2] in {
            "pid,comm,command",
            "pid,ppid,comm,command",
            "pid,command",
        }:
            return _allow(SystemDiagnosticsFamily.PROCESS, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "ps supports only bounded snapshot formats", argv, cwd)

    def _classify_pgrep(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if len(argv) == 3 and argv[1] in {"-fl", "-lf", "-l"} and _safe_pattern(argv[2]):
            return _allow(SystemDiagnosticsFamily.PROCESS, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "pgrep supports only read-only bounded flags", argv, cwd)

    def _classify_uptime(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "uptime does not accept arguments", argv, cwd)

    def _classify_df(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if set(argv[1:]).issubset({"-h", "-k", "-P"}) and len(argv) <= 3:
            return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "df supports only selected snapshot flags", argv, cwd)

    def _classify_du(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if len(argv) != 3 or argv[1] not in {"-sh", "-sk"}:
            return _deny("unsupported_arguments", "du supports only du -sh PATH or du -sk PATH", argv, cwd)
        if _secret_like_path(argv[2]):
            return _deny("secret_path_denied", "du path is secret-like", argv, cwd)
        try:
            resolved = _resolve_command_path(argv[2], cwd)
        except (OSError, RuntimeError, ValueError):
            return _deny("invalid_path", "du path is invalid", argv, cwd)
        if not _is_inside_any_root(resolved, self._allowed_roots):
            return _deny("path_outside_workspace", "du path escapes allowed roots", argv, cwd)
        if _secret_like_path(str(resolved)):
            return _deny("secret_path_denied", "du path is secret-like", argv, cwd)
        return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)

    def _classify_top(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "darwin" and argv in {
            ("top", "-l", "1"),
            ("top", "-l", "1", "-n", "0"),
        }:
            return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)
        if self._platform == "linux" and argv == ("top", "-b", "-n", "1"):
            return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "top supports only one-shot snapshot mode", argv, cwd)

    def _classify_vm_stat(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "darwin" and len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "vm_stat is allowed only on macOS", argv, cwd)

    def _classify_free(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and (len(argv) == 1 or argv[1:] in {("-m",), ("-h",)}):
            return _allow(SystemDiagnosticsFamily.RESOURCES, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "free is allowed only on Linux", argv, cwd)

    def _classify_sysctl(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if (
            self._platform == "darwin"
            and len(argv) == 3
            and argv[1] == "-n"
            and argv[2] in _DARWIN_SYSCTL_KEYS
        ):
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "sysctl key is not allowlisted", argv, cwd)

    def _classify_sw_vers(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "darwin" and len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "sw_vers is allowed only on macOS", argv, cwd)

    def _classify_lscpu(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "lscpu is allowed only on Linux", argv, cwd)

    def _classify_lshw(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and (len(argv) == 1 or argv == ("lshw", "-short")):
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "lshw is allowed only on Linux", argv, cwd)

    def _classify_uname(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and argv == ("uname", "-a"):
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "uname -a is allowed only on Linux", argv, cwd)

    def _classify_upower(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and argv == (
            "upower",
            "-i",
            "/org/freedesktop/UPower/devices/DisplayDevice",
        ):
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny(
            "unsupported_arguments",
            "upower is limited to the DisplayDevice battery snapshot",
            argv,
            cwd,
        )

    def _classify_pmset(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "darwin" and argv == ("pmset", "-g", "batt"):
            return _allow(SystemDiagnosticsFamily.HARDWARE, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "pmset battery snapshots are allowed only on macOS", argv, cwd)

    def _classify_netstat(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if argv in {("netstat", "-an"), ("netstat", "-ant"), ("netstat", "-anv")}:
            return _allow(SystemDiagnosticsFamily.NETWORK, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "netstat flags are not allowlisted", argv, cwd)

    def _classify_scutil(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "darwin" and argv == ("scutil", "--nc", "list"):
            return _allow(SystemDiagnosticsFamily.NETWORK, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "scutil VPN snapshots are allowed only on macOS", argv, cwd)

    def _classify_ifconfig(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "darwin" and len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.NETWORK, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "ifconfig is allowed only on macOS", argv, cwd)

    def _classify_lsof(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if argv == ("lsof", "-nP", "-iTCP", "-sTCP:LISTEN"):
            return _allow(SystemDiagnosticsFamily.NETWORK, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "lsof flags are not allowlisted", argv, cwd)

    def _classify_ss(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and argv in {("ss", "-tulpen"), ("ss", "-tulpn")}:
            return _allow(SystemDiagnosticsFamily.NETWORK, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "ss is allowed only on Linux", argv, cwd)

    def _classify_ip(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and argv == ("ip", "addr"):
            return _allow(SystemDiagnosticsFamily.NETWORK, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "ip addr is allowed only on Linux", argv, cwd)

    def _classify_powermetrics(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if "-i" in argv or ("-n" in argv and _option_value(argv, "-n") != "1"):
            return _deny("sensor_polling_denied", "sensor diagnostics must be one-shot", argv, cwd)
        if self._platform != "darwin":
            return _deny(
                "unsupported_platform_command",
                "powermetrics sensor snapshot is allowed only on macOS",
                argv,
                cwd,
            )
        if argv in {
            ("powermetrics", "--samplers", "smc", "-n", "1"),
            ("powermetrics", "--samplers", "thermal", "-n", "1"),
        }:
            return _allow(SystemDiagnosticsFamily.SENSORS, argv, cwd, self._platform)
        return _deny(
            "unsupported_arguments",
            "powermetrics supports only one-shot smc or thermal sampler snapshots",
            argv,
            cwd,
        )

    def _classify_sensors(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.SENSORS, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "sensors is allowed only on Linux", argv, cwd)

    def _classify_thermal_sysfs(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and len(argv) == 1:
            return _allow(SystemDiagnosticsFamily.SENSORS, argv, cwd, self._platform)
        return _deny("unsupported_platform_command", "thermal-sysfs is allowed only on Linux", argv, cwd)

    def _classify_nvidia_smi(self, argv: tuple[str, ...], cwd: Path) -> SystemDiagnosticsDecision:
        if self._platform == "linux" and argv == (
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ):
            return _allow(SystemDiagnosticsFamily.SENSORS, argv, cwd, self._platform)
        return _deny("unsupported_arguments", "nvidia-smi is limited to GPU temperature query mode", argv, cwd)


class ReadOnlyThermalSysfsSensorProvider:
    def __init__(self, root: Path = Path("/sys/class/thermal")) -> None:
        self._root = root

    async def snapshot_temperatures(self) -> SensorSnapshot:
        if not self._root.exists():
            return SensorSnapshot.unavailable(source="thermal-sysfs", reason="thermal sysfs not available")
        readings: list[SensorReading] = []
        try:
            for zone in sorted(self._root.glob("thermal_zone*/temp"))[:32]:
                raw = zone.read_text(encoding="utf-8").strip()
                if not raw:
                    continue
                value = float(raw)
                if value > 1000:
                    value /= 1000.0
                label_path = zone.with_name("type")
                label = zone.parent.name
                if label_path.exists():
                    label = label_path.read_text(encoding="utf-8").strip() or label
                readings.append(
                    SensorReading(
                        label=label,
                        value=value,
                        unit="C",
                        source=str(zone),
                    ),
                )
        except (OSError, ValueError) as exc:
            return SensorSnapshot.unavailable(source="thermal-sysfs", reason=type(exc).__name__)
        if not readings:
            return SensorSnapshot.unavailable(source="thermal-sysfs", reason="no thermal readings")
        return SensorSnapshot(source="thermal-sysfs", readings=readings).normalized_celsius()


class SystemDiagnosticsTool:
    content_type = "application/json"

    def __init__(
        self,
        *,
        family: SystemDiagnosticsFamily | str,
        allowed_roots: Sequence[str | Path],
        executor: ShellExecutorPort | None = None,
        sensor_provider: TemperatureSensorProviderPort | None = None,
        max_stdout_bytes: int = 20_000,
        max_stderr_bytes: int = 20_000,
        max_lines: int = 200,
        timeout_seconds: float = 10.0,
        platform: str | None = None,
    ) -> None:
        self._family = family if isinstance(family, SystemDiagnosticsFamily) else SystemDiagnosticsFamily(family)
        self._allowed_roots = [_resolve_root(root) for root in allowed_roots]
        self._classifier = SystemDiagnosticsClassifier(
            allowed_roots=self._allowed_roots,
            platform=platform,
        )
        self._executor = executor or SubprocessShellExecutor(
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            max_lines=max_lines,
        )
        self._sensor_provider = sensor_provider or ReadOnlyThermalSysfsSensorProvider()
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._max_lines = max_lines
        self._timeout_seconds = timeout_seconds
        self.spec = ToolSpec(
            name=_SYSTEM_DIAGNOSTICS_TOOL_NAMES[self._family],
            display_name=f"System {self._family.value.title()} Diagnostics",
            description="Runs curated read-only local system diagnostics.",
            capability=_SYSTEM_DIAGNOSTICS_CAPABILITIES[self._family],
            risk_classes=frozenset({RiskClass.READ_ONLY}),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 256},
                        "minItems": 1,
                        "maxItems": 16,
                    },
                    "cwd": {"type": "string", "maxLength": 512},
                },
                "required": ["argv", "cwd"],
                "additionalProperties": False,
            },
            adapter_name=f"system_diagnostics.{self._family.value}",
            output_schema={"type": "object"},
            default_timeout_seconds=timeout_seconds,
            max_output_bytes=max_stdout_bytes + max_stderr_bytes + 2048,
            sensitivity_ceiling=Sensitivity.INFRA,
            metadata={"family": self._family.value},
        )

    def classify(self, arguments: dict[str, Any]) -> ToolClassificationResult:
        decision = self._classify_arguments(arguments)
        return ToolClassificationResult(
            allowed=decision.allowed,
            code=decision.code,
            reason=decision.reason,
            metadata={
                "family": decision.family.value if decision.family is not None else None,
                "cwd": _redacted_cwd(decision.cwd, allowed=decision.allowed),
                "argv": _redacted_argv(decision.argv),
                **decision.metadata,
            },
        )

    async def invoke(self, arguments: dict[str, Any]) -> ToolInvocationResult:
        decision = self._classify_arguments(arguments)
        if not decision.allowed:
            raise ToolExecutionDenied(
                decision.code,
                decision.reason,
                metadata=self.classify(arguments).metadata,
            )

        if decision.family == SystemDiagnosticsFamily.SENSORS and decision.argv == ("thermal-sysfs",):
            snapshot = (await self._sensor_provider.snapshot_temperatures()).normalized_celsius()
            content = snapshot.to_dict()
            encoded = json.dumps(content, sort_keys=True)
            return ToolInvocationResult(
                content=encoded,
                content_type="application/json",
                truncated=False,
                output_bytes=len(encoded.encode("utf-8")),
                metadata={
                    "family": decision.family.value,
                    "cwd": decision.cwd,
                    "source": snapshot.source,
                    "unavailable": not snapshot.available,
                },
            )

        try:
            result = await self._executor.execute(
                argv=list(decision.argv),
                cwd=Path(decision.cwd or _arguments_cwd(arguments)),
                env=dict(_MINIMAL_ENV),
                timeout_seconds=self._timeout_seconds,
            )
        except FileNotFoundError:
            return _unavailable_result(decision, source=decision.argv[0], reason="backend_not_found")
        except ShellExecutionTimeout:
            raise

        if decision.family == SystemDiagnosticsFamily.SENSORS:
            sensor_result = _sensor_invocation_result(decision, result)
            if sensor_result is not None:
                return sensor_result

        stdout = _redact_diagnostics_output(result.stdout)
        stderr = _redact_diagnostics_output(result.stderr)
        bounded_stdout = _bounded_text(stdout, max_bytes=self._max_stdout_bytes, max_lines=self._max_lines)
        bounded_stderr = _bounded_text(stderr, max_bytes=self._max_stderr_bytes, max_lines=self._max_lines)
        stdout_truncated = bool(bounded_stdout.truncated or result.stdout_truncated)
        stderr_truncated = bool(bounded_stderr.truncated or result.stderr_truncated)
        raw_stdout_bytes = (
            result.raw_stdout_bytes
            if result.raw_stdout_bytes is not None
            else bounded_stdout.raw_bytes
        )
        raw_stderr_bytes = (
            result.raw_stderr_bytes
            if result.raw_stderr_bytes is not None
            else bounded_stderr.raw_bytes
        )
        raw_output_bytes = raw_stdout_bytes + raw_stderr_bytes
        content = {
            "exit_code": result.exit_code,
            "stdout": bounded_stdout.text,
            "stderr": bounded_stderr.text,
            "truncated": {
                "stdout": stdout_truncated,
                "stderr": stderr_truncated,
            },
        }
        return ToolInvocationResult(
            content=json.dumps(content, sort_keys=True),
            content_type="application/json",
            truncated=stdout_truncated or stderr_truncated,
            output_bytes=raw_output_bytes,
            metadata={
                "family": decision.family.value if decision.family is not None else None,
                "cwd": decision.cwd,
                "exit_code": result.exit_code,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "raw_stdout_bytes": raw_stdout_bytes,
                "raw_stderr_bytes": raw_stderr_bytes,
            },
        )

    def _classify_arguments(self, arguments: dict[str, Any]) -> SystemDiagnosticsDecision:
        argv = _arguments_argv(arguments)
        cwd = _arguments_cwd(arguments)
        decision = self._classifier.classify(argv, cwd=cwd)
        if decision.allowed and decision.family != self._family:
            return _deny(
                "diagnostics_family_mismatch",
                "diagnostics command family does not match tool capability",
                decision.argv,
                Path(decision.cwd or cwd),
            )
        return decision


def system_diagnostics_tools_from_config(capabilities: dict[str, Any]) -> list[SystemDiagnosticsTool]:
    config = capabilities["tool.system.read"]
    max_output_bytes = int(config["max_output_bytes"])
    enabled_families = [
        SystemDiagnosticsFamily(family)
        for family in config.get("enabled_families", [family.value for family in SystemDiagnosticsFamily])
    ]
    return [
        SystemDiagnosticsTool(
            family=family,
            allowed_roots=[Path(root) for root in config["allowed_roots"]],
            max_stdout_bytes=max_output_bytes // 2,
            max_stderr_bytes=max_output_bytes - (max_output_bytes // 2),
            max_lines=int(config.get("max_lines", 200)),
            timeout_seconds=float(config["timeout_seconds"]),
        )
        for family in enabled_families
    ]


def _unavailable_result(
    decision: SystemDiagnosticsDecision,
    *,
    source: str,
    reason: str,
) -> ToolInvocationResult:
    content = {
        "source": source,
        "available": False,
        "reason": reason,
        "readings": [],
    }
    encoded = json.dumps(content, sort_keys=True)
    return ToolInvocationResult(
        content=encoded,
        content_type="application/json",
        truncated=False,
        output_bytes=len(encoded.encode("utf-8")),
        metadata={
            "family": decision.family.value if decision.family is not None else None,
            "cwd": decision.cwd,
            "source": source,
            "unavailable": True,
        },
    )


def _sensor_invocation_result(
    decision: SystemDiagnosticsDecision,
    result: ShellExecutionResult,
) -> ToolInvocationResult | None:
    source = decision.argv[0] if decision.argv else "sensor"
    if result.exit_code != 0 and _permission_required(result.stderr):
        snapshot = SensorSnapshot.unavailable(source=source, reason="permission_required")
        return _sensor_snapshot_result(decision, snapshot)

    if source == "nvidia-smi":
        snapshot = _parse_nvidia_smi_temperatures(result.stdout)
    elif source in {"sensors", "powermetrics"}:
        snapshot = _parse_text_temperatures(source, result.stdout)
    else:
        return None
    return _sensor_snapshot_result(decision, snapshot.normalized_celsius())


def _sensor_snapshot_result(
    decision: SystemDiagnosticsDecision,
    snapshot: SensorSnapshot,
) -> ToolInvocationResult:
    content = snapshot.to_dict()
    encoded = json.dumps(content, sort_keys=True)
    return ToolInvocationResult(
        content=encoded,
        content_type="application/json",
        truncated=False,
        output_bytes=len(encoded.encode("utf-8")),
        metadata={
            "family": decision.family.value if decision.family is not None else None,
            "cwd": decision.cwd,
            "source": snapshot.source,
            "unavailable": not snapshot.available,
        },
    )


def _permission_required(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in (
            "permission",
            "must be run as root",
            "requires root",
            "superuser",
            "sudo",
        )
    )


def _parse_text_temperatures(source: str, stdout: str) -> SensorSnapshot:
    readings: list[SensorReading] = []
    for index, line in enumerate(stdout.splitlines(), start=1):
        match = re.search(
            r"(?P<label>[^:\n]+):.*?(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?:°\s*)?(?P<unit>[CFK])\b",
            line,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        readings.append(
            SensorReading(
                label=match.group("label").strip() or f"sensor-{index}",
                value=float(match.group("value")),
                unit=match.group("unit").upper(),
                source=source,
            ),
        )
    if not readings:
        return SensorSnapshot.unavailable(source=source, reason="no_temperature_readings")
    return SensorSnapshot(source=source, readings=readings)


def _parse_nvidia_smi_temperatures(stdout: str) -> SensorSnapshot:
    readings: list[SensorReading] = []
    for index, line in enumerate(stdout.splitlines()):
        raw = line.strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            return SensorSnapshot.unavailable(source="nvidia-smi", reason="invalid_temperature_reading")
        readings.append(
            SensorReading(
                label=f"gpu{index}",
                value=value,
                unit="C",
                source="nvidia-smi",
            ),
        )
    if not readings:
        return SensorSnapshot.unavailable(source="nvidia-smi", reason="no_temperature_readings")
    return SensorSnapshot(source="nvidia-smi", readings=readings)


def _arguments_argv(arguments: dict[str, Any]) -> list[str]:
    argv = arguments.get("argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("argv must be a list of strings")
    return argv


def _arguments_cwd(arguments: dict[str, Any]) -> str:
    cwd = arguments.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("cwd must be a non-empty string")
    return cwd


def _resolve_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve(strict=True)


def _resolve_command_path(raw_path: str, cwd: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve(strict=True)


def _secret_like_path(raw_path: str) -> bool:
    lowered = raw_path.lower()
    parts = [part.lower() for part in Path(raw_path).parts]
    if any(part == ".env" or part.startswith(".env.") for part in parts):
        return True
    if any(part in {".git", ".ssh", "id_rsa", "id_ed25519", "known_hosts"} for part in parts):
        return True
    if any(part in {".aws", ".gcp", ".azure", ".kube"} for part in parts):
        return True
    if lowered.endswith((".pem", ".key", ".crt")):
        return True
    return any(marker in lowered for marker in ("credentials", "token", "secrets"))


def _syntax_decision(
    argv: tuple[str, ...],
    cwd: Path,
) -> SystemDiagnosticsDecision | None:
    if not argv:
        return None
    if _ENV_ASSIGNMENT.match(argv[0]):
        return _deny("shell_syntax_denied", "environment assignment prefixes are denied", argv, cwd)
    for arg in argv:
        if any(marker in arg for marker in _SHELL_SYNTAX_MARKERS):
            return _deny("shell_syntax_denied", "shell metacharacters are denied", argv, cwd)
    return None


def _command_is_path(command: str) -> bool:
    return (
        "/" in command
        or "\\" in command
        or command in {".", ".."}
        or command.startswith(".")
        or Path(command).is_absolute()
    )


def _safe_pattern(pattern: str) -> bool:
    return 0 < len(pattern) <= 128 and not any(marker in pattern for marker in _SHELL_SYNTAX_MARKERS)


def _is_pmset_mutation(argv: tuple[str, ...]) -> bool:
    if not argv or argv[0] != "pmset":
        return False
    return argv != ("pmset", "-g", "batt")


def _option_value(argv: tuple[str, ...], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _normalize_platform(platform: str) -> str:
    lowered = platform.lower()
    if lowered.startswith("darwin"):
        return "darwin"
    if lowered.startswith("linux"):
        return "linux"
    return lowered


def _allow(
    family: SystemDiagnosticsFamily,
    argv: tuple[str, ...],
    cwd: Path,
    platform: str,
) -> SystemDiagnosticsDecision:
    return SystemDiagnosticsDecision(
        allowed=True,
        code="allowed",
        reason="diagnostics command is allowed",
        family=family,
        argv=argv,
        cwd=str(cwd),
        metadata={"platform": platform},
    )


def _deny(
    code: str,
    reason: str,
    argv: tuple[str, ...],
    cwd: Path | None,
) -> SystemDiagnosticsDecision:
    return SystemDiagnosticsDecision(
        allowed=False,
        code=code,
        reason=reason,
        argv=argv,
        cwd=str(cwd) if cwd is not None else None,
    )


def _is_inside_any_root(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _bounded_text(text: str, *, max_bytes: int, max_lines: int) -> _LimitedText:
    raw_bytes = len(text.encode("utf-8"))
    truncated = False
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        text = "".join(lines[:max_lines])
        truncated = True
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        text = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True
    return _LimitedText(text=text, raw_bytes=raw_bytes, truncated=truncated)


def _redact_diagnostics_output(output: str) -> str:
    redacted = _CREDENTIAL_URL.sub(
        lambda match: f"{match.group('scheme')}<redacted>@{match.group(0).split('@', 1)[1]}",
        output,
    )
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def _redacted_argv(argv: tuple[str, ...]) -> list[str]:
    if not argv:
        return []
    redacted = [argv[0] if argv[0] in _KNOWN_COMMANDS and not _command_is_path(argv[0]) else "<command>"]
    for arg in argv[1:]:
        redacted.append("<option>" if arg.startswith("-") else "<arg>")
    return redacted


def _redacted_cwd(cwd: str | None, *, allowed: bool) -> str | None:
    if cwd is None:
        return None
    if not allowed:
        return "<redacted>"
    return cwd
