from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from assistant_core.domain.events import EventType
from assistant_core.domain.policy import Capability, PolicyDecision, PolicyDecisionOutcome, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.system_diagnostics import SensorReading, SensorSnapshot
from assistant_core.domain.system_diagnostics import SystemDiagnosticsFamily
from assistant_core.domain.tools import ToolCallRequest, ToolObservationStatus, ToolParseStatus
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.event_log import EventFilter
from assistant_core.tools.gateway import ToolGateway
from assistant_core.tools.registry import ToolRegistry
from assistant_core.tools.shell_read import ShellExecutionResult, ShellExecutionTimeout
from assistant_core.tools.system_diagnostics import SystemDiagnosticsTool


pytestmark = pytest.mark.contract


class AllowPolicy:
    def __init__(self, call_log: list[str] | None = None) -> None:
        self.requests = []
        self.call_log = call_log

    async def evaluate_capability_request(self, request):
        if self.call_log is not None:
            self.call_log.append("policy")
        self.requests.append(request)
        return PolicyDecision(
            allowed=True,
            code="allowed_system_diagnostics",
            reason="contract policy decision",
            outcome=PolicyDecisionOutcome.ALLOW,
            capability=request.capability,
            risk_classes=request.risk_classes,
            sensitivity=request.sensitivity,
            permission_mode=request.permission_mode,
        )


class RecordingDiagnosticsExecutor:
    def __init__(
        self,
        result: ShellExecutionResult | None = None,
        *,
        timeout: bool = False,
        call_log: list[str] | None = None,
    ) -> None:
        self.result = result or ShellExecutionResult(exit_code=0, stdout="ok\n", stderr="")
        self.timeout = timeout
        self.call_log = call_log
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> ShellExecutionResult:
        if self.call_log is not None:
            self.call_log.append("executor")
        self.calls.append(
            {
                "argv": argv,
                "cwd": cwd,
                "env": env,
                "timeout_seconds": timeout_seconds,
            },
        )
        if self.timeout:
            raise ShellExecutionTimeout("timed out")
        return self.result


class FakeSensorProvider:
    def __init__(self, snapshot: SensorSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    async def snapshot_temperatures(self) -> SensorSnapshot:
        self.calls += 1
        return self.snapshot


def _tool(
    tmp_path: Path,
    executor: RecordingDiagnosticsExecutor,
    *,
    family: SystemDiagnosticsFamily = SystemDiagnosticsFamily.PROCESS,
    sensor_provider: FakeSensorProvider | None = None,
    platform: str = "linux",
) -> SystemDiagnosticsTool:
    return SystemDiagnosticsTool(
        family=family,
        allowed_roots=[tmp_path],
        executor=executor,
        sensor_provider=sensor_provider,
        max_stdout_bytes=200,
        max_stderr_bytes=200,
        max_lines=3,
        timeout_seconds=0.5,
        platform=platform,
    )


def _gateway(
    tmp_path: Path,
    executor: RecordingDiagnosticsExecutor,
    *,
    family: SystemDiagnosticsFamily = SystemDiagnosticsFamily.PROCESS,
    sensor_provider: FakeSensorProvider | None = None,
    platform: str = "linux",
    policy: AllowPolicy | None = None,
):
    event_log = InMemoryEventLog()
    policy = policy or AllowPolicy()
    gateway = ToolGateway(
        registry=ToolRegistry(
            [
                _tool(
                    tmp_path,
                    executor,
                    family=family,
                    sensor_provider=sensor_provider,
                    platform=platform,
                ),
            ],
        ),
        policy=policy,
        event_log=event_log,
    )
    return gateway, policy, event_log


def _request(
    tmp_path: Path,
    argv: list[str],
    *,
    tool_name: str = "tool.system.read.process",
    sensitivity: Sensitivity = Sensitivity.INFRA,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        arguments={"argv": argv, "cwd": str(tmp_path)},
        request_id="req-system-diagnostics-contract",
        conversation_id="conv-system-diagnostics-contract",
        user_id="user-system-diagnostics-contract",
        working_directory=str(tmp_path),
        sensitivity=sensitivity,
    )


def _request_with_arguments(
    tmp_path: Path,
    arguments: dict[str, Any],
    *,
    tool_name: str = "tool.system.read.process",
    sensitivity: Sensitivity = Sensitivity.INFRA,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        arguments=arguments,
        request_id="req-system-diagnostics-contract",
        conversation_id="conv-system-diagnostics-contract",
        user_id="user-system-diagnostics-contract",
        working_directory=str(tmp_path),
        sensitivity=sensitivity,
    )


def _json_content(observation) -> dict[str, Any]:
    return json.loads(observation.content)


def test_system_diagnostics_tool_executes_allowed_argv_command(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="123 ollama\n", stderr=""),
    )
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert _json_content(observation)["stdout"] == "123 ollama\n"
    assert executor.calls[0]["argv"] == ["ps", "-Ao", "pid,comm,command"]
    assert policy.requests[0].capability == Capability.TOOL_SYSTEM_READ_PROCESS
    assert policy.requests[0].risk_classes == frozenset({RiskClass.READ_ONLY})


def test_system_diagnostics_tool_uses_request_working_directory_when_cwd_missing(
    tmp_path: Path,
) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="123 ollama\n", stderr=""),
    )
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(
            ToolCallRequest(
                tool_name="tool.system.read.process",
                arguments={"argv": ["ps", "-Ao", "pid,comm,command"]},
                request_id="req-system-diagnostics-contract",
                conversation_id="conv-system-diagnostics-contract",
                user_id="user-system-diagnostics-contract",
                working_directory=str(tmp_path),
                sensitivity=Sensitivity.INFRA,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert executor.calls[0]["cwd"] == tmp_path
    assert policy.requests[0].working_directory == str(tmp_path)


def test_system_diagnostics_tool_does_not_use_cwd_argument_as_request_scope(
    tmp_path: Path,
) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="123 ollama\n", stderr=""),
    )
    gateway, policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(
            ToolCallRequest(
                tool_name="tool.system.read.process",
                arguments={"argv": ["ps", "-Ao", "pid,comm,command"], "cwd": str(tmp_path)},
                request_id="req-system-diagnostics-contract",
                conversation_id="conv-system-diagnostics-contract",
                user_id="user-system-diagnostics-contract",
                sensitivity=Sensitivity.INFRA,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.DENIED
    assert observation.error["code"] == "working_directory_required"
    assert executor.calls == []
    assert policy.requests == []


def test_system_diagnostics_tool_rejects_unknown_arguments_before_policy_or_execution(
    tmp_path: Path,
) -> None:
    call_log: list[str] = []
    executor = RecordingDiagnosticsExecutor(call_log=call_log)
    gateway, policy, _event_log = _gateway(
        tmp_path,
        executor,
        policy=AllowPolicy(call_log=call_log),
    )

    observation = asyncio.run(
        gateway.invoke(
            _request_with_arguments(
                tmp_path,
                {
                    "argv": ["ps", "-Ao", "pid,comm,command"],
                    "cwd": str(tmp_path),
                    "github_pat_secret_key": "value",
                },
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.FAILED
    assert observation.error["code"] == "invalid_arguments"
    assert policy.requests == []
    assert executor.calls == []
    assert call_log == []


def test_system_diagnostics_tool_returns_bounded_stdout(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="a\nb\nc\nd\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))
    content = _json_content(observation)

    assert observation.truncated is True
    assert content["stdout"] == "a\nb\nc\n"
    assert observation.metadata["stdout_truncated"] is True


def test_system_diagnostics_tool_truncates_large_output_with_metadata(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(ShellExecutionResult(exit_code=0, stdout="x" * 300, stderr=""))
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-system-diagnostics-contract")))

    assert observation.truncated is True
    assert observation.metadata["raw_stdout_bytes"] == 300
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_OUTPUT_TRUNCATED in [event.event_type for event in events]


def test_system_diagnostics_tool_times_out(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(timeout=True)
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-system-diagnostics-contract")))

    assert observation.status == ToolObservationStatus.TIMEOUT
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_TIMEOUT in [event.event_type for event in events]


def test_system_diagnostics_tool_redacts_process_command_line_secrets(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="123 python app.py --api_key=sk-prod-token\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))
    content = _json_content(observation)

    assert "sk-prod-token" not in observation.content
    assert "api_key" not in observation.content
    assert content["stdout"] == "123 python app.py <redacted>\n"


def test_system_diagnostics_tool_redacts_auth_flags_and_key_values(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout=(
                "123 server --auth eyJhbGciOiJIUzI1NiJ9.payload.signature\n"
                "456 worker AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
            ),
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))
    content = _json_content(observation)

    assert "eyJhbGci" not in observation.content
    assert "AKIAIOSFODNN7EXAMPLE" not in observation.content
    assert content["stdout"] == "123 server <redacted>\n456 worker <redacted>\n"


def test_system_diagnostics_tool_redacts_network_sensitive_output(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="tcp token=secret localhost:8080\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.NETWORK,
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                ["netstat", "-an"],
                tool_name="tool.system.read.network",
            ),
        ),
    )
    content = _json_content(observation)

    assert "secret" not in observation.content
    assert "token" not in observation.content
    assert content["stdout"] == "tcp <redacted> localhost:8080\n"


def test_system_diagnostics_tool_redacts_credential_urls(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout=(
                "postgres://jarvis:dbpass@127.0.0.1:55432/jarvis\n"
                "redis://:redispass@127.0.0.1:6379/0\n"
            ),
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.NETWORK,
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                ["netstat", "-an"],
                tool_name="tool.system.read.network",
            ),
        ),
    )
    content = _json_content(observation)

    assert "dbpass" not in observation.content
    assert "redispass" not in observation.content
    assert content["stdout"] == (
        "postgres://<redacted>@127.0.0.1:55432/jarvis\n"
        "redis://<redacted>@127.0.0.1:6379/0\n"
    )


def test_system_diagnostics_tool_returns_typed_os_version_payload(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout="ProductName:\t\tmacOS\nProductVersion:\t\t15.6\nBuildVersion:\t\t24G84\n",
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.HARDWARE,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(tmp_path, ["sw_vers"], tool_name="tool.system.read.hardware"),
        ),
    )

    assert observation.structured_schema == "system.os_version"
    assert observation.structured_schema_version == 1
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content == {
        "product_name": "macOS",
        "version": "15.6",
        "build": "24G84",
        "platform": "darwin",
        "source": "sw_vers",
    }


def test_system_diagnostics_tool_returns_typed_battery_payload(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout=(
                "Now drawing from 'Battery Power'\n"
                " -InternalBattery-0 (id=1234567)\t82%; discharging; 4:12 remaining present: true\n"
            ),
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.HARDWARE,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(tmp_path, ["pmset", "-g", "batt"], tool_name="tool.system.read.hardware"),
        ),
    )

    assert observation.structured_schema == "system.battery_charge"
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content["percent"] == 82
    assert observation.structured_content["state"] == "discharging"
    assert observation.structured_content["source"] == "pmset"


def test_system_diagnostics_tool_returns_typed_disk_payload(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout=(
                "Filesystem      Size  Used Avail Use% Mounted on\n"
                "/dev/disk3s1s1  460G   15G  120G  12% /\n"
            ),
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.RESOURCES,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(_request(tmp_path, ["df", "-h"], tool_name="tool.system.read.resources")),
    )

    assert observation.structured_schema == "system.disk_free"
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content["filesystems"] == [
        {
            "filesystem": "/dev/disk3s1s1",
            "mount": "/",
            "size": "460G",
            "used": "15G",
            "available": "120G",
            "used_percent": "12%",
        },
    ]


def test_system_diagnostics_tool_returns_typed_memory_payload(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout=(
                "              total        used        free      shared  buff/cache   available\n"
                "Mem:          32768       12000        1024         128       19744       18000\n"
                "Swap:          2048         256        1792\n"
            ),
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.RESOURCES,
        platform="linux",
    )

    observation = asyncio.run(
        gateway.invoke(_request(tmp_path, ["free", "-m"], tool_name="tool.system.read.resources")),
    )

    assert observation.structured_schema == "system.memory_overview"
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content["total"] == "32768 MiB"
    assert observation.structured_content["used"] == "12000 MiB"
    assert observation.structured_content["free"] == "1024 MiB"
    assert observation.structured_content["available"] == "18000 MiB"
    assert observation.structured_content["swap_total"] == "2048 MiB"
    assert observation.structured_content["swap_used"] == "256 MiB"


def test_system_diagnostics_resources_defaults_to_safe_snapshot_when_arguments_are_empty(
    tmp_path: Path,
) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout=(
                "CPU usage: 12.5% user, 7.5% sys, 80.0% idle\n"
                "PhysMem: 16G used (2G wired), 16G unused.\n"
            ),
            stderr="",
        ),
    )
    gateway, policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.RESOURCES,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request_with_arguments(
                tmp_path,
                {},
                tool_name="tool.system.read.resources",
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert executor.calls[0]["argv"] == ["top", "-l", "1", "-n", "0"]
    assert executor.calls[0]["cwd"] == tmp_path
    assert policy.requests[0].capability == Capability.TOOL_SYSTEM_READ_RESOURCES
    assert _json_content(observation)["stdout"].startswith("CPU usage:")


def test_system_diagnostics_resources_ignores_model_hint_arguments_for_default_snapshot(
    tmp_path: Path,
) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="CPU usage: 1% user, 99% idle\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.RESOURCES,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request_with_arguments(
                tmp_path,
                {"metric": "cpu_and_memory"},
                tool_name="tool.system.read.resources",
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.COMPLETED
    assert executor.calls[0]["argv"] == ["top", "-l", "1", "-n", "0"]


def test_system_diagnostics_resources_rejects_invalid_metric_before_policy_or_execution(
    tmp_path: Path,
) -> None:
    call_log: list[str] = []
    executor = RecordingDiagnosticsExecutor(call_log=call_log)
    gateway, policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.RESOURCES,
        platform="darwin",
        policy=AllowPolicy(call_log=call_log),
    )

    observation = asyncio.run(
        gateway.invoke(
            _request_with_arguments(
                tmp_path,
                {"metric": "github_pat_secret_key"},
                tool_name="tool.system.read.resources",
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.FAILED
    assert observation.error["code"] == "invalid_arguments"
    assert policy.requests == []
    assert executor.calls == []
    assert call_log == []


@pytest.mark.parametrize(
    ("family", "tool_name", "platform"),
    [
        (SystemDiagnosticsFamily.PROCESS, "tool.system.read.process", "linux"),
        (SystemDiagnosticsFamily.HARDWARE, "tool.system.read.hardware", "darwin"),
        (SystemDiagnosticsFamily.NETWORK, "tool.system.read.network", "darwin"),
        (SystemDiagnosticsFamily.SENSORS, "tool.system.read.sensors", "linux"),
    ],
)
def test_system_diagnostics_non_resources_tools_reject_empty_arguments_before_policy_or_execution(
    tmp_path: Path,
    family: SystemDiagnosticsFamily,
    tool_name: str,
    platform: str,
) -> None:
    call_log: list[str] = []
    executor = RecordingDiagnosticsExecutor(call_log=call_log)
    gateway, policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=family,
        platform=platform,
        policy=AllowPolicy(call_log=call_log),
    )

    observation = asyncio.run(
        gateway.invoke(
            _request_with_arguments(
                tmp_path,
                {},
                tool_name=tool_name,
            ),
        ),
    )

    assert observation.status == ToolObservationStatus.FAILED
    assert observation.error["code"] == "invalid_arguments"
    assert policy.requests == []
    assert executor.calls == []
    assert call_log == []


def test_system_diagnostics_tool_returns_typed_vpn_payload(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout="* (Connected)   JarvisVPN               [VPN]\n",
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.NETWORK,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(tmp_path, ["scutil", "--nc", "list"], tool_name="tool.system.read.network"),
        ),
    )

    assert observation.structured_schema == "system.vpn_status"
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content["connected"] is True
    assert observation.structured_content["interface_or_service"] == "JarvisVPN"


def test_system_diagnostics_tool_marks_failed_vpn_command_unparsed(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=1, stdout="", stderr="scutil: unavailable"),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.NETWORK,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(tmp_path, ["scutil", "--nc", "list"], tool_name="tool.system.read.network"),
        ),
    )

    assert observation.structured_schema == "system.vpn_status"
    assert observation.parse_status is ToolParseStatus.UNPARSED
    assert observation.structured_content is None
    assert "command_failed" in observation.parse_warnings


def test_system_diagnostics_tool_returns_typed_process_search_payload(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="12345 HFT-strategy-runner\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(
        gateway.invoke(_request(tmp_path, ["pgrep", "-l", "HFT"], tool_name="tool.system.read.process")),
    )

    assert observation.structured_schema == "system.process_name_search"
    assert observation.parse_status is ToolParseStatus.PARSED
    assert observation.structured_content["query"] == "HFT"
    assert observation.structured_content["matches"] == [
        {"pid": 12345, "name": "HFT-strategy-runner"},
    ]


def test_sensor_command_stdout_returns_normalized_snapshot(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(
            exit_code=0,
            stdout="Package id 0: +149.0°F\nGPU Temp: +42.5°C\n",
            stderr="",
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.SENSORS,
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                ["sensors"],
                tool_name="tool.system.read.sensors",
            ),
        ),
    )
    content = _json_content(observation)

    assert observation.status == ToolObservationStatus.COMPLETED
    assert content["source"] == "sensors"
    assert content["available"] is True
    assert content["readings"][0]["unit"] == "C"
    assert content["readings"][0]["value"] == pytest.approx(65.0)
    assert content["readings"][1]["value"] == pytest.approx(42.5)


def test_nvidia_smi_temperature_query_returns_sensor_snapshot(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=0, stdout="43\n41\n", stderr=""),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.SENSORS,
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                tool_name="tool.system.read.sensors",
            ),
        ),
    )
    content = _json_content(observation)

    assert observation.status == ToolObservationStatus.COMPLETED
    assert content["source"] == "nvidia-smi"
    assert [reading["value"] for reading in content["readings"]] == [43.0, 41.0]
    assert {reading["unit"] for reading in content["readings"]} == {"C"}


def test_powermetrics_permission_required_returns_unavailable_snapshot(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(
        ShellExecutionResult(exit_code=1, stdout="", stderr="powermetrics must be invoked as the superuser\n"),
    )
    gateway, _policy, event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.SENSORS,
        platform="darwin",
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                ["powermetrics", "--samplers", "thermal", "-n", "1"],
                tool_name="tool.system.read.sensors",
            ),
        ),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-system-diagnostics-contract")))
    content = _json_content(observation)

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.metadata["unavailable"] is True
    assert content["available"] is False
    assert content["reason"] == "permission_required"
    assert "root" not in observation.content.lower()
    assert "superuser" not in observation.content.lower()
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE in [event.event_type for event in events]


def test_sensor_backend_unavailable_returns_unavailable_observation(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor()
    sensor_provider = FakeSensorProvider(SensorSnapshot.unavailable(source="thermal-sysfs", reason="not available"))
    gateway, _policy, event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.SENSORS,
        sensor_provider=sensor_provider,
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                ["thermal-sysfs"],
                tool_name="tool.system.read.sensors",
            ),
        ),
    )
    events = asyncio.run(event_log.query(EventFilter(request_id="req-system-diagnostics-contract")))
    content = _json_content(observation)

    assert observation.status == ToolObservationStatus.COMPLETED
    assert observation.metadata["unavailable"] is True
    assert content["available"] is False
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_UNAVAILABLE in [event.event_type for event in events]
    assert executor.calls == []
    assert sensor_provider.calls == 1


def test_sensor_backend_snapshot_is_normalized(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor()
    sensor_provider = FakeSensorProvider(
        SensorSnapshot(
            source="thermal-sysfs",
            readings=[SensorReading(label="cpu", value=149.0, unit="F")],
        ),
    )
    gateway, _policy, _event_log = _gateway(
        tmp_path,
        executor,
        family=SystemDiagnosticsFamily.SENSORS,
        sensor_provider=sensor_provider,
    )

    observation = asyncio.run(
        gateway.invoke(
            _request(
                tmp_path,
                ["thermal-sysfs"],
                tool_name="tool.system.read.sensors",
            ),
        ),
    )
    content = _json_content(observation)

    assert content["readings"][0]["unit"] == "C"
    assert content["readings"][0]["value"] == pytest.approx(65.0)


def test_system_diagnostics_tool_emits_audit_events(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor(ShellExecutionResult(exit_code=0, stdout="ok\n", stderr=""))
    gateway, _policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-system-diagnostics-contract")))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert [
        event.event_type
        for event in events
        if event.event_type.value.startswith("tool.system.diagnostics.")
    ] == [
        EventType.TOOL_SYSTEM_DIAGNOSTICS_CLASSIFIED,
        EventType.TOOL_SYSTEM_DIAGNOSTICS_STARTED,
        EventType.TOOL_SYSTEM_DIAGNOSTICS_COMPLETED,
    ]


def test_system_diagnostics_tool_returns_denied_observation_without_execution(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor()
    gateway, policy, event_log = _gateway(tmp_path, executor)

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["kill", "123"])))
    events = asyncio.run(event_log.query(EventFilter(request_id="req-system-diagnostics-contract")))

    assert observation.status == ToolObservationStatus.DENIED
    assert observation.error["code"] == "mutating_command_denied"
    assert executor.calls == []
    assert len(policy.requests) == 1
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_DENIED in [event.event_type for event in events]
    assert EventType.TOOL_SYSTEM_DIAGNOSTICS_STARTED not in [event.event_type for event in events]


def test_toolgateway_consults_policy_before_diagnostics_execution(tmp_path: Path) -> None:
    call_log: list[str] = []
    executor = RecordingDiagnosticsExecutor(call_log=call_log)
    gateway, _policy, _event_log = _gateway(tmp_path, executor, policy=AllowPolicy(call_log=call_log))

    observation = asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))

    assert observation.status == ToolObservationStatus.COMPLETED
    assert call_log == ["policy", "executor"]


def test_system_diagnostics_tool_uses_minimal_environment(tmp_path: Path) -> None:
    executor = RecordingDiagnosticsExecutor()
    gateway, _policy, _event_log = _gateway(tmp_path, executor)

    asyncio.run(gateway.invoke(_request(tmp_path, ["ps", "-Ao", "pid,comm,command"])))

    env = executor.calls[0]["env"]
    assert sorted(env) == ["LANG", "LC_ALL", "PATH"]
    assert "HOME" not in env
    assert "TOKEN" not in str(env).upper()
