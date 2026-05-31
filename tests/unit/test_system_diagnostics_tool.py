from __future__ import annotations

from pathlib import Path

import pytest

from assistant_core.domain.system_diagnostics import SensorReading, SensorSnapshot
from assistant_core.domain.system_diagnostics import SystemDiagnosticsFamily
from assistant_core.tools.system_diagnostics import SystemDiagnosticsClassifier


pytestmark = pytest.mark.unit


def _classifier(root: Path, *, platform: str = "darwin") -> SystemDiagnosticsClassifier:
    return SystemDiagnosticsClassifier(allowed_roots=[root], platform=platform)


def test_allows_ps_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["ps", "-Ao", "pid,comm,command"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.PROCESS


def test_allows_pgrep(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["pgrep", "-fl", "ollama"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.PROCESS


def test_denies_pipeline_style_ps_grep(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(
        ["ps", "-Ao", "pid,comm,command", "|", "grep", "HFT"],
        cwd=tmp_path,
    )

    assert decision.allowed is False
    assert decision.code == "shell_syntax_denied"


def test_allows_uptime(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["uptime"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_df(tmp_path: Path) -> None:
    decision = _classifier(tmp_path).classify(["df", "-h"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_macos_battery_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(
        ["pmset", "-g", "batt"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


def test_allows_macos_vpn_status_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(
        ["scutil", "--nc", "list"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.NETWORK


def test_allows_du_inside_workspace(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()

    decision = _classifier(tmp_path).classify(["du", "-sh", "docs"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_denies_du_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()

    decision = _classifier(workspace).classify(["du", "-sh", "../outside"], cwd=workspace)

    assert decision.allowed is False
    assert decision.code == "path_outside_workspace"


def test_denies_secret_like_cwd(tmp_path: Path) -> None:
    secret_cwd = tmp_path / ".ssh"
    secret_cwd.mkdir()

    decision = _classifier(tmp_path).classify(["ps", "-Ao", "pid,comm,command"], cwd=secret_cwd)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"
    assert decision.cwd == str(secret_cwd)


def test_denies_du_secret_like_path(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("SECRET=value\n", encoding="utf-8")

    decision = _classifier(tmp_path).classify(["du", "-sh", ".env"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "secret_path_denied"


def test_allows_macos_top_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(["top", "-l", "1"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_macos_top_summary_without_process_rows(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(
        ["top", "-l", "1", "-n", "0"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_macos_vm_stat(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(["vm_stat"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_macos_sysctl_selected_keys(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(
        ["sysctl", "-n", "hw.memsize"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


def test_allows_macos_sw_vers_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(["sw_vers"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


def test_allows_linux_top_batch_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(
        ["top", "-b", "-n", "1"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_linux_free(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(["free", "-m"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.RESOURCES


def test_allows_linux_lscpu(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(["lscpu"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


def test_allows_linux_uname_os_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(["uname", "-a"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


def test_allows_linux_upower_battery_snapshot(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(
        ["upower", "-i", "/org/freedesktop/UPower/devices/DisplayDevice"],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


def test_allows_linux_lshw(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(["lshw", "-short"], cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.HARDWARE


@pytest.mark.parametrize(
    ("platform", "argv"),
    [
        ("darwin", ["netstat", "-an"]),
        ("darwin", ["ifconfig"]),
        ("darwin", ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"]),
        ("linux", ["netstat", "-an"]),
        ("linux", ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"]),
        ("linux", ["ss", "-tulpen"]),
        ("linux", ["ip", "addr"]),
    ],
)
def test_allows_network_diagnostics_selected_flags(
    tmp_path: Path,
    platform: str,
    argv: list[str],
) -> None:
    decision = _classifier(tmp_path, platform=platform).classify(argv, cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.NETWORK


def test_denies_linux_ifconfig_to_match_platform_allowlist(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(["ifconfig"], cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "unsupported_platform_command"


@pytest.mark.parametrize("argv", [["htop"], ["watch", "ps"], ["less", "/tmp/x"], ["vim", "x"]])
def test_denies_interactive_diagnostics(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "interactive_command_denied"


@pytest.mark.parametrize(
    "argv",
    [
        ["sudo", "powermetrics"],
        ["kill", "123"],
        ["renice", "10", "123"],
        ["launchctl", "list"],
        ["systemctl", "status"],
    ],
)
def test_denies_kill_sudo_and_system_mutations(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "mutating_command_denied"


@pytest.mark.parametrize(
    "argv",
    [["curl", "https://example.com"], ["wget", "https://example.com"], ["nc", "-vz", "127.0.0.1", "80"], ["ssh", "host"]],
)
def test_denies_network_clients(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path).classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "network_client_denied"


@pytest.mark.parametrize(
    ("platform", "argv"),
    [
        ("darwin", ["powermetrics", "--samplers", "smc", "-n", "1"]),
        ("darwin", ["powermetrics", "--samplers", "thermal", "-n", "1"]),
        ("linux", ["sensors"]),
        ("linux", ["thermal-sysfs"]),
    ],
)
def test_allows_temperature_sensor_snapshot(tmp_path: Path, platform: str, argv: list[str]) -> None:
    decision = _classifier(tmp_path, platform=platform).classify(argv, cwd=tmp_path)

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.SENSORS


def test_temperature_snapshot_normalizes_celsius_when_possible() -> None:
    snapshot = SensorSnapshot(
        source="fake",
        readings=[
            SensorReading(label="cpu", value=149.0, unit="F"),
            SensorReading(label="gpu", value=42.5, unit="C"),
        ],
    ).normalized_celsius()

    assert [reading.unit for reading in snapshot.readings] == ["C", "C"]
    assert snapshot.readings[0].value == pytest.approx(65.0)
    assert snapshot.readings[1].value == pytest.approx(42.5)


def test_temperature_source_unavailable_is_non_fatal() -> None:
    snapshot = SensorSnapshot.unavailable(source="sensors", reason="binary not found")

    assert snapshot.available is False
    assert snapshot.reason == "binary not found"
    assert snapshot.readings == []


def test_denies_sudo_powermetrics(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(
        ["sudo", "powermetrics", "--samplers", "smc", "-n", "1"],
        cwd=tmp_path,
    )

    assert decision.allowed is False
    assert decision.code == "mutating_command_denied"


def test_denies_unsupported_powermetrics_sampler_as_arguments_on_macos(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(
        ["powermetrics", "--samplers", "cpu_power", "-n", "1"],
        cwd=tmp_path,
    )

    assert decision.allowed is False
    assert decision.code == "unsupported_arguments"


@pytest.mark.parametrize(
    "argv",
    [
        ["tee", "/sys/class/thermal/thermal_zone0/temp"],
        ["echo", "1", "/sys/class/thermal/thermal_zone0/temp"],
        ["fanctl", "set", "100"],
        ["pmset", "-a", "lowpowermode", "1"],
    ],
)
def test_denies_sensor_write_paths(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code == "sensor_mutation_denied"


@pytest.mark.parametrize(
    "argv",
    [
        ["powermetrics", "--samplers", "smc", "-n", "0"],
        ["powermetrics", "--samplers", "smc", "-i", "1000"],
        ["watch", "sensors"],
    ],
)
def test_denies_long_running_sensor_polling(tmp_path: Path, argv: list[str]) -> None:
    decision = _classifier(tmp_path, platform="darwin").classify(argv, cwd=tmp_path)

    assert decision.allowed is False
    assert decision.code in {"sensor_polling_denied", "interactive_command_denied"}


def test_gpu_temperature_uses_nvidia_smi_query_mode(tmp_path: Path) -> None:
    decision = _classifier(tmp_path, platform="linux").classify(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=tmp_path,
    )

    assert decision.allowed is True
    assert decision.family == SystemDiagnosticsFamily.SENSORS


def test_platform_specific_classifier_is_deterministic(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, platform="linux")

    first = classifier.classify(["top", "-b", "-n", "1"], cwd=tmp_path)
    second = classifier.classify(["top", "-b", "-n", "1"], cwd=tmp_path)

    assert first == second
