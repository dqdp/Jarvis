from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from assistant_core.domain.system_diagnostics import SystemDiagnosticsDecision
from assistant_core.domain.tools import ToolParseStatus


@dataclass(frozen=True)
class DiagnosticsStructuredPayload:
    structured_content: dict[str, Any] | None
    structured_schema: str | None
    structured_schema_version: int | None
    parse_status: ToolParseStatus
    parse_warnings: tuple[str, ...] = ()


def not_applicable() -> DiagnosticsStructuredPayload:
    return DiagnosticsStructuredPayload(
        structured_content=None,
        structured_schema=None,
        structured_schema_version=None,
        parse_status=ToolParseStatus.NOT_APPLICABLE,
    )


def normalize_command_output(
    *,
    decision: SystemDiagnosticsDecision,
    stdout: str,
    stderr: str,
    exit_code: int,
    platform: str,
) -> DiagnosticsStructuredPayload:
    argv = tuple(decision.argv)
    if argv == ("sw_vers",):
        if exit_code != 0:
            return _unparsed("system.os_version", "command_failed")
        return _normalize_sw_vers(stdout, platform=platform)
    if argv == ("uname", "-a"):
        if exit_code != 0:
            return _unparsed("system.os_version", "command_failed")
        return _parsed(
            "system.os_version",
            {
                "product_name": "Linux",
                "version": stdout.strip().splitlines()[0] if stdout.strip() else "",
                "build": None,
                "platform": platform,
                "source": "uname",
            },
        )
    if argv == ("pmset", "-g", "batt"):
        if exit_code != 0:
            return _unparsed("system.battery_charge", "command_failed")
        return _normalize_pmset_battery(stdout)
    if argv == ("upower", "-i", "/org/freedesktop/UPower/devices/DisplayDevice"):
        if exit_code != 0:
            return _unparsed("system.battery_charge", "command_failed")
        return _normalize_upower_battery(stdout)
    if argv == ("df", "-h"):
        if exit_code != 0:
            return _unparsed("system.disk_free", "command_failed")
        return _normalize_df(stdout)
    if argv == ("free", "-m"):
        if exit_code != 0:
            return _unparsed("system.memory_overview", "command_failed")
        return _normalize_free(stdout)
    if argv == ("vm_stat",):
        if exit_code != 0:
            return _unparsed("system.memory_overview", "command_failed")
        return _normalize_vm_stat(stdout)
    if argv == ("scutil", "--nc", "list"):
        if exit_code != 0:
            return _unparsed("system.vpn_status", "command_failed")
        return _normalize_scutil_vpn(stdout)
    if argv == ("ip", "addr"):
        if exit_code != 0:
            return _unparsed("system.vpn_status", "command_failed")
        return _normalize_ip_vpn(stdout)
    if len(argv) == 3 and argv[:2] in {("pgrep", "-l"), ("pgrep", "-fl")}:
        return _normalize_pgrep(stdout, stderr=stderr, exit_code=exit_code, query=argv[2])
    if argv == ("sysctl", "-n", "hw.logicalcpu"):
        if exit_code != 0:
            return _unparsed("system.cpu_overview", "command_failed")
        return _normalize_logical_cpu(stdout, source="sysctl")
    if argv == ("lscpu",):
        if exit_code != 0:
            return _unparsed("system.cpu_overview", "command_failed")
        return _normalize_lscpu(stdout)
    if argv in {("top", "-l", "1", "-n", "0"), ("top", "-b", "-n", "1")}:
        if exit_code != 0:
            return _unparsed("system.cpu_overview", "command_failed")
        return _normalize_top_cpu(stdout)
    return not_applicable()


def sensor_payload(content: dict[str, Any]) -> DiagnosticsStructuredPayload:
    status = ToolParseStatus.PARSED if isinstance(content, dict) else ToolParseStatus.UNPARSED
    return DiagnosticsStructuredPayload(
        structured_content=content if isinstance(content, dict) else None,
        structured_schema="system.sensor_snapshot",
        structured_schema_version=1,
        parse_status=status,
    )


def unavailable_sensor_payload(content: dict[str, Any]) -> DiagnosticsStructuredPayload:
    return sensor_payload(content)


def _parsed(schema: str, content: dict[str, Any]) -> DiagnosticsStructuredPayload:
    return DiagnosticsStructuredPayload(
        structured_content=content,
        structured_schema=schema,
        structured_schema_version=1,
        parse_status=ToolParseStatus.PARSED,
    )


def _unparsed(schema: str, warning: str = "unrecognized_output") -> DiagnosticsStructuredPayload:
    return DiagnosticsStructuredPayload(
        structured_content=None,
        structured_schema=schema,
        structured_schema_version=1,
        parse_status=ToolParseStatus.UNPARSED,
        parse_warnings=(warning,),
    )


def _partial(schema: str, content: dict[str, Any], warning: str) -> DiagnosticsStructuredPayload:
    return DiagnosticsStructuredPayload(
        structured_content=content,
        structured_schema=schema,
        structured_schema_version=1,
        parse_status=ToolParseStatus.PARTIAL,
        parse_warnings=(warning,),
    )


def _normalize_sw_vers(stdout: str, *, platform: str) -> DiagnosticsStructuredPayload:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    product_name = values.get("ProductName")
    version = values.get("ProductVersion")
    if not product_name and not version:
        return _unparsed("system.os_version")
    content = {
        "product_name": product_name or "macOS",
        "version": version,
        "build": values.get("BuildVersion"),
        "platform": platform,
        "source": "sw_vers",
    }
    if product_name and version:
        return _parsed("system.os_version", content)
    return _partial("system.os_version", content, "missing_os_version_field")


def _normalize_pmset_battery(stdout: str) -> DiagnosticsStructuredPayload:
    match = re.search(r"\b(\d{1,3})%;\s*([^;\n]+)", stdout)
    if match is None:
        return _unparsed("system.battery_charge")
    return _parsed(
        "system.battery_charge",
        {
            "percent": int(match.group(1)),
            "state": match.group(2).strip().casefold(),
            "source": "pmset",
        },
    )


def _normalize_upower_battery(stdout: str) -> DiagnosticsStructuredPayload:
    percent_match = re.search(r"percentage:\s*(\d{1,3})%", stdout, flags=re.IGNORECASE)
    state_match = re.search(r"state:\s*([^\n]+)", stdout, flags=re.IGNORECASE)
    if percent_match is None:
        return _unparsed("system.battery_charge")
    return _parsed(
        "system.battery_charge",
        {
            "percent": int(percent_match.group(1)),
            "state": state_match.group(1).strip().casefold() if state_match else None,
            "source": "upower",
        },
    )


def _normalize_df(stdout: str) -> DiagnosticsStructuredPayload:
    rows = [line.split() for line in stdout.splitlines() if line.strip()]
    if len(rows) < 2:
        return _unparsed("system.disk_free")
    filesystems = []
    for row in rows[1:]:
        if len(row) < 6:
            continue
        filesystems.append(
            {
                "filesystem": row[0],
                "mount": row[-1],
                "size": row[1],
                "used": row[2],
                "available": row[3],
                "used_percent": row[4],
            }
        )
    if not filesystems:
        return _unparsed("system.disk_free")
    return _parsed("system.disk_free", {"filesystems": filesystems, "source": "df"})


def _normalize_free(stdout: str) -> DiagnosticsStructuredPayload:
    lines = [line.split() for line in stdout.splitlines() if line.strip()]
    memory: dict[str, Any] = {"source": "free"}
    for index, columns in enumerate(lines):
        if columns and columns[0].casefold().startswith("mem:") and index > 0:
            headers = [header.casefold() for header in lines[index - 1]]
            values = columns[1:] if columns[0].casefold() == "mem:" else columns
            memory.update(_free_columns(headers, values))
        if columns and columns[0].casefold().startswith("swap:") and index > 0:
            values = columns[1:] if columns[0].casefold() == "swap:" else columns
            if len(values) >= 2:
                memory["swap_total"] = _mib(values[0])
                memory["swap_used"] = _mib(values[1])
    if "total" not in memory and "available" not in memory:
        return _unparsed("system.memory_overview")
    return _parsed("system.memory_overview", memory)


def _free_columns(headers: list[str], values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("total", "used", "free", "available"):
        value = _column_value(headers, values, field)
        if value is not None:
            result[field] = _mib(value)
    total = _number(_column_value(headers, values, "total"))
    used = _number(_column_value(headers, values, "used"))
    if total and used is not None:
        result["used_percent"] = round((used / total) * 100, 1)
    return result


def _column_value(headers: list[str], values: list[str], name: str) -> str | None:
    try:
        index = headers.index(name)
    except ValueError:
        return None
    if index >= len(values):
        return None
    return values[index]


def _normalize_vm_stat(stdout: str) -> DiagnosticsStructuredPayload:
    page_size_match = re.search(r"page size of (\d+) bytes", stdout)
    free_pages = _vm_stat_pages(stdout, "Pages free")
    speculative_pages = _vm_stat_pages(stdout, "Pages speculative")
    if page_size_match is None or free_pages is None:
        return _unparsed("system.memory_overview")
    page_size = int(page_size_match.group(1))
    free_bytes = free_pages * page_size
    available_bytes = free_bytes + ((speculative_pages or 0) * page_size)
    content = {
        "free": _format_bytes(free_bytes),
        "available": _format_bytes(available_bytes),
        "source": "vm_stat",
    }
    return _partial("system.memory_overview", content, "total_memory_unavailable")


def _vm_stat_pages(stdout: str, label: str) -> int | None:
    pattern = rf"^{re.escape(label)}:\s+(\d+)\."
    match = re.search(pattern, stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _normalize_scutil_vpn(stdout: str) -> DiagnosticsStructuredPayload:
    connected_lines = [line.strip() for line in stdout.splitlines() if "(connected)" in line.casefold()]
    service = None
    if connected_lines:
        parts = connected_lines[0].split()
        service = next((part for part in parts if part not in {"*", "(Connected)"}), None)
    return _parsed(
        "system.vpn_status",
        {
            "connected": bool(connected_lines),
            "interface_or_service": service,
            "evidence": connected_lines[:3],
            "source": "scutil",
        },
    )


def _normalize_ip_vpn(stdout: str) -> DiagnosticsStructuredPayload:
    connected = False
    evidence: list[str] = []
    for block in re.split(r"\n(?=\d+:\s)", stdout.strip()):
        header = block.splitlines()[0].strip() if block.strip() else ""
        if _linux_vpn_interface_is_up(header):
            connected = True
            evidence.append(header)
    return _parsed(
        "system.vpn_status",
        {
            "connected": connected,
            "interface_or_service": evidence[0].split(":", 2)[1].strip().split("@", 1)[0] if evidence else None,
            "evidence": evidence[:3],
            "source": "ip",
        },
    )


def _linux_vpn_interface_is_up(header: str) -> bool:
    lowered_header = header.casefold()
    name_match = re.match(r"\d+:\s+([^:@\s]+)", header)
    interface_name = name_match.group(1).casefold() if name_match else ""
    if not any(marker in interface_name or marker in lowered_header for marker in ("tun", "tap", "wg", "vpn", "utun")):
        return False
    flags_match = re.search(r"<([^>]+)>", header)
    flags = {flag.strip().casefold() for flag in (flags_match.group(1).split(",") if flags_match else ())}
    return "state up" in lowered_header or "up" in flags


def _normalize_pgrep(
    stdout: str,
    *,
    stderr: str,
    exit_code: int,
    query: str,
) -> DiagnosticsStructuredPayload:
    if exit_code not in {0, 1}:
        return _partial(
            "system.process_name_search",
            {"query": query, "matches": [], "error": stderr.strip() or stdout.strip(), "source": "pgrep"},
            "process_search_failed",
        )
    matches = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        pid, _, name = line.strip().partition(" ")
        if not pid.isdigit() or not name:
            continue
        matches.append({"pid": int(pid), "name": name})
    return _parsed(
        "system.process_name_search",
        {"query": query, "matches": matches, "source": "pgrep"},
    )


def _normalize_logical_cpu(stdout: str, *, source: str) -> DiagnosticsStructuredPayload:
    match = re.search(r"\b(\d+)\b", stdout)
    if match is None:
        return _unparsed("system.cpu_overview")
    return _partial(
        "system.cpu_overview",
        {"logical_cores": int(match.group(1)), "source": source},
        "load_unavailable",
    )


def _normalize_lscpu(stdout: str) -> DiagnosticsStructuredPayload:
    match = re.search(r"^CPU\(s\):\s*(\d+)", stdout, flags=re.MULTILINE)
    if match is None:
        return _unparsed("system.cpu_overview")
    return _partial(
        "system.cpu_overview",
        {"logical_cores": int(match.group(1)), "source": "lscpu"},
        "load_unavailable",
    )


def _normalize_top_cpu(stdout: str) -> DiagnosticsStructuredPayload:
    macos = re.search(
        r"CPU usage:\s*([0-9.]+)% user,\s*([0-9.]+)% sys,\s*([0-9.]+)% idle",
        stdout,
    )
    if macos:
        return _partial(
            "system.cpu_overview",
            {
                "user_percent": float(macos.group(1)),
                "system_percent": float(macos.group(2)),
                "idle_percent": float(macos.group(3)),
                "source": "top",
            },
            "core_count_unavailable",
        )
    linux = re.search(
        r"%Cpu\(s\):\s*([0-9.]+)\s*us,\s*([0-9.]+)\s*sy,.*?([0-9.]+)\s*id",
        stdout,
    )
    if linux:
        return _partial(
            "system.cpu_overview",
            {
                "user_percent": float(linux.group(1)),
                "system_percent": float(linux.group(2)),
                "idle_percent": float(linux.group(3)),
                "source": "top",
            },
            "core_count_unavailable",
        )
    return _unparsed("system.cpu_overview")


def _number(value: str | None) -> float | None:
    if value is None or not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    return float(value)


def _mib(value: str) -> str:
    return f"{value} MiB"


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{value / (1024**2):.0f} MiB"
