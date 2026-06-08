from __future__ import annotations

from enum import StrEnum


class LiveStateEvidenceKind(StrEnum):
    CURRENT_TIMESTAMP = "current_timestamp"
    COUNTDOWN_INTERVAL = "countdown_interval"
    FIXED_TIME_INTERVAL = "fixed_time_interval"
    CALENDAR_INTERVAL = "calendar_interval"
    NUMERIC_TRANSFORM = "numeric_transform"
    SYSTEM_RESOURCES = "system_resources"
    SYSTEM_NETWORK = "system_network"
    SYSTEM_HARDWARE = "system_hardware"
    SYSTEM_SENSORS = "system_sensors"
    SYSTEM_PROCESS = "system_process"
    DAEMON_STATUS = "daemon_status"


_KIND_BY_TOOL_NAME: dict[str, LiveStateEvidenceKind] = {
    "calculator.evaluate": LiveStateEvidenceKind.NUMERIC_TRANSFORM,
    "calendar.diff": LiveStateEvidenceKind.CALENDAR_INTERVAL,
    "daemon.status": LiveStateEvidenceKind.DAEMON_STATUS,
    "datetime.diff": LiveStateEvidenceKind.FIXED_TIME_INTERVAL,
    "datetime.now": LiveStateEvidenceKind.CURRENT_TIMESTAMP,
    "datetime.until": LiveStateEvidenceKind.COUNTDOWN_INTERVAL,
    "tool.system.read.hardware": LiveStateEvidenceKind.SYSTEM_HARDWARE,
    "tool.system.read.network": LiveStateEvidenceKind.SYSTEM_NETWORK,
    "tool.system.read.process": LiveStateEvidenceKind.SYSTEM_PROCESS,
    "tool.system.read.resources": LiveStateEvidenceKind.SYSTEM_RESOURCES,
    "tool.system.read.sensors": LiveStateEvidenceKind.SYSTEM_SENSORS,
}


def evidence_kinds_for_tool_names(
    tool_names: frozenset[str],
) -> frozenset[LiveStateEvidenceKind]:
    return frozenset(
        kind
        for tool_name in tool_names
        if (kind := _KIND_BY_TOOL_NAME.get(tool_name)) is not None
    )


__all__ = ["LiveStateEvidenceKind", "evidence_kinds_for_tool_names"]
