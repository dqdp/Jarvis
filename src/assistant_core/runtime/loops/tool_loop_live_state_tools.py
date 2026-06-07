from __future__ import annotations


KNOWN_LIVE_STATE_TOOL_NAMES = frozenset(
    {
        "calendar.diff",
        "datetime.diff",
        "datetime.now",
        "datetime.until",
        "daemon.status",
        "tool.system.read.hardware",
        "tool.system.read.network",
        "tool.system.read.process",
        "tool.system.read.resources",
        "tool.system.read.sensors",
    }
)


def is_known_live_state_tool_name(tool_name: str) -> bool:
    return tool_name in KNOWN_LIVE_STATE_TOOL_NAMES


__all__ = ["KNOWN_LIVE_STATE_TOOL_NAMES", "is_known_live_state_tool_name"]
