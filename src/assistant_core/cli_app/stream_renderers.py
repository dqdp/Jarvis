from __future__ import annotations

from typing import TextIO

from assistant_core.cli_app.role_rendering import write_role_line
from assistant_core.cli_app.terminal_rendering import TerminalColorScheme
from assistant_core.cli_app.utils import _display_text


def is_tool_stream_event(event_type: str) -> bool:
    return (
        event_type.startswith("tool.call.")
        or event_type.startswith("tool.shell.")
        or event_type.startswith("tool.system.diagnostics.")
    )


def write_tool_stream_event(
    stdout: TextIO,
    *,
    event_type: str,
    data: dict,
    color_scheme: TerminalColorScheme | None = None,
) -> None:
    tool_name = _display_text(data.get("tool_name"))
    if event_type.endswith(".started"):
        _write_tool_started(stdout, data=data, tool_name=tool_name, color_scheme=color_scheme)
        return
    if event_type.endswith(".completed"):
        _write_tool_completed(stdout, data=data, tool_name=tool_name, color_scheme=color_scheme)
        return
    if event_type.endswith(".denied"):
        code = _display_text(data.get("error_code") or data.get("policy_outcome"))
        suffix = f" ({code})" if code else ""
        _write_tool_line(stdout, color_scheme=color_scheme, text=f"tool> denied {tool_name}{suffix}")
        return
    if event_type.endswith(".unavailable"):
        source = _display_text(data.get("source") or data.get("family"))
        suffix = f" {source}" if source else ""
        _write_tool_line(stdout, color_scheme=color_scheme, text=f"tool> unavailable {tool_name}{suffix}")
        return
    _write_tool_line(
        stdout,
        color_scheme=color_scheme,
        text=f"tool> {_display_text(event_type)} {tool_name}",
    )


def write_request_error(
    stdout: TextIO,
    data: dict,
    *,
    color_scheme: TerminalColorScheme | None = None,
) -> None:
    error = data.get("error")
    if not isinstance(error, dict):
        write_role_line(
            stdout,
            color_scheme=color_scheme,
            role="error",
            text=f"error> {_display_text(data)}",
        )
        return
    message = _display_text(error.get("message") or "request failed")
    code = _display_text(error.get("code"))
    suffix = f" ({code})" if code else ""
    write_role_line(
        stdout,
        color_scheme=color_scheme,
        role="error",
        text=f"error> {message}{suffix}",
    )


def _write_tool_started(
    stdout: TextIO,
    *,
    data: dict,
    tool_name: str,
    color_scheme: TerminalColorScheme | None,
) -> None:
    argv = _display_argv(data.get("argv"))
    suffix = f" {argv}" if argv else ""
    _write_tool_line(stdout, color_scheme=color_scheme, text=f"tool> running {tool_name}{suffix}")


def _write_tool_completed(
    stdout: TextIO,
    *,
    data: dict,
    tool_name: str,
    color_scheme: TerminalColorScheme | None,
) -> None:
    details = []
    if data.get("exit_code") is not None:
        details.append(f"exit={_display_text(data.get('exit_code'))}")
    if data.get("output_bytes") is not None:
        details.append(f"bytes={_display_text(data.get('output_bytes'))}")
    suffix = f" {' '.join(details)}" if details else ""
    _write_tool_line(stdout, color_scheme=color_scheme, text=f"tool> completed {tool_name}{suffix}")


def _write_tool_line(
    stdout: TextIO,
    *,
    color_scheme: TerminalColorScheme | None,
    text: str,
) -> None:
    write_role_line(stdout, color_scheme=color_scheme, role="tool", text=text)


def _display_argv(value) -> str:
    if not isinstance(value, list):
        return ""
    return " ".join(_display_text(item) for item in value)


__all__ = ["is_tool_stream_event", "write_request_error", "write_tool_stream_event"]
