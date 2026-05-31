from __future__ import annotations

from typing import Any, TextIO

from assistant_core.cli_app.client import CliUserError, JarvisClient
from assistant_core.cli_app.stream_control import cancel_server_request
from assistant_core.cli_app.terminal_rendering import TerminalColorScheme
from assistant_core.cli_app.utils import _display_text, _required_str


async def handle_approval_prompt(
    *,
    client: JarvisClient,
    stdout: TextIO,
    stdin: TextIO,
    data: dict[str, Any],
    request_id: str | None = None,
    color_scheme: TerminalColorScheme | None = None,
) -> bool:
    approval_id = _required_str(data, "approval_id")
    approval = await client.get_approval(approval_id)
    status = str(approval.get("status") or data.get("status") or "")
    if status == "expired":
        _write_approval_line(stdout, color_scheme=color_scheme, text="approval> expired")
        return False
    capability = _display_text(approval.get("capability") or data.get("capability"))
    summary = _approval_summary(approval, data)
    _write_approval_line(
        stdout,
        color_scheme=color_scheme,
        text=f"approval> {capability} wants to perform {summary}",
    )
    stdout.write(_style_approval(color_scheme, "approve? [y/N] "))
    stdout.flush()
    try:
        answer = stdin.readline()
    except KeyboardInterrupt:
        stdout.write("\n")
        await _cancel_request_from_approval_prompt(
            client=client,
            request_id=request_id,
            stdout=stdout,
        )
        return True
    normalized = answer.strip().lower()
    if normalized in {"y", "yes"}:
        try:
            await client.grant_approval(approval_id)
        except CliUserError as exc:
            if _approval_error_is_expired(exc):
                _write_approval_line(
                    stdout,
                    color_scheme=color_scheme,
                    text="approval> expired",
                )
                return False
            raise
        _write_approval_line(stdout, color_scheme=color_scheme, text="approval> granted")
        return False
    if normalized in {"c", "cancel", "/cancel"}:
        await _cancel_request_from_approval_prompt(
            client=client,
            request_id=request_id,
            stdout=stdout,
        )
        return True
    try:
        await client.deny_approval(approval_id)
    except CliUserError as exc:
        if _approval_error_is_expired(exc):
            _write_approval_line(
                stdout,
                color_scheme=color_scheme,
                text="approval> expired",
            )
            return False
        raise
    _write_approval_line(stdout, color_scheme=color_scheme, text="approval> denied")
    return False


def _approval_summary(approval: dict[str, Any], event_data: dict[str, Any]) -> str:
    if isinstance(event_data.get("redacted_summary"), str):
        return _display_text(event_data["redacted_summary"])
    payload = approval.get("redacted_payload")
    if isinstance(payload, dict) and isinstance(payload.get("summary"), str):
        return _display_text(payload["summary"])
    scope = approval.get("scope")
    if isinstance(scope, dict):
        tool_name = _display_text(scope.get("tool_name"))
        argument_keys = scope.get("argument_keys")
        if isinstance(argument_keys, list):
            return f"{tool_name}({', '.join(str(key) for key in argument_keys)})"
        return tool_name
    return "requested action"


def _approval_error_is_expired(exc: CliUserError) -> bool:
    message = str(exc).lower()
    return "approval_expired" in message or "expired" in message


def _write_approval_line(
    stdout: TextIO,
    *,
    color_scheme: TerminalColorScheme | None,
    text: str,
) -> None:
    stdout.write(f"{_style_approval(color_scheme, text)}\n")


def _style_approval(color_scheme: TerminalColorScheme | None, text: str) -> str:
    if color_scheme is None:
        return text
    return color_scheme.style("approval", text)


async def _cancel_request_from_approval_prompt(
    *,
    client: JarvisClient,
    request_id: str | None,
    stdout: TextIO,
) -> None:
    if request_id is None:
        stdout.write("approval> cancelled\n")
        return
    await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
    stdout.write("approval> cancelled\n")
