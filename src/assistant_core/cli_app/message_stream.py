from __future__ import annotations

import asyncio
from asyncio import sleep as async_sleep
from collections.abc import Callable
from dataclasses import dataclass, field
import sys
from time import monotonic
from typing import Any, TextIO
from uuid import uuid4

from assistant_core.cli_app.approval_flow import handle_approval_prompt
from assistant_core.cli_app.client import JarvisClient
from assistant_core.cli_app.renderers import (
    is_tool_stream_event,
    write_request_error,
    write_tool_stream_event,
)
from assistant_core.cli_app.stream_control import (
    CLI_CANCEL_EVENT,
    CLI_IGNORED_INPUT_EVENT,
    cancel_server_request,
    is_cancel_command,
    poll_tty_line,
    stream_with_optional_cancel_command,
)
from assistant_core.cli_app.terminal_rendering import TerminalColorScheme, render_status_rule_line
from assistant_core.cli_app.utils import _display_text, _required_str


DEFAULT_ASSISTANT_CHARACTER_DELAY_SECONDS = 0.006


def assistant_character_delay_seconds_for_output(*, stdout: TextIO, plain: bool) -> float:
    if plain:
        return 0.0
    if not bool(getattr(stdout, "isatty", lambda: False)()):
        return 0.0
    return DEFAULT_ASSISTANT_CHARACTER_DELAY_SECONDS


@dataclass
class _ResponseStreamSummary:
    started_at: float
    resources: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def observe(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "token":
            self._add_resource("model")
            return
        resource = _resource_label(event_type)
        if resource is not None:
            self._add_resource(resource)
        if event_type.startswith("tool."):
            tool_name = data.get("tool_name")
            if isinstance(tool_name, str) and tool_name.strip():
                self._add_tool(_display_text(tool_name))

    def render(self, *, finished_at: float) -> str:
        elapsed = _format_elapsed_seconds(finished_at - self.started_at)
        resources = _join_summary_items(self.resources)
        tools = _join_summary_items(self.tools)
        return f"summary: response_time={elapsed} | resources={resources} | tools={tools}"

    def _add_resource(self, resource: str) -> None:
        if resource not in self.resources:
            self.resources.append(resource)

    def _add_tool(self, tool_name: str) -> None:
        if tool_name not in self.tools:
            self.tools.append(tool_name)


def _resource_label(event_type: str) -> str | None:
    if event_type in {"context.assembly.started", "context.assembled"}:
        return "context"
    if event_type == "memory.retrieved":
        return "memory"
    if event_type == "content.retrieved":
        return "content"
    if event_type.startswith("model."):
        return "model"
    return None


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    seconds = max(0, int(elapsed_seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m{remaining_seconds:02d}s"


def _join_summary_items(items: list[str]) -> str:
    return ",".join(items) if items else "none"


def _write_response_summary(
    stdout: TextIO,
    summary_line: str,
    *,
    color_scheme: TerminalColorScheme | None,
) -> None:
    line = render_status_rule_line(summary_line)
    if color_scheme is not None:
        line = color_scheme.style("summary", line)
    stdout.write(f"{line}\n\n")


async def _write_assistant_delta(
    stdout: TextIO,
    delta: str,
    *,
    character_delay_seconds: float,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    if not delta:
        return False
    if character_delay_seconds <= 0:
        stdout.write(delta)
        stdout.flush()
        return False
    for character in delta:
        if should_cancel is not None and should_cancel():
            return True
        stdout.write(character)
        stdout.flush()
        await async_sleep(character_delay_seconds)
    return False


def _assistant_delta_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _display_text(value)


async def submit_and_stream_message(
    *,
    client: JarvisClient,
    stdout: TextIO,
    conversation_id: str,
    content: str,
    sensitivity: str,
    loop_strategy: str | None = None,
    working_directory: str | None = None,
    client_message_id: str | None,
    assistant_prefix: str | None,
    assistant_suffix: str = "",
    stdin: TextIO = sys.stdin,
    on_submit_started: Callable[[], None] | None = None,
    on_request_started: Callable[[str], None] | None = None,
    on_stream_event: Callable[[str, dict[str, Any]], None] | None = None,
    on_transcript_output_started: Callable[[], None] | None = None,
    on_response_summary: Callable[[str], None] | None = None,
    assistant_character_delay_seconds: float = 0.0,
    allow_tty_cancel_command: bool = False,
    color_scheme: TerminalColorScheme | None = None,
) -> int:
    summary = _ResponseStreamSummary(started_at=monotonic())
    if on_submit_started is not None:
        on_submit_started()
    submitted = await client.submit_message(
        conversation_id=conversation_id,
        client_message_id=client_message_id or str(uuid4()),
        content=content,
        sensitivity=sensitivity,
        loop_strategy=loop_strategy,
        working_directory=working_directory,
    )
    request_id = _required_str(submitted, "request_id")
    if on_request_started is not None:
        on_request_started(request_id)
    assistant_prefix_pending = assistant_prefix is not None
    assistant_line_open = False
    transcript_output_started = False

    def begin_transcript_output() -> None:
        nonlocal transcript_output_started
        if transcript_output_started:
            return
        transcript_output_started = True
        if on_transcript_output_started is not None:
            on_transcript_output_started()

    def write_pending_assistant_prefix() -> None:
        nonlocal assistant_prefix_pending, assistant_line_open
        if not assistant_prefix_pending or assistant_prefix is None:
            return
        stdout.write(assistant_prefix)
        stdout.flush()
        assistant_prefix_pending = False
        assistant_line_open = True

    def break_assistant_line_if_needed() -> None:
        nonlocal assistant_line_open
        if assistant_line_open:
            stdout.write(f"{assistant_suffix}\n")
            assistant_line_open = False

    def write_response_summary() -> None:
        summary_line = summary.render(finished_at=monotonic())
        if on_response_summary is not None:
            on_response_summary(summary_line)
        else:
            _write_response_summary(stdout, summary_line, color_scheme=color_scheme)

    def poll_typewriter_cancel() -> bool:
        nonlocal assistant_line_open
        line = poll_tty_line(stdin, enabled=allow_tty_cancel_command)
        if line is None or line == "":
            return False
        if is_cancel_command(line):
            return True
        break_assistant_line_if_needed()
        stdout.write("input> ignored while request is running; use /cancel to cancel\n")
        assistant_line_open = True
        return False

    try:
        async for event_type, data in stream_with_optional_cancel_command(
            client=client,
            request_id=request_id,
            stdin=stdin,
            enabled=allow_tty_cancel_command,
        ):
            if event_type == CLI_CANCEL_EVENT:
                begin_transcript_output()
                break_assistant_line_if_needed()
                if on_stream_event is not None:
                    on_stream_event("request.processing.cancelled", {})
                await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
                return 130
            if event_type == CLI_IGNORED_INPUT_EVENT:
                begin_transcript_output()
                break_assistant_line_if_needed()
                stdout.write("input> ignored while request is running; use /cancel to cancel\n")
                continue
            if on_stream_event is not None:
                on_stream_event(event_type, data)
            summary.observe(event_type, data)
            if event_type == "token":
                begin_transcript_output()
                write_pending_assistant_prefix()
                delta_text = _assistant_delta_text(data.get("delta", ""))
                if delta_text:
                    assistant_line_open = True
                cancelled_during_delta = await _write_assistant_delta(
                    stdout,
                    delta_text,
                    character_delay_seconds=assistant_character_delay_seconds,
                    should_cancel=poll_typewriter_cancel,
                )
                if cancelled_during_delta:
                    if assistant_line_open:
                        break_assistant_line_if_needed()
                    else:
                        stdout.write("\n")
                    if on_stream_event is not None:
                        on_stream_event("request.processing.cancelled", {})
                    await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
                    return 130
            elif is_tool_stream_event(event_type):
                begin_transcript_output()
                break_assistant_line_if_needed()
                write_tool_stream_event(
                    stdout,
                    event_type=event_type,
                    data=data,
                    color_scheme=color_scheme,
                )
            elif event_type == "request.processing.failed":
                begin_transcript_output()
                break_assistant_line_if_needed()
                write_request_error(stdout, data, color_scheme=color_scheme)
                write_response_summary()
                return 1
            elif event_type == "request.processing.cancelled":
                begin_transcript_output()
                break_assistant_line_if_needed()
                stdout.write(f"cancelled> request {request_id}\n")
                return 130
            elif event_type == "approval.required":
                begin_transcript_output()
                break_assistant_line_if_needed()
                cancelled = await handle_approval_prompt(
                    client=client,
                    stdout=stdout,
                    stdin=stdin,
                    data=data,
                    request_id=request_id,
                    color_scheme=color_scheme,
                )
                if cancelled:
                    if on_stream_event is not None:
                        on_stream_event("request.processing.cancelled", {})
                    return 130
    except (asyncio.CancelledError, KeyboardInterrupt):
        line_was_open = assistant_line_open
        break_assistant_line_if_needed()
        if not line_was_open:
            stdout.write("\n")
        if on_stream_event is not None:
            on_stream_event("request.processing.cancelled", {})
        await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
        return 130
    if assistant_line_open:
        stdout.write(assistant_suffix)
    stdout.write("\n")
    write_response_summary()
    return 0


__all__ = ["assistant_character_delay_seconds_for_output", "submit_and_stream_message"]
