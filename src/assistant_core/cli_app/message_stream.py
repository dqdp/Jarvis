from __future__ import annotations

import asyncio
from collections.abc import Callable
import sys
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
    stream_with_optional_cancel_command,
)
from assistant_core.cli_app.terminal_rendering import TerminalColorScheme
from assistant_core.cli_app.utils import _required_str


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
    allow_tty_cancel_command: bool = False,
    color_scheme: TerminalColorScheme | None = None,
) -> int:
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

    try:
        async for event_type, data in stream_with_optional_cancel_command(
            client=client,
            request_id=request_id,
            stdin=stdin,
            enabled=allow_tty_cancel_command,
        ):
            if event_type == CLI_CANCEL_EVENT:
                break_assistant_line_if_needed()
                if on_stream_event is not None:
                    on_stream_event("request.processing.cancelled", {})
                await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
                return 130
            if event_type == CLI_IGNORED_INPUT_EVENT:
                break_assistant_line_if_needed()
                stdout.write("input> ignored while request is running; use /cancel to cancel\n")
                continue
            if on_stream_event is not None:
                on_stream_event(event_type, data)
            if event_type == "token":
                write_pending_assistant_prefix()
                stdout.write(data.get("delta", ""))
                stdout.flush()
            elif is_tool_stream_event(event_type):
                break_assistant_line_if_needed()
                write_tool_stream_event(
                    stdout,
                    event_type=event_type,
                    data=data,
                    color_scheme=color_scheme,
                )
            elif event_type == "request.processing.failed":
                break_assistant_line_if_needed()
                write_request_error(stdout, data, color_scheme=color_scheme)
                return 1
            elif event_type == "request.processing.cancelled":
                break_assistant_line_if_needed()
                stdout.write(f"cancelled> request {request_id}\n")
                return 130
            elif event_type == "approval.required":
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
    return 0


__all__ = ["submit_and_stream_message"]
