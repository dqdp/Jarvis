from __future__ import annotations

import asyncio
from collections.abc import Callable
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from assistant_core.cli_app.approval_flow import handle_approval_prompt
from assistant_core.cli_app.client import CliUserError, JarvisClient
from assistant_core.cli_app.config import DEFAULT_MEMORY_TYPE, LOOP_STRATEGY_CHOICES
from assistant_core.cli_app.line_reader import create_interactive_line_reader
from assistant_core.cli_app.renderers import (
    is_tool_stream_event,
    write_conversation_list,
    write_interactive_help,
    write_memory_list,
    write_model_status,
    write_request_error,
    write_status,
    write_tool_stream_event,
)
from assistant_core.cli_app.shell import (
    ShellActivityState,
    context_remaining_summary,
    display_loop_mode,
    model_context_limit,
    model_status_summary,
    render_status_line,
    write_activity_indicator,
)
from assistant_core.cli_app.stream_control import (
    CLI_CANCEL_EVENT,
    cancel_server_request,
    stream_with_optional_cancel_command,
)
from assistant_core.cli_app.utils import _display_text, _required_str


@dataclass
class ChatShellState:
    conversation_id: str | None
    next_title: str | None
    last_request_id: str | None = None
    loop_strategy: str | None = None


async def run_interactive_chat(
    *,
    client: JarvisClient,
    base_url: str,
    stdin: TextIO,
    stdout: TextIO,
    project_namespace: str,
    sensitivity: str,
    title: str | None,
    loop_strategy: str | None = None,
    working_directory: str | None = None,
    plain: bool = False,
) -> int:
    state = ChatShellState(conversation_id=None, next_title=title, loop_strategy=loop_strategy)
    request_working_directory = working_directory or str(Path.cwd())
    activity_state = ShellActivityState.idle()
    readiness_summary = "unknown"
    model_summary: str | None = None
    max_input_tokens: int | None = None
    context_remaining: str | None = None

    def status_provider() -> str:
        return render_status_line(
            mode=display_loop_mode(state.loop_strategy),
            readiness=readiness_summary,
            conversation_id=state.conversation_id,
            phase=activity_state.phase,
            model=model_summary,
            context_remaining=context_remaining,
            cwd=request_working_directory,
        )

    def mark_request_started(request_id: str) -> None:
        nonlocal activity_state
        state.last_request_id = request_id
        activity_state = activity_state.mark_submitting(request_id)
        write_activity_indicator(stdout, activity_state, enabled=False if plain else None)

    def mark_stream_event(event_type: str, data: dict[str, Any]) -> None:
        nonlocal activity_state, context_remaining
        previous_phase = activity_state.phase
        activity_state = activity_state.apply_stream_event(event_type, data)
        if event_type == "context.assembled":
            context_remaining = context_remaining_summary(
                token_estimate=_optional_int(data.get("token_estimate")),
                max_input_tokens=max_input_tokens,
            )
        if activity_state.phase != previous_phase:
            write_activity_indicator(stdout, activity_state, enabled=False if plain else None)

    if _should_load_toolbar_status(stdin=stdin, stdout=stdout, plain=plain):
        try:
            payload = await client.runtime_status()
        except CliUserError:
            payload = {}
        readiness_summary = _display_text(payload.get("status") or readiness_summary)
        model_summary = model_status_summary(payload)
        max_input_tokens = model_context_limit(payload)
        context_remaining = context_remaining_summary(
            token_estimate=None,
            max_input_tokens=max_input_tokens,
        )

    line_reader = create_interactive_line_reader(
        stdin=stdin,
        stdout=stdout,
        sensitivity=sensitivity,
        plain=plain,
        status_provider=status_provider,
    )

    stdout.write("Jarvis CLI\n")
    stdout.write(f"Connected to {base_url}\n")
    stdout.write("Type / to show commands, /exit to quit.\n\n")
    stdout.write("Use Up/Down for in-session history; history is not saved to disk.\n\n")

    while True:
        raw_line = await line_reader.read_line("jarvis> ")
        if raw_line is None:
            stdout.write("bye\n")
            return 0

        line = raw_line.strip()
        if not line:
            continue
        try:
            if line in {"/exit", "/quit"}:
                stdout.write("bye\n")
                return 0
            if line == "/help":
                write_interactive_help(stdout)
                continue
            if line == "/status":
                stdout.write(f"mode> {display_loop_mode(state.loop_strategy)}\n")
                payload = await write_status(client=client, stdout=stdout)
                readiness_summary = _display_text(payload.get("status") or "unknown")
                continue
            if line == "/mode" or line.startswith("/mode "):
                requested_mode = line.removeprefix("/mode").strip()
                if not requested_mode:
                    stdout.write(f"mode> {display_loop_mode(state.loop_strategy)}\n")
                    continue
                if requested_mode not in LOOP_STRATEGY_CHOICES:
                    stdout.write("usage> /mode auto|chat|tools\n")
                    continue
                state.loop_strategy = None if requested_mode == "auto" else requested_mode
                stdout.write(f"mode> {display_loop_mode(state.loop_strategy)}\n")
                continue
            if line == "/model":
                payload = await write_model_status(client=client, stdout=stdout)
                model_summary = model_status_summary(payload)
                max_input_tokens = model_context_limit(payload)
                context_remaining = context_remaining_summary(
                    token_estimate=None,
                    max_input_tokens=max_input_tokens,
                )
                continue
            if line == "/sessions":
                await write_conversation_list(client=client, stdout=stdout)
                continue
            if line.startswith("/resume"):
                conversation_id = line.removeprefix("/resume").strip()
                if not conversation_id:
                    stdout.write("usage> /resume <conversation_id>\n")
                    continue
                conversation = await client.get_conversation(conversation_id)
                state.conversation_id = _required_str(conversation, "conversation_id")
                state.next_title = None
                state.last_request_id = None
                title_suffix = _display_text(conversation.get("title"))
                stdout.write(f"conversation> resumed {state.conversation_id}")
                if title_suffix:
                    stdout.write(f" {title_suffix}")
                stdout.write("\n")
                continue
            if line.startswith("/new"):
                state.conversation_id = None
                state.next_title = line.removeprefix("/new").strip() or None
                state.last_request_id = None
                stdout.write("conversation> new conversation\n")
                continue
            if line == "/clear":
                state.conversation_id = None
                state.next_title = None
                state.last_request_id = None
                stdout.write("conversation> cleared\n")
                continue
            if line.startswith("/cancel"):
                request_id = line.removeprefix("/cancel").strip() or state.last_request_id
                if request_id is None:
                    stdout.write("cancelled> no request\n")
                    continue
                await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
                state.last_request_id = None
                continue
            if line == "/memory list":
                await write_memory_list(client=client, stdout=stdout)
                continue
            if line.startswith("/memory search"):
                query = line.removeprefix("/memory search").strip()
                if not query:
                    stdout.write("usage> /memory search <query>\n")
                    continue
                await write_memory_list(client=client, stdout=stdout, query=query)
                continue
            if line.startswith("/memory delete"):
                memory_id = line.removeprefix("/memory delete").strip()
                if not memory_id:
                    stdout.write("usage> /memory delete <memory_id>\n")
                    continue
                memory = await client.delete_memory(memory_id)
                stdout.write(
                    "memory> "
                    f"{_required_str(memory, 'memory_id')} "
                    f"{_display_text(memory.get('status'))}\n"
                )
                continue
            if line.startswith("/memory add"):
                content = line.removeprefix("/memory add").strip()
                if not content:
                    stdout.write("usage> /memory add <content>\n")
                    continue
                memory = await client.create_memory(
                    namespace=project_namespace,
                    memory_type=DEFAULT_MEMORY_TYPE,
                    content=content,
                    sensitivity=sensitivity,
                )
                stdout.write(f"memory> {_required_str(memory, 'memory_id')}\n")
                continue
            if line.startswith("/"):
                stdout.write("error> unknown command; type /help\n")
                continue

            if state.conversation_id is None:
                conversation = await client.create_conversation(
                    title=state.next_title,
                    active_project_namespace=project_namespace,
                )
                state.conversation_id = _required_str(conversation, "conversation_id")
                state.next_title = None

            exit_code = await submit_and_stream_message(
                client=client,
                stdout=stdout,
                conversation_id=state.conversation_id,
                content=line,
                sensitivity=sensitivity,
                loop_strategy=state.loop_strategy,
                working_directory=request_working_directory,
                client_message_id=None,
                assistant_prefix="assistant> ",
                stdin=stdin,
                on_request_started=mark_request_started,
                on_stream_event=mark_stream_event,
                allow_tty_cancel_command=not plain,
            )
            state.last_request_id = None
            if exit_code != 0:
                continue
        except CliUserError as exc:
            stdout.write(f"error> {exc}\n")
            continue


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
    stdin: TextIO = sys.stdin,
    on_request_started: Callable[[str], None] | None = None,
    on_stream_event: Callable[[str, dict[str, Any]], None] | None = None,
    allow_tty_cancel_command: bool = False,
) -> int:
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
    if assistant_prefix is not None:
        stdout.write(assistant_prefix)
        stdout.flush()
    try:
        async for event_type, data in stream_with_optional_cancel_command(
            client=client,
            request_id=request_id,
            stdin=stdin,
            enabled=allow_tty_cancel_command,
        ):
            if event_type == CLI_CANCEL_EVENT:
                stdout.write("\n")
                await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
                return 130
            if on_stream_event is not None:
                on_stream_event(event_type, data)
            if event_type == "token":
                stdout.write(data.get("delta", ""))
                stdout.flush()
            elif is_tool_stream_event(event_type):
                if assistant_prefix is not None:
                    stdout.write("\n")
                write_tool_stream_event(stdout, event_type=event_type, data=data)
            elif event_type == "request.processing.failed":
                stdout.write("\n")
                write_request_error(stdout, data)
                return 1
            elif event_type == "request.processing.cancelled":
                stdout.write("\n")
                stdout.write(f"cancelled> request {request_id}\n")
                return 130
            elif event_type == "approval.required":
                if assistant_prefix is not None:
                    stdout.write("\n")
                cancelled = await handle_approval_prompt(
                    client=client,
                    stdout=stdout,
                    stdin=stdin,
                    data=data,
                    request_id=request_id,
                )
                if cancelled:
                    return 130
    except (asyncio.CancelledError, KeyboardInterrupt):
        stdout.write("\n")
        await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
        return 130
    stdout.write("\n")
    return 0


def _should_load_toolbar_status(*, stdin: TextIO, stdout: TextIO, plain: bool) -> bool:
    if plain:
        return False
    return bool(
        getattr(stdin, "isatty", lambda: False)()
        and getattr(stdout, "isatty", lambda: False)()
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None
