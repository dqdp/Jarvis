from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, TextIO

from assistant_core.cli_app.client import CliUserError, JarvisClient
from assistant_core.cli_app.config import DEFAULT_MEMORY_TYPE, LOOP_STRATEGY_CHOICES
from assistant_core.cli_app.line_reader import create_interactive_line_reader
from assistant_core.cli_app.message_stream import (
    assistant_character_delay_seconds_for_output,
    submit_and_stream_message,
)
from assistant_core.cli_app.renderers import (
    write_conversation_list,
    write_interactive_help,
    write_memory_list,
    write_model_status,
    write_status,
)
from assistant_core.cli_app.shell import (
    ShellActivityState,
    context_remaining_summary,
    display_loop_mode,
    format_elapsed_seconds,
    model_context_limit,
    model_status_summary,
    render_status_line,
    write_activity_indicator,
)
from assistant_core.cli_app.stream_control import cancel_server_request
from assistant_core.cli_app.terminal_rendering import (
    TerminalColorScheme,
    TerminalInlineStatusLine,
    TerminalStatusAnimator,
    TerminalStatusBar,
    render_status_rule_line,
    resolve_terminal_color_enabled,
)
from assistant_core.cli_app.utils import _display_text, _required_str


@dataclass
class ChatShellState:
    conversation_id: str | None
    next_title: str | None
    last_request_id: str | None = None
    loop_strategy: str | None = None
    interaction_count: int = 0
    current_interaction_started_at: float | None = None


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
    developer_mode: bool = False,
    color_mode: str = "auto",
    status_animation_interval: float = 0.12,
) -> int:
    state = ChatShellState(conversation_id=None, next_title=title, loop_strategy=loop_strategy)
    request_working_directory = working_directory or str(Path.cwd())
    activity_state = ShellActivityState.idle()
    readiness_summary = "unknown"
    model_summary: str | None = None
    max_input_tokens: int | None = None
    context_remaining: str | None = None
    color_scheme = TerminalColorScheme(
        enabled=resolve_terminal_color_enabled(color_mode, stdout=stdout, plain=plain),
    )

    def status_provider() -> str:
        elapsed_seconds: int | None = None
        if (
            state.current_interaction_started_at is not None
            and activity_state.phase != "idle"
        ):
            elapsed_seconds = int(time.monotonic() - state.current_interaction_started_at)
        return render_status_line(
            mode=display_loop_mode(state.loop_strategy),
            readiness=readiness_summary,
            conversation_id=state.conversation_id,
            phase=activity_state.phase,
            model=model_summary,
            context_remaining=context_remaining,
            cwd=request_working_directory,
            interaction_count=state.interaction_count,
            elapsed_seconds=elapsed_seconds,
        )

    def inline_status_provider() -> str:
        if state.current_interaction_started_at is None:
            return "Worked for 0s"
        elapsed = format_elapsed_seconds(time.monotonic() - state.current_interaction_started_at)
        return f"Worked for {elapsed or '0s'}"

    status_bar = TerminalStatusBar(
        stdout=stdout,
        status_provider=status_provider,
        enabled=not plain and bool(getattr(stdout, "isatty", lambda: False)()),
        color_scheme=color_scheme,
        animation_style="none",
    )
    inline_status = TerminalInlineStatusLine(
        stdout=stdout,
        status_provider=inline_status_provider,
        enabled=not plain and bool(getattr(stdout, "isatty", lambda: False)()),
        color_scheme=color_scheme,
    )
    inline_status_animator = TerminalStatusAnimator(
        status_bar=inline_status,
        interval_seconds=status_animation_interval,
    )

    def write_response_summary(summary_line: str) -> None:
        line = render_status_rule_line(summary_line)
        if color_scheme is not None:
            line = color_scheme.style("summary", line)
        stdout.write(f"{line}\n\n")
        stdout.flush()

    def mark_submit_started() -> None:
        nonlocal activity_state
        if activity_state.phase == "submitting":
            return
        state.interaction_count += 1
        state.current_interaction_started_at = time.monotonic()
        activity_state = activity_state.mark_submitting()
        status_bar.start()
        inline_status.start()
        inline_status_animator.start()
        write_activity_indicator(stdout, activity_state, enabled=developer_mode)

    def mark_request_started(request_id: str) -> None:
        nonlocal activity_state
        previous_phase = activity_state.phase
        state.last_request_id = request_id
        activity_state = activity_state.mark_submitting(request_id)
        status_bar.start()
        inline_status.start()
        inline_status_animator.start()
        status_bar.render()
        inline_status.render()
        if previous_phase != "submitting":
            write_activity_indicator(stdout, activity_state, enabled=developer_mode)

    def mark_stream_event(event_type: str, data: dict[str, Any]) -> None:
        nonlocal activity_state, context_remaining
        previous_phase = activity_state.phase
        activity_state = activity_state.apply_stream_event(event_type, data)
        if event_type == "context.assembled":
            context_remaining = context_remaining_summary(
                token_estimate=_optional_int(data.get("token_estimate")),
                max_input_tokens=max_input_tokens,
            )
        phase_changed = activity_state.phase != previous_phase
        context_changed = event_type == "context.assembled"
        if (phase_changed or context_changed) and event_type != "request.processing.completed":
            status_bar.render()
            inline_status.render()
        if phase_changed:
            write_activity_indicator(stdout, activity_state, enabled=developer_mode)

    tty_shell_enabled = _should_load_toolbar_status(stdin=stdin, stdout=stdout, plain=plain)

    if tty_shell_enabled:
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
        user_input_style=color_scheme.prompt_toolkit_style("user") if tty_shell_enabled else None,
    )
    user_prompt = "you> " if tty_shell_enabled else ""

    stdout.write("Jarvis CLI\n")
    stdout.write(f"Connected to {base_url}\n")
    stdout.write("Type / to show commands, /exit to quit.\n\n")
    stdout.write("Use Up/Down for in-session history; history is not saved to disk.\n\n")

    while True:
        raw_line = await line_reader.read_line(user_prompt)
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
                payload = await write_status(
                    client=client,
                    stdout=stdout,
                    color_scheme=color_scheme,
                )
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
                payload = await write_model_status(
                    client=client,
                    stdout=stdout,
                    color_scheme=color_scheme,
                )
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

            try:
                exit_code = await submit_and_stream_message(
                    client=client,
                    stdout=stdout,
                    conversation_id=state.conversation_id,
                    content=line,
                    sensitivity=sensitivity,
                    loop_strategy=state.loop_strategy,
                    working_directory=request_working_directory,
                    client_message_id=None,
                    assistant_prefix=color_scheme.start("assistant"),
                    assistant_suffix=color_scheme.reset,
                    stdin=stdin,
                    on_submit_started=mark_submit_started,
                    on_request_started=mark_request_started,
                    on_stream_event=mark_stream_event,
                    on_transcript_output_started=inline_status.finish_active_line,
                    on_response_summary=write_response_summary,
                    assistant_character_delay_seconds=assistant_character_delay_seconds_for_output(
                        stdout=stdout,
                        plain=plain,
                    ),
                    allow_tty_cancel_command=not plain,
                    color_scheme=color_scheme,
                )
            finally:
                await inline_status_animator.stop()
                status_bar.stop()
                inline_status.stop()
                activity_state = ShellActivityState.idle()
                state.current_interaction_started_at = None
            state.last_request_id = None
            if exit_code != 0:
                continue
        except CliUserError as exc:
            stdout.write(f"error> {exc}\n")
            continue


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
