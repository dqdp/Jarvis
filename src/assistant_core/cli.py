from __future__ import annotations

from assistant_core.cli_app.approval_flow import handle_approval_prompt
from assistant_core.cli_app.chat_flow import ChatShellState
from assistant_core.cli_app.client import (
    CliUserError,
    HttpJarvisClient,
    JarvisClient,
    _http_error_message,
)
from assistant_core.cli_app.commands import _parser, main, run
from assistant_core.cli_app.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MEMORY_TYPE,
    DEFAULT_PROJECT_NAMESPACE,
    DEFAULT_SENSITIVITY,
    LOOP_STRATEGY_CHOICES,
    REQUEST_TIMEOUT_SECONDS,
    SLASH_COMMANDS,
    STREAM_CONNECT_TIMEOUT_SECONDS,
    STREAM_READ_TIMEOUT_SECONDS,
    SlashCommand,
)
from assistant_core.cli_app.line_reader import (
    InteractiveLineReader,
    ReadlineModule,
    TerminalInteractiveLineReader,
    _is_tty,
    _readline_history_length,
    _should_add_interactive_history,
    _terminal_input_mode,
    _trim_readline_history,
    create_interactive_line_reader,
)
from assistant_core.cli_app.message_stream import submit_and_stream_message
from assistant_core.cli_app.renderers import (
    write_content_ingest,
    write_content_reindex,
    write_content_sources,
    write_content_status,
    write_conversation_list,
    write_interactive_help,
    write_memory_list,
    write_model_status,
    write_slash_command_menu,
    write_status,
)
from assistant_core.cli_app.shell import (
    PromptToolkitLineReader,
    ShellActivityState,
    SlashCommandCompletion,
    SlashCommandDefinition,
    SlashCommandRegistry,
    context_remaining_summary,
    format_elapsed_seconds,
    model_context_limit,
    model_status_summary,
    render_status_line,
    write_activity_indicator,
)
from assistant_core.cli_app.sse import parse_sse_blocks
from assistant_core.cli_app.stream_control import cancel_server_request
from assistant_core.cli_app.terminal_rendering import (
    TerminalColorScheme,
    TerminalStatusAnimator,
    TerminalStatusBar,
    resolve_terminal_color_enabled,
)
from assistant_core.cli_app.utils import _display_text, _required_str


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MEMORY_TYPE",
    "DEFAULT_PROJECT_NAMESPACE",
    "DEFAULT_SENSITIVITY",
    "LOOP_STRATEGY_CHOICES",
    "REQUEST_TIMEOUT_SECONDS",
    "SLASH_COMMANDS",
    "STREAM_CONNECT_TIMEOUT_SECONDS",
    "STREAM_READ_TIMEOUT_SECONDS",
    "ChatShellState",
    "CliUserError",
    "HttpJarvisClient",
    "InteractiveLineReader",
    "JarvisClient",
    "PromptToolkitLineReader",
    "ReadlineModule",
    "ShellActivityState",
    "SlashCommand",
    "SlashCommandCompletion",
    "SlashCommandDefinition",
    "SlashCommandRegistry",
    "TerminalInteractiveLineReader",
    "TerminalColorScheme",
    "TerminalStatusAnimator",
    "TerminalStatusBar",
    "context_remaining_summary",
    "format_elapsed_seconds",
    "_display_text",
    "_http_error_message",
    "_is_tty",
    "_parser",
    "_readline_history_length",
    "_required_str",
    "_should_add_interactive_history",
    "_terminal_input_mode",
    "_trim_readline_history",
    "cancel_server_request",
    "create_interactive_line_reader",
    "handle_approval_prompt",
    "main",
    "model_context_limit",
    "model_status_summary",
    "parse_sse_blocks",
    "render_status_line",
    "resolve_terminal_color_enabled",
    "run",
    "submit_and_stream_message",
    "write_activity_indicator",
    "write_content_ingest",
    "write_content_reindex",
    "write_content_sources",
    "write_content_status",
    "write_conversation_list",
    "write_interactive_help",
    "write_memory_list",
    "write_model_status",
    "write_slash_command_menu",
    "write_status",
]


if __name__ == "__main__":
    main()
