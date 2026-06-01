from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, TextIO

from assistant_core.cli_app.config import SlashCommand


@dataclass(frozen=True)
class SlashCommandDefinition:
    usage: str
    description: str
    command_text: str
    argument_hint: str | None = None


@dataclass(frozen=True)
class SlashCommandCompletion:
    text: str
    display: str
    description: str
    argument_hint: str | None = None


class SlashCommandRegistry:
    def __init__(self, commands: Iterable[SlashCommandDefinition]) -> None:
        self._commands = tuple(commands)

    @classmethod
    def from_commands(cls, commands: Iterable[SlashCommand]) -> SlashCommandRegistry:
        return cls(_definition_from_command(command) for command in commands)

    def matches(self, prefix: str) -> list[SlashCommandDefinition]:
        normalized = prefix.strip()
        if normalized == "/":
            return list(self._commands)
        return [
            command
            for command in self._commands
            if command.usage.startswith(normalized)
            or command.command_text.startswith(normalized)
        ]

    def best_completion(self, prefix: str) -> SlashCommandCompletion | None:
        matches = self.matches(prefix)
        if not matches:
            return None
        command = matches[0]
        return SlashCommandCompletion(
            text=command.command_text + (" " if command.argument_hint else ""),
            display=command.usage,
            description=command.description,
            argument_hint=command.argument_hint,
        )


@dataclass(frozen=True)
class ShellActivityState:
    phase: str
    request_id: str | None = None
    detail: str | None = None

    @classmethod
    def idle(cls) -> ShellActivityState:
        return cls(phase="idle")

    def mark_submitting(self, request_id: str | None = None) -> ShellActivityState:
        return ShellActivityState(phase="submitting", request_id=request_id)

    def apply_stream_event(self, event_type: str, data: dict[str, Any]) -> ShellActivityState:
        phase = _phase_for_event(event_type)
        if phase is None:
            return self
        detail = _activity_detail(event_type, data)
        return ShellActivityState(phase=phase, request_id=self.request_id, detail=detail)

    def render_indicator(self) -> str:
        if self.detail:
            return f"activity={self.phase} {self.detail}"
        return f"activity={self.phase}"


def write_activity_indicator(
    stdout: TextIO,
    activity: ShellActivityState,
    *,
    enabled: bool | None = None,
) -> None:
    should_write = (
        bool(getattr(stdout, "isatty", lambda: False)())
        if enabled is None
        else enabled
    )
    if not should_write:
        return
    stdout.write(f"\n{activity.render_indicator()}\n")
    stdout.flush()


class PromptToolkitLineReader:
    def __init__(
        self,
        *,
        stdin: TextIO,
        stdout: TextIO,
        should_add_history: Callable[[str], bool] | None = None,
        command_registry: SlashCommandRegistry,
        status_provider: Callable[[], str] | None = None,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._should_add_history = should_add_history or (lambda _line: True)
        self._command_registry = command_registry
        self._status_provider = status_provider or (lambda: "")
        self._session = None

    def readline(self, prompt: str) -> str | None:
        try:
            session = self._ensure_session()
        except ImportError:
            return self._fallback_readline(prompt)

        try:
            return session.prompt(prompt)
        except EOFError:
            return None
        except KeyboardInterrupt:
            return ""

    async def read_line(self, prompt: str) -> str | None:
        try:
            session = self._ensure_session()
        except ImportError:
            return self._fallback_readline(prompt)

        try:
            return await session.prompt_async(prompt)
        except EOFError:
            return None
        except KeyboardInterrupt:
            return ""

    def _fallback_readline(self, prompt: str) -> str | None:
        self._stdout.write(prompt)
        self._stdout.flush()
        raw_line = self._stdin.readline()
        if raw_line == "":
            return None
        line = raw_line.rstrip("\n")
        return line if self._should_add_history(line) or not line.strip() else line

    def _ensure_session(self):
        if self._session is not None:
            return self._session

        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import History
        from prompt_toolkit.key_binding import KeyBindings

        registry = self._command_registry
        should_add_history = self._should_add_history

        class SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if not text.lstrip().startswith("/"):
                    return
                for match in registry.matches(text):
                    yield Completion(
                        match.command_text + (" " if match.argument_hint else ""),
                        start_position=-len(text),
                        display=match.usage,
                        display_meta=match.description,
                    )

        class FilteredHistory(History):
            def __init__(self) -> None:
                super().__init__()
                self._strings: list[str] = []

            def load_history_strings(self):
                yield from self._strings

            def store_string(self, string: str) -> None:
                if string.strip() and should_add_history(string):
                    self._strings.append(string)

        self._session = PromptSession(
            completer=SlashCompleter(),
            complete_while_typing=True,
            bottom_toolbar=self._status_provider,
            history=FilteredHistory(),
            key_bindings=_slash_palette_key_bindings(KeyBindings),
            **_prompt_toolkit_stdio_kwargs(self._stdin, self._stdout),
        )
        return self._session


def _compact_model_label(model: str) -> str:
    if model.startswith("hf.co/"):
        _prefix, _separator, tag = model.rpartition(":")
        if tag:
            return tag
    return model

def render_status_line(
    *,
    mode: str,
    readiness: str | None,
    conversation_id: str | None,
    phase: str | None,
    model: str | None,
    context_remaining: str | None = None,
    cwd: str | None,
    width: int = 120,
) -> str:
    parts = [
        f"mode={_clip(mode, 16)}",
        f"status={_clip(readiness or 'unknown', 18)}",
        f"phase={_clip(phase or 'idle', 24)}",
    ]
    if model:
        parts.append(f"model={_clip(_compact_model_label(model), 32)}")
    if context_remaining:
        parts.append(f"ctx={_clip(context_remaining, 8)}")
    if cwd:
        parts.append(f"cwd={_clip(_cwd_scope(cwd), 28)}")
    if conversation_id:
        parts.append(f"conv={_clip(conversation_id, 28)}")
    line = " | ".join(parts)
    if len(line) > width:
        line = "|".join(parts)
    return _fit_width(line, width)


def display_loop_mode(loop_strategy: str | None) -> str:
    return loop_strategy or "auto"


def model_status_summary(payload: dict[str, Any]) -> str | None:
    profile_name = payload.get("default_model_profile")
    if not isinstance(profile_name, str) or not profile_name:
        return None
    profiles = payload.get("model_profiles")
    if not isinstance(profiles, dict):
        return profile_name
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return profile_name
    model = profile.get("model")
    if isinstance(model, str) and model:
        return model
    return profile_name


def model_context_limit(payload: dict[str, Any]) -> int | None:
    profile_name = payload.get("default_model_profile")
    profiles = payload.get("model_profiles")
    if not isinstance(profile_name, str) or not isinstance(profiles, dict):
        return None
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return None
    max_input_tokens = profile.get("max_input_tokens")
    if isinstance(max_input_tokens, int) and max_input_tokens > 0:
        return max_input_tokens
    return None


def context_remaining_summary(
    *,
    token_estimate: int | None,
    max_input_tokens: int | None,
) -> str | None:
    if max_input_tokens is None or max_input_tokens <= 0:
        return None
    if token_estimate is None:
        return None
    remaining = max(0, max_input_tokens - max(0, token_estimate))
    return f"{round((remaining / max_input_tokens) * 100)}%"


def _slash_palette_key_bindings(key_bindings_factory):
    bindings = key_bindings_factory()

    @bindings.add("right")
    def _(event):
        buffer = event.current_buffer
        complete_state = getattr(buffer, "complete_state", None)
        current_completion = getattr(complete_state, "current_completion", None)
        if current_completion is not None:
            buffer.apply_completion(current_completion)
            return
        buffer.cursor_right(count=event.arg)

    @bindings.add("escape")
    def _(event):
        event.current_buffer.cancel_completion()

    @bindings.add("up")
    def _(event):
        buffer = event.current_buffer
        if getattr(buffer, "complete_state", None) is not None:
            buffer.complete_previous()
            return
        buffer.history_backward(count=event.arg)

    @bindings.add("down")
    def _(event):
        buffer = event.current_buffer
        if getattr(buffer, "complete_state", None) is not None:
            buffer.complete_next()
            return
        buffer.history_forward(count=event.arg)

    return bindings


def _definition_from_command(command: SlashCommand) -> SlashCommandDefinition:
    words = command.usage.split()
    command_words: list[str] = []
    argument_words: list[str] = []
    for word in words:
        if _is_argument_word(word):
            argument_words.append(word)
            continue
        if argument_words:
            argument_words.append(word)
            continue
        command_words.append(word)
    return SlashCommandDefinition(
        usage=command.usage,
        description=command.description,
        command_text=" ".join(command_words),
        argument_hint=" ".join(argument_words) or None,
    )


def _is_argument_word(word: str) -> bool:
    return (
        word.isupper()
        or word.startswith("[")
        or word.endswith("]")
        or "|" in word
    )


def _prompt_toolkit_stdio_kwargs(stdin: TextIO, stdout: TextIO) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    try:
        from prompt_toolkit.input.defaults import create_input
        from prompt_toolkit.output.defaults import create_output
    except ImportError:
        return kwargs
    try:
        kwargs["input"] = create_input(stdin=stdin)
    except Exception:
        kwargs.pop("input", None)
    try:
        kwargs["output"] = create_output(stdout=stdout)
    except Exception:
        kwargs.pop("output", None)
    return kwargs


def _phase_for_event(event_type: str) -> str | None:
    if event_type in {"request.loop_selection.started", "loop.selection.started"}:
        return "selecting"
    if event_type == "context.assembly.started":
        return "assembling context"
    if event_type in {"memory.retrieved", "content.retrieved"}:
        return "retrieving context"
    if event_type.startswith("tool.") and event_type.endswith(".started"):
        return "tool running"
    if event_type == "approval.required":
        return "waiting approval"
    if event_type == "token":
        return "streaming"
    if event_type == "request.processing.cancelled":
        return "cancelled"
    if event_type == "request.processing.failed":
        return "failed"
    if event_type == "request.processing.completed":
        return "done"
    return None


def _activity_detail(event_type: str, data: dict[str, Any]) -> str | None:
    if event_type.startswith("tool."):
        tool_name = data.get("tool_name")
        return str(tool_name) if tool_name else None
    if event_type == "approval.required":
        approval_id = data.get("approval_id")
        return str(approval_id) if approval_id else None
    return None


def _cwd_scope(cwd: str) -> str:
    parts = Path(cwd).parts
    if not parts:
        return ""
    redacted = [_redact_path_part(part) for part in parts]
    if "[redacted]" in redacted:
        index = redacted.index("[redacted]")
        return "/".join(redacted[index : index + 2])
    return redacted[-1]


def _redact_path_part(part: str) -> str:
    lowered = part.lower()
    if (
        lowered in {".ssh", ".env"}
        or "secret" in lowered
        or "token" in lowered
        or "password" in lowered
        or "private" in lowered
        or re.search(r"\bsk-[a-z0-9_-]+", lowered)
    ):
        return "[redacted]"
    return part


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "…"


def _fit_width(value: str, width: int) -> str:
    if width <= 0 or len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "…"


__all__ = [
    "PromptToolkitLineReader",
    "ShellActivityState",
    "SlashCommandCompletion",
    "SlashCommandDefinition",
    "SlashCommandRegistry",
    "display_loop_mode",
    "context_remaining_summary",
    "model_context_limit",
    "model_status_summary",
    "render_status_line",
    "write_activity_indicator",
]
