from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol, TextIO

from assistant_core.cli_app.config import ARROW_DOWN, ARROW_UP, DEFAULT_SENSITIVITY, SLASH_COMMANDS
from assistant_core.cli_app.renderers import write_slash_command_menu


class ReadlineModule(Protocol):
    def add_history(self, line: str) -> None: ...


class InteractiveLineReader:
    def __init__(
        self,
        *,
        stdin: TextIO,
        stdout: TextIO,
        input_func: Callable[[str], str] | None = None,
        readline_module: ReadlineModule | None = None,
        should_add_history: Callable[[str], bool] | None = None,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._input_func = input_func
        self._readline = readline_module
        self._should_add_history = should_add_history or (lambda _line: True)

    def readline(self, prompt: str) -> str | None:
        history_length = _readline_history_length(self._readline)
        try:
            if self._input_func is not None:
                line = self._input_func(prompt)
            else:
                self._stdout.write(prompt)
                self._stdout.flush()
                raw_line = self._stdin.readline()
                if raw_line == "":
                    return None
                line = raw_line.rstrip("\n")
        except EOFError:
            return None

        if self._readline is not None:
            _trim_readline_history(self._readline, history_length)
            if line.strip() and self._should_add_history(line):
                self._readline.add_history(line)
        return line


class TerminalInteractiveLineReader:
    def __init__(
        self,
        *,
        stdin: TextIO,
        stdout: TextIO,
        should_add_history: Callable[[str], bool] | None = None,
        raw_mode: bool = True,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._should_add_history = should_add_history or (lambda _line: True)
        self._history: list[str] = []
        self._raw_mode = raw_mode

    def readline(self, prompt: str) -> str | None:
        with _terminal_input_mode(self._stdin, enabled=self._raw_mode):
            return self._readline(prompt)

    def _readline(self, prompt: str) -> str | None:
        buffer = ""
        draft = ""
        history_index = len(self._history)
        slash_menu_shown = False

        self._stdout.write(prompt)
        self._stdout.flush()

        while True:
            char = self._stdin.read(1)
            if char == "":
                return buffer if buffer else None
            if char in {"\n", "\r"}:
                self._stdout.write("\n")
                self._stdout.flush()
                self._add_history(buffer)
                return buffer
            if char == "\x03":
                buffer = ""
                draft = ""
                history_index = len(self._history)
                self._stdout.write("^C\n")
                self._stdout.flush()
                return ""
            if char == "\x04":
                if not buffer:
                    self._stdout.write("\n")
                    self._stdout.flush()
                    return None
                continue
            if char in {"\x7f", "\b"}:
                if buffer:
                    buffer = buffer[:-1]
                    self._redraw(prompt, buffer)
                continue
            if char == "\x1b":
                sequence = char + self._stdin.read(2)
                if sequence == ARROW_UP:
                    if self._history and history_index > 0:
                        if history_index == len(self._history):
                            draft = buffer
                        history_index -= 1
                        buffer = self._history[history_index]
                        self._redraw(prompt, buffer)
                    continue
                if sequence == ARROW_DOWN:
                    if history_index < len(self._history):
                        history_index += 1
                        buffer = draft if history_index == len(self._history) else self._history[history_index]
                        self._redraw(prompt, buffer)
                    continue
                continue

            if history_index != len(self._history):
                history_index = len(self._history)
                draft = ""
            buffer += char
            self._stdout.write(char)
            self._stdout.flush()
            if buffer == "/" and not slash_menu_shown:
                slash_menu_shown = True
                self._stdout.write("\n")
                write_slash_command_menu(self._stdout, prefix=buffer)
                self._redraw(prompt, buffer)

    def _redraw(self, prompt: str, text: str) -> None:
        self._stdout.write(f"\r\x1b[2K{prompt}{text}")
        self._stdout.flush()

    def _add_history(self, line: str) -> None:
        if line.strip() and self._should_add_history(line):
            self._history.append(line)


def create_interactive_line_reader(
    *,
    stdin: TextIO,
    stdout: TextIO,
    sensitivity: str = DEFAULT_SENSITIVITY,
    plain: bool = False,
    status_provider: Callable[[], str] | None = None,
) -> InteractiveLineReader | TerminalInteractiveLineReader:
    should_add_history = lambda line: _should_add_interactive_history(
        line,
        sensitivity=sensitivity,
    )
    if _is_tty(stdin, stdout) and not plain:
        from assistant_core.cli_app.shell import PromptToolkitLineReader, SlashCommandRegistry

        return PromptToolkitLineReader(
            stdin=stdin,
            stdout=stdout,
            should_add_history=should_add_history,
            command_registry=SlashCommandRegistry.from_commands(SLASH_COMMANDS),
            status_provider=status_provider,
        )
    return InteractiveLineReader(
        stdin=stdin,
        stdout=stdout,
        should_add_history=should_add_history,
    )


def _should_add_interactive_history(line: str, *, sensitivity: str) -> bool:
    if sensitivity == "secret":
        return False
    return not line.lstrip().startswith("/memory add")


def _readline_history_length(readline_module: ReadlineModule | None) -> int | None:
    if readline_module is None:
        return None
    get_length = getattr(readline_module, "get_current_history_length", None)
    if not callable(get_length):
        return None
    try:
        value = get_length()
    except Exception:
        return None
    return value if isinstance(value, int) and value >= 0 else None


def _trim_readline_history(readline_module: ReadlineModule, target_length: int | None) -> None:
    if target_length is None:
        return
    get_length = getattr(readline_module, "get_current_history_length", None)
    remove_item = getattr(readline_module, "remove_history_item", None)
    if not callable(get_length) or not callable(remove_item):
        return

    while True:
        try:
            current_length = get_length()
        except Exception:
            return
        if not isinstance(current_length, int) or current_length <= target_length:
            return
        try:
            remove_item(current_length - 1)
        except Exception:
            return


@contextmanager
def _terminal_input_mode(stdin: TextIO, *, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    try:
        file_descriptor = stdin.fileno()
    except Exception:
        yield
        return

    try:
        import termios
        import tty
    except ImportError:
        yield
        return

    original_attrs = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        yield
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_attrs)


def _is_tty(stdin: TextIO, stdout: TextIO) -> bool:
    return bool(
        getattr(stdin, "isatty", lambda: False)()
        and getattr(stdout, "isatty", lambda: False)()
    )
