from __future__ import annotations

from typing import TextIO

from assistant_core.cli_app.terminal_rendering import TerminalColorScheme


def write_role_line(
    stdout: TextIO,
    *,
    color_scheme: TerminalColorScheme | None,
    role: str,
    text: str,
) -> None:
    if color_scheme is not None:
        text = color_scheme.style(role, text)
    stdout.write(f"{text}\n")


__all__ = ["write_role_line"]
