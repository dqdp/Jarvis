from __future__ import annotations

from typing import TextIO

from assistant_core.cli_app.config import SLASH_COMMANDS


def write_interactive_help(stdout: TextIO) -> None:
    longest = max(len(command.usage) for command in SLASH_COMMANDS)
    stdout.write("Commands:\n")
    for command in SLASH_COMMANDS:
        stdout.write(f"  {command.usage:<{longest}}  {command.description}\n")
    stdout.write(
        "\n"
        "Keys:\n"
        "  Up/Down           Browse in-session input history on Unix TTY.\n"
    )


def write_slash_command_menu(stdout: TextIO, *, prefix: str) -> None:
    matching_commands = [
        command
        for command in SLASH_COMMANDS
        if command.usage.startswith(prefix) or prefix == "/"
    ]
    if not matching_commands:
        return
    longest = max(len(command.usage) for command in matching_commands)
    stdout.write("commands>\n")
    for command in matching_commands:
        stdout.write(f"  {command.usage:<{longest}}  {command.description}\n")


__all__ = ["write_interactive_help", "write_slash_command_menu"]
