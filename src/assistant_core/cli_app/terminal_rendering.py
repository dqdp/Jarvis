from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import os
import shutil
from typing import TextIO


@dataclass(frozen=True)
class TerminalColorScheme:
    enabled: bool

    def style(self, role: str, text: str) -> str:
        code = _ANSI_ROLE_CODES.get(role)
        if not self.enabled or code is None:
            return text
        return f"\x1b[{code}m{text}\x1b[0m"

    def start(self, role: str) -> str:
        code = _ANSI_ROLE_CODES.get(role)
        if not self.enabled or code is None:
            return ""
        return f"\x1b[{code}m"

    def status_bar(self, text: str) -> str:
        code = "7;36" if self.enabled else "7"
        return f"\x1b[{code}m{text}\x1b[0m"

    @property
    def reset(self) -> str:
        return "\x1b[0m" if self.enabled else ""


_ANSI_ROLE_CODES = {
    "approval": "35",
    "assistant": "32",
    "content": "36",
    "error": "31",
    "memory": "35",
    "session": "36",
    "tool": "33",
    "status": "36",
    "prompt": "36",
    "dim": "2",
}

_SPINNER_FRAMES = ("-", "\\", "|", "/")


def resolve_terminal_color_enabled(
    mode: str,
    *,
    stdout: TextIO,
    plain: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    if plain or mode == "never":
        return False
    if mode == "always":
        return True
    environment = os.environ if env is None else env
    if "NO_COLOR" in environment or environment.get("TERM") == "dumb":
        return False
    return bool(getattr(stdout, "isatty", lambda: False)())


class TerminalStatusBar:
    def __init__(
        self,
        *,
        stdout: TextIO,
        status_provider: Callable[[], str],
        enabled: bool | None = None,
        color_scheme: TerminalColorScheme | None = None,
        spinner_frames: Sequence[str] = _SPINNER_FRAMES,
    ) -> None:
        self._stdout = stdout
        self._status_provider = status_provider
        self._enabled = (
            bool(getattr(stdout, "isatty", lambda: False)())
            if enabled is None
            else enabled
        )
        self._color_scheme = color_scheme or TerminalColorScheme(enabled=False)
        self._spinner_frames = tuple(spinner_frames) or _SPINNER_FRAMES
        self._spinner_index = 0
        self._spinner_frame: str | None = None
        self._active = False
        self._rows = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled or self._active:
            return
        size = _terminal_size()
        self._rows = max(2, size.lines)
        self._spinner_index = 0
        self._spinner_frame = self._spinner_frames[self._spinner_index]
        self._stdout.write(f"\x1b7\x1b[1;{self._rows - 1}r\x1b8")
        self._active = True
        self.render()

    def tick(self) -> None:
        if not self._enabled:
            return
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self._spinner_frame = self._spinner_frames[self._spinner_index]
        self.render()

    def render(self) -> None:
        if not self._enabled:
            return
        size = _terminal_size()
        rows = self._rows or max(2, size.lines)
        columns = max(20, size.columns)
        line = self._status_provider()
        if self._spinner_frame is not None:
            line = f"{self._spinner_frame} {line}"
        line = _fit_width(line, columns)
        self._stdout.write(
            f"\x1b7\x1b[{rows};1H\x1b[2K{self._color_scheme.status_bar(line)}\x1b8"
        )
        self._stdout.flush()

    def stop(self) -> None:
        if not self._enabled or not self._active:
            return
        rows = self._rows or max(2, _terminal_size().lines)
        self._stdout.write(f"\x1b7\x1b[r\x1b[{rows};1H\x1b[2K\x1b8")
        self._stdout.flush()
        self._active = False
        self._rows = 0
        self._spinner_frame = None


class TerminalStatusAnimator:
    def __init__(
        self,
        *,
        status_bar: TerminalStatusBar,
        interval_seconds: float,
    ) -> None:
        self._status_bar = status_bar
        self._interval_seconds = interval_seconds
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if not self._status_bar.enabled:
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval_seconds)
                self._status_bar.tick()
        except asyncio.CancelledError:
            return


def _terminal_size() -> shutil.terminal_size:
    return shutil.get_terminal_size((120, 24))


def _fit_width(value: str, width: int) -> str:
    if width <= 0 or len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    if width <= 3:
        return "." * width
    return value[: width - 3] + "..."


__all__ = [
    "TerminalColorScheme",
    "TerminalStatusAnimator",
    "TerminalStatusBar",
    "resolve_terminal_color_enabled",
]
