from __future__ import annotations

import asyncio
from contextlib import suppress
import select
import sys
from typing import Any, TextIO

from assistant_core.cli_app.client import CliUserError, JarvisClient


CLI_CANCEL_EVENT = "__cli.cancel_command__"
CLI_IGNORED_INPUT_EVENT = "__cli.ignored_input__"
APPROVAL_REQUIRED_EVENT = "approval.required"
TTY_CANCEL_POLL_SECONDS = 0.05


async def stream_with_optional_cancel_command(
    *,
    client: JarvisClient,
    request_id: str,
    stdin: TextIO,
    enabled: bool,
):
    stream = client.stream_request(request_id).__aiter__()
    stream_task = asyncio.create_task(_next_stream_event(stream))
    line_polling_enabled = enabled and _can_poll_tty_line(stdin)
    try:
        while True:
            timeout = TTY_CANCEL_POLL_SECONDS if line_polling_enabled else None
            done, _pending = await asyncio.wait(
                [stream_task],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stream_task not in done:
                line = _read_available_tty_line(stdin)
                if line is None:
                    continue
                if line == "":
                    line_polling_enabled = False
                    continue
                if line.strip().startswith("/cancel"):
                    yield CLI_CANCEL_EVENT, {}
                    return
                yield CLI_IGNORED_INPUT_EVENT, {}
                continue

            event = stream_task.result()
            if event is None:
                return
            yield event
            stream_task = asyncio.create_task(_next_stream_event(stream))
    finally:
        stream_task.cancel()
        with suppress(asyncio.CancelledError):
            await stream_task
        stream_close = getattr(stream, "aclose", None)
        if callable(stream_close):
            with suppress(Exception):
                await stream_close()


async def cancel_server_request(
    *,
    client: JarvisClient,
    request_id: str,
    stdout: TextIO,
) -> None:
    try:
        payload = await client.cancel_request(request_id)
    except CliUserError as exc:
        stdout.write(f"cancelled> local client interrupted; server cancel failed: {exc}\n")
        return
    status = payload.get("status")
    if status == "cancelled":
        stdout.write(f"cancelled> request {request_id}\n")
        return
    stdout.write(f"request> {request_id} {status or 'unchanged'}\n")


def _can_poll_tty_line(stdin: TextIO) -> bool:
    if not bool(getattr(stdin, "isatty", lambda: False)()):
        return False
    try:
        stdin.fileno()
    except Exception:
        return False
    return True


async def _next_stream_event(stream) -> tuple[str, dict[str, Any]] | None:
    try:
        return await anext(stream)
    except StopAsyncIteration:
        return None


def _read_available_tty_line(stdin: TextIO) -> str | None:
    try:
        file_descriptor = stdin.fileno()
    except Exception:
        return None
    try:
        ready, _, _ = select.select([file_descriptor], [], [], 0)
    except (OSError, ValueError):
        return None
    if not ready:
        return None
    return stdin.readline()


__all__ = [
    "CLI_CANCEL_EVENT",
    "CLI_IGNORED_INPUT_EVENT",
    "cancel_server_request",
    "stream_with_optional_cancel_command",
]
