from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import sys
from dataclasses import dataclass
from typing import Any, Protocol, TextIO
from uuid import uuid4

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_PROJECT_NAMESPACE = "project.personal_assistant"
DEFAULT_MEMORY_TYPE = "fact"
DEFAULT_SENSITIVITY = "project"
REQUEST_TIMEOUT_SECONDS = 60.0
STREAM_CONNECT_TIMEOUT_SECONDS = 5.0
STREAM_READ_TIMEOUT_SECONDS = 180.0
ARROW_UP = "\x1b[A"
ARROW_DOWN = "\x1b[B"


@dataclass(frozen=True)
class SlashCommand:
    usage: str
    description: str


SLASH_COMMANDS = (
    SlashCommand("/help", "Show this help."),
    SlashCommand("/new [title]", "Start a new conversation."),
    SlashCommand("/memory add TEXT", "Save manual memory."),
    SlashCommand("/memory list", "List manual memories."),
    SlashCommand("/exit", "Quit."),
    SlashCommand("/quit", "Quit."),
)


class JarvisClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def create_conversation(
        self,
        *,
        title: str | None,
        active_project_namespace: str | None,
    ) -> dict[str, Any]: ...

    async def submit_message(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        content: str,
        sensitivity: str,
    ) -> dict[str, Any]: ...

    def stream_request(self, request_id: str): ...

    async def create_memory(
        self,
        *,
        namespace: str,
        memory_type: str,
        content: str,
        sensitivity: str,
    ) -> dict[str, Any]: ...

    async def list_memories(self) -> dict[str, Any]: ...

    async def cancel_request(self, request_id: str) -> dict[str, Any]: ...


class ReadlineModule(Protocol):
    def add_history(self, line: str) -> None: ...


class CliUserError(Exception):
    pass


class HttpJarvisClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def health(self) -> dict[str, Any]:
        return await self._get_json("/v1/health")

    async def create_conversation(
        self,
        *,
        title: str | None,
        active_project_namespace: str | None,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/v1/conversations",
            {
                "title": title,
                "active_project_namespace": active_project_namespace,
                "metadata": {},
            },
        )

    async def submit_message(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        content: str,
        sensitivity: str,
    ) -> dict[str, Any]:
        return await self._post_json(
            f"/v1/conversations/{conversation_id}/messages",
            {
                "client_message_id": client_message_id,
                "content": content,
                "sensitivity": sensitivity,
                "metadata": {},
            },
        )

    async def stream_request(self, request_id: str):
        try:
            timeout = httpx.Timeout(
                connect=STREAM_CONNECT_TIMEOUT_SECONDS,
                read=STREAM_READ_TIMEOUT_SECONDS,
                write=STREAM_CONNECT_TIMEOUT_SECONDS,
                pool=STREAM_CONNECT_TIMEOUT_SECONDS,
            )
            async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout) as client:
                async with client.stream("GET", f"/v1/requests/{request_id}/stream") as response:
                    response.raise_for_status()
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n\n" in buffer:
                            block, buffer = buffer.split("\n\n", 1)
                            for event in parse_sse_blocks(f"{block}\n\n"):
                                yield event
        except CliUserError:
            raise
        except httpx.HTTPStatusError as exc:
            raise CliUserError(_http_error_message(exc, "stream request")) from exc
        except httpx.HTTPError as exc:
            raise CliUserError(f"cannot reach daemon at {self._base_url}: {exc}") from exc
        except ValueError as exc:
            raise CliUserError("invalid streaming response from daemon") from exc

    async def create_memory(
        self,
        *,
        namespace: str,
        memory_type: str,
        content: str,
        sensitivity: str,
    ) -> dict[str, Any]:
        return await self._post_json(
            "/v1/memories",
            {
                "namespace": namespace,
                "memory_type": memory_type,
                "content": content,
                "sensitivity": sensitivity,
                "metadata": {},
            },
        )

    async def list_memories(self) -> dict[str, Any]:
        return await self._get_json("/v1/memories")

    async def cancel_request(self, request_id: str) -> dict[str, Any]:
        return await self._post_json(f"/v1/requests/{request_id}/cancel", {})

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as client:
                response = await client.get(path)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise CliUserError(_http_error_message(exc, path)) from exc
        except httpx.HTTPError as exc:
            raise CliUserError(f"cannot reach daemon at {self._base_url}: {exc}") from exc
        except ValueError as exc:
            raise CliUserError(f"invalid JSON response from daemon for {path}") from exc

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise CliUserError(_http_error_message(exc, path)) from exc
        except httpx.HTTPError as exc:
            raise CliUserError(f"cannot reach daemon at {self._base_url}: {exc}") from exc
        except ValueError as exc:
            raise CliUserError(f"invalid JSON response from daemon for {path}") from exc


async def run(
    argv: list[str] | None = None,
    *,
    client_factory=HttpJarvisClient,
    stdout: TextIO = sys.stdout,
    stdin: TextIO = sys.stdin,
) -> int:
    args = _parser().parse_args(argv)
    client = client_factory(args.base_url)

    try:
        return await _run_command(args=args, client=client, stdout=stdout, stdin=stdin)
    except CliUserError as exc:
        stdout.write(f"error> {exc}\n")
        return 1


async def _run_command(
    *,
    args: argparse.Namespace,
    client: JarvisClient,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if args.command is None:
        return await run_interactive_chat(
            client=client,
            base_url=args.base_url,
            stdin=stdin,
            stdout=stdout,
            project_namespace=DEFAULT_PROJECT_NAMESPACE,
            sensitivity=DEFAULT_SENSITIVITY,
            title=None,
        )

    if args.command == "health":
        stdout.write(json.dumps(await client.health(), ensure_ascii=False, indent=2))
        stdout.write("\n")
        return 0

    if args.command == "chat":
        if not args.message:
            return await run_interactive_chat(
                client=client,
                base_url=args.base_url,
                stdin=stdin,
                stdout=stdout,
                project_namespace=args.project_namespace,
                sensitivity=args.sensitivity,
                title=args.title,
            )
        conversation_id = args.conversation_id
        if conversation_id is None:
            conversation = await client.create_conversation(
                title=args.title,
                active_project_namespace=args.project_namespace,
            )
            conversation_id = _required_str(conversation, "conversation_id")
        return await submit_and_stream_message(
            client=client,
            stdout=stdout,
            conversation_id=conversation_id,
            content=" ".join(args.message),
            sensitivity=args.sensitivity,
            client_message_id=args.client_message_id,
            assistant_prefix=None,
        )

    if args.command == "memory" and args.memory_command == "add":
        memory = await client.create_memory(
            namespace=args.namespace,
            memory_type=args.memory_type,
            content=" ".join(args.content),
            sensitivity=args.sensitivity,
        )
        stdout.write(f"{_required_str(memory, 'memory_id')}\n")
        return 0

    if args.command == "memory" and args.memory_command == "list":
        payload = await client.list_memories()
        for memory in payload.get("memories", []):
            stdout.write(f"{memory['memory_id']} {memory['namespace']} {memory['content']}\n")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


@dataclass
class ChatShellState:
    conversation_id: str | None
    next_title: str | None


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
                raise KeyboardInterrupt
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
) -> InteractiveLineReader | TerminalInteractiveLineReader:
    should_add_history = lambda line: _should_add_interactive_history(
        line,
        sensitivity=sensitivity,
    )
    if _is_tty(stdin, stdout):
        return TerminalInteractiveLineReader(
            stdin=stdin,
            stdout=stdout,
            should_add_history=should_add_history,
        )
    return InteractiveLineReader(
        stdin=stdin,
        stdout=stdout,
        should_add_history=should_add_history,
    )


async def run_interactive_chat(
    *,
    client: JarvisClient,
    base_url: str,
    stdin: TextIO,
    stdout: TextIO,
    project_namespace: str,
    sensitivity: str,
    title: str | None,
) -> int:
    state = ChatShellState(conversation_id=None, next_title=title)
    line_reader = create_interactive_line_reader(
        stdin=stdin,
        stdout=stdout,
        sensitivity=sensitivity,
    )

    stdout.write("Jarvis CLI\n")
    stdout.write(f"Connected to {base_url}\n")
    stdout.write("Type / to show commands, /exit to quit.\n\n")
    stdout.write("Use Up/Down for in-session history; history is not saved to disk.\n\n")

    while True:
        raw_line = line_reader.readline("jarvis> ")
        if raw_line is None:
            stdout.write("bye\n")
            return 0

        line = raw_line.strip()
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            stdout.write("bye\n")
            return 0
        if line == "/help":
            write_interactive_help(stdout)
            continue
        if line.startswith("/new"):
            state.conversation_id = None
            state.next_title = line.removeprefix("/new").strip() or None
            stdout.write("conversation> new conversation\n")
            continue
        if line == "/memory list":
            await write_memory_list(client=client, stdout=stdout)
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
            client_message_id=None,
            assistant_prefix="assistant> ",
        )
        if exit_code != 0:
            return exit_code


def write_interactive_help(stdout: TextIO) -> None:
    longest = max(len(command.usage) for command in SLASH_COMMANDS)
    stdout.write(
        "Commands:\n"
    )
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


async def write_memory_list(*, client: JarvisClient, stdout: TextIO) -> None:
    payload = await client.list_memories()
    for memory in payload.get("memories", []):
        stdout.write(
            f"{memory['memory_id']} {memory.get('namespace', '')} {memory['content']}\n"
        )


async def submit_and_stream_message(
    *,
    client: JarvisClient,
    stdout: TextIO,
    conversation_id: str,
    content: str,
    sensitivity: str,
    client_message_id: str | None,
    assistant_prefix: str | None,
) -> int:
    submitted = await client.submit_message(
        conversation_id=conversation_id,
        client_message_id=client_message_id or str(uuid4()),
        content=content,
        sensitivity=sensitivity,
    )
    request_id = _required_str(submitted, "request_id")
    if assistant_prefix is not None:
        stdout.write(assistant_prefix)
        stdout.flush()
    try:
        async for event_type, data in client.stream_request(request_id):
            if event_type == "token":
                stdout.write(data.get("delta", ""))
                stdout.flush()
            elif event_type == "request.processing.failed":
                stdout.write("\n")
                stdout.write(json.dumps(data.get("error", data), ensure_ascii=False))
                stdout.write("\n")
                return 1
    except (asyncio.CancelledError, KeyboardInterrupt):
        stdout.write("\n")
        await cancel_server_request(client=client, request_id=request_id, stdout=stdout)
        return 130
    stdout.write("\n")
    return 0


async def cancel_server_request(
    *,
    client: JarvisClient,
    request_id: str,
    stdout: TextIO,
) -> None:
    try:
        await client.cancel_request(request_id)
    except Exception as exc:
        stdout.write(f"cancelled> local client interrupted; server cancel failed: {exc}\n")
        return
    stdout.write(f"cancelled> request {request_id}\n")


def parse_sse_blocks(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        if not block:
            continue
        event_type = "message"
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((event_type, json.loads(data)))
    return events


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CliUserError(f"daemon response missing string field: {key}")
    return value


def _http_error_message(exc: httpx.HTTPStatusError, action: str) -> str:
    detail = exc.response.text
    try:
        payload = exc.response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = error["message"]
        elif isinstance(payload.get("detail"), str):
            detail = payload["detail"]
    return f"{action} failed: {exc.response.status_code} {detail}"


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


def main(argv: list[str] | None = None) -> None:
    try:
        exit_code = asyncio.run(run(argv))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health")

    chat = subparsers.add_parser("chat")
    chat.add_argument("message", nargs="*")
    chat.add_argument("--conversation-id")
    chat.add_argument("--client-message-id")
    chat.add_argument("--project-namespace", default=DEFAULT_PROJECT_NAMESPACE)
    chat.add_argument("--sensitivity", default=DEFAULT_SENSITIVITY)
    chat.add_argument("--title")

    memory = subparsers.add_parser("memory")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)
    memory_add = memory_subparsers.add_parser("add")
    memory_add.add_argument("content", nargs="+")
    memory_add.add_argument("--namespace", default=DEFAULT_PROJECT_NAMESPACE)
    memory_add.add_argument("--memory-type", default=DEFAULT_MEMORY_TYPE)
    memory_add.add_argument("--sensitivity", default=DEFAULT_SENSITIVITY)
    memory_subparsers.add_parser("list")

    return parser


if __name__ == "__main__":
    main()
