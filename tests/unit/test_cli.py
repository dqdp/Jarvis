from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any

import httpx
import pytest

from assistant_core import cli


pytestmark = pytest.mark.unit


class FakeCliClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.calls: list[tuple[str, Any]] = []

    async def health(self):
        self.calls.append(("health", None))
        return {"status": "ready"}

    async def create_conversation(self, *, title: str | None, active_project_namespace: str | None):
        self.calls.append(
            (
                "create_conversation",
                {
                    "title": title,
                    "active_project_namespace": active_project_namespace,
                },
            ),
        )
        return {"conversation_id": "conversation-1"}

    async def submit_message(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        content: str,
        sensitivity: str,
    ):
        self.calls.append(
            (
                "submit_message",
                {
                    "conversation_id": conversation_id,
                    "client_message_id": client_message_id,
                    "content": content,
                    "sensitivity": sensitivity,
                },
            ),
        )
        return {"request_id": "request-1"}

    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        yield "token", {"delta": "O"}
        yield "token", {"delta": "K"}
        yield "request.processing.completed", {"assistant_message_id": "message-1"}

    async def create_memory(self, *, namespace: str, memory_type: str, content: str, sensitivity: str):
        self.calls.append(
            (
                "create_memory",
                {
                    "namespace": namespace,
                    "memory_type": memory_type,
                    "content": content,
                    "sensitivity": sensitivity,
                },
            ),
        )
        return {"memory_id": "memory-1"}

    async def list_memories(self):
        self.calls.append(("list_memories", None))
        return {
            "memories": [
                {
                    "memory_id": "memory-1",
                    "namespace": "project.personal_assistant",
                    "content": "saved",
                },
            ],
        }

    async def cancel_request(self, request_id: str):
        self.calls.append(("cancel_request", request_id))
        return {"request_id": request_id, "status": "cancelled"}


class InterruptedStreamCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        raise asyncio.CancelledError
        yield


class FailingHealthCliClient(FakeCliClient):
    async def health(self):
        self.calls.append(("health", None))
        raise cli.CliUserError("daemon unavailable")


class FakeReadline:
    def __init__(self) -> None:
        self.history: list[str] = []

    def add_history(self, line: str) -> None:
        self.history.append(line)


class RemovableFakeReadline(FakeReadline):
    def get_current_history_length(self) -> int:
        return len(self.history)

    def remove_history_item(self, index: int) -> None:
        del self.history[index]


def test_terminal_line_reader_shows_slash_commands_when_slash_is_typed() -> None:
    stdin = StringIO("/help\n")
    stdout = StringIO()
    reader = cli.TerminalInteractiveLineReader(
        stdin=stdin,
        stdout=stdout,
        raw_mode=False,
    )

    assert reader.readline("jarvis> ") == "/help"

    output = stdout.getvalue()
    assert "jarvis> /" in output
    assert "commands>" in output
    assert "/help" in output
    assert "/memory list" in output
    assert output.rstrip().endswith("jarvis> /help")


def test_terminal_line_reader_uses_in_session_arrow_history() -> None:
    stdin = StringIO("first message\n\x1b[A\n")
    stdout = StringIO()
    reader = cli.TerminalInteractiveLineReader(
        stdin=stdin,
        stdout=stdout,
        raw_mode=False,
    )

    assert reader.readline("jarvis> ") == "first message"
    assert reader.readline("jarvis> ") == "first message"


def test_terminal_line_reader_filters_sensitive_history() -> None:
    stdin = StringIO("/memory add private fact\n\x1b[Aplain\n")
    stdout = StringIO()
    reader = cli.TerminalInteractiveLineReader(
        stdin=stdin,
        stdout=stdout,
        raw_mode=False,
        should_add_history=lambda line: not line.startswith("/memory add"),
    )

    assert reader.readline("jarvis> ") == "/memory add private fact"
    assert reader.readline("jarvis> ") == "plain"


def test_cli_chat_creates_conversation_and_streams_tokens() -> None:
    stdout = StringIO()
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            [
                "--base-url",
                "http://testserver",
                "chat",
                "hello",
                "--client-message-id",
                "client-1",
            ],
            client_factory=client_factory,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    assert stdout.getvalue() == "OK\n"
    assert clients[0].calls == [
        (
            "create_conversation",
            {
                "title": None,
                "active_project_namespace": "project.personal_assistant",
            },
        ),
        (
            "submit_message",
            {
                "conversation_id": "conversation-1",
                "client_message_id": "client-1",
                "content": "hello",
                "sensitivity": "project",
            },
        ),
        ("stream_request", "request-1"),
    ]


def test_cli_chat_cancels_server_request_when_stream_is_interrupted() -> None:
    stdout = StringIO()
    clients: list[InterruptedStreamCliClient] = []

    def client_factory(base_url: str) -> InterruptedStreamCliClient:
        client = InterruptedStreamCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat", "looping", "answer"],
            client_factory=client_factory,
            stdout=stdout,
        ),
    )

    assert exit_code == 130
    assert "cancelled> request request-1" in stdout.getvalue()
    assert clients[0].calls[-1] == ("cancel_request", "request-1")


def test_cli_reports_user_errors_without_traceback() -> None:
    stdout = StringIO()

    exit_code = asyncio.run(
        cli.run(
            ["health"],
            client_factory=FailingHealthCliClient,
            stdout=stdout,
        ),
    )

    assert exit_code == 1
    assert stdout.getvalue() == "error> daemon unavailable\n"


def test_interactive_line_reader_uses_readline_history() -> None:
    prompts: list[str] = []
    lines = iter(["hello", "", "/help"])
    readline = FakeReadline()

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return next(lines)

    reader = cli.InteractiveLineReader(
        stdin=StringIO(),
        stdout=StringIO(),
        input_func=input_func,
        readline_module=readline,
    )

    assert reader.readline("jarvis> ") == "hello"
    assert reader.readline("jarvis> ") == ""
    assert reader.readline("jarvis> ") == "/help"
    assert prompts == ["jarvis> ", "jarvis> ", "jarvis> "]
    assert readline.history == ["hello", "/help"]


def test_interactive_line_reader_can_filter_sensitive_history() -> None:
    lines = iter(["normal message", "/memory add private fact"])
    readline = FakeReadline()

    def input_func(_prompt: str) -> str:
        return next(lines)

    reader = cli.InteractiveLineReader(
        stdin=StringIO(),
        stdout=StringIO(),
        input_func=input_func,
        readline_module=readline,
        should_add_history=lambda line: not line.startswith("/memory add"),
    )

    assert reader.readline("jarvis> ") == "normal message"
    assert reader.readline("jarvis> ") == "/memory add private fact"
    assert readline.history == ["normal message"]


def test_interactive_line_reader_removes_input_auto_history_before_filtering() -> None:
    lines = iter(["normal message", "/memory add private fact"])
    readline = RemovableFakeReadline()

    def input_func(_prompt: str) -> str:
        line = next(lines)
        readline.add_history(line)
        return line

    reader = cli.InteractiveLineReader(
        stdin=StringIO(),
        stdout=StringIO(),
        input_func=input_func,
        readline_module=readline,
        should_add_history=lambda line: not line.startswith("/memory add"),
    )

    assert reader.readline("jarvis> ") == "normal message"
    assert reader.readline("jarvis> ") == "/memory add private fact"
    assert readline.history == ["normal message"]


def test_interactive_history_filter_rejects_secret_sessions() -> None:
    assert cli._should_add_interactive_history("normal message", sensitivity="secret") is False
    assert cli._should_add_interactive_history("normal message", sensitivity="project") is True


def test_interactive_line_reader_returns_none_on_eof() -> None:
    def input_func(_prompt: str) -> str:
        raise EOFError

    reader = cli.InteractiveLineReader(
        stdin=StringIO(),
        stdout=StringIO(),
        input_func=input_func,
        readline_module=FakeReadline(),
    )

    assert reader.readline("jarvis> ") is None


def test_cli_without_subcommand_starts_interactive_chat() -> None:
    stdout = StringIO()
    stdin = StringIO("hello\n/exit\n")
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["--base-url", "http://testserver"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "Jarvis CLI" in output
    assert "Type / to show commands" in output
    assert "Use Up/Down for in-session history" in output
    assert "assistant> OK\n" in output
    assert output.rstrip().endswith("bye")
    assert clients[0].calls == [
        (
            "create_conversation",
            {
                "title": None,
                "active_project_namespace": "project.personal_assistant",
            },
        ),
        (
            "submit_message",
            {
                "conversation_id": "conversation-1",
                "client_message_id": clients[0].calls[1][1]["client_message_id"],
                "content": "hello",
                "sensitivity": "project",
            },
        ),
        ("stream_request", "request-1"),
    ]


def test_cli_interactive_slash_commands_manage_session_and_memory() -> None:
    stdout = StringIO()
    stdin = StringIO(
        "/help\n"
        "/memory add remember this\n"
        "/memory list\n"
        "/new next topic\n"
        "hello again\n"
        "/exit\n"
    )
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat", "--project-namespace", "project.demo"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "/new [title]" in output
    assert "memory> memory-1" in output
    assert "memory-1 project.personal_assistant saved" in output
    assert "conversation> new conversation" in output
    assert "assistant> OK\n" in output
    assert clients[0].calls == [
        (
            "create_memory",
            {
                "namespace": "project.demo",
                "memory_type": "fact",
                "content": "remember this",
                "sensitivity": "project",
            },
        ),
        ("list_memories", None),
        (
            "create_conversation",
            {
                "title": "next topic",
                "active_project_namespace": "project.demo",
            },
        ),
        (
            "submit_message",
            {
                "conversation_id": "conversation-1",
                "client_message_id": clients[0].calls[3][1]["client_message_id"],
                "content": "hello again",
                "sensitivity": "project",
            },
        ),
        ("stream_request", "request-1"),
    ]


def test_cli_memory_add_prints_memory_id() -> None:
    stdout = StringIO()

    exit_code = asyncio.run(
        cli.run(
            ["memory", "add", "remember this"],
            client_factory=FakeCliClient,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    assert stdout.getvalue() == "memory-1\n"


def test_parse_sse_blocks() -> None:
    events = cli.parse_sse_blocks(
        'event: token\ndata: {"delta": "A"}\n\n'
        'event: request.processing.completed\ndata: {"request_id": "r"}\n\n',
    )

    assert events == [
        ("token", {"delta": "A"}),
        ("request.processing.completed", {"request_id": "r"}),
    ]


def test_http_error_message_uses_api_error_payload() -> None:
    request = httpx.Request("GET", "http://testserver/v1/health")
    response = httpx.Response(
        503,
        request=request,
        json={"error": {"message": "daemon warming up"}},
    )
    exc = httpx.HTTPStatusError("service unavailable", request=request, response=response)

    assert cli._http_error_message(exc, "health") == "health failed: 503 daemon warming up"


def test_http_client_uses_explicit_non_stream_timeout(monkeypatch) -> None:
    timeouts: list[float | None] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"status": "ready"}

    class RecordingAsyncClient:
        def __init__(self, *, base_url: str, timeout=None) -> None:
            self.base_url = base_url
            timeouts.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, path: str) -> FakeResponse:
            assert path == "/v1/health"
            return FakeResponse()

    monkeypatch.setattr(cli.httpx, "AsyncClient", RecordingAsyncClient)

    response = asyncio.run(cli.HttpJarvisClient("http://testserver").health())

    assert response == {"status": "ready"}
    assert timeouts == [cli.REQUEST_TIMEOUT_SECONDS]
