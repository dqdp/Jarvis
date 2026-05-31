from __future__ import annotations

import asyncio
from io import StringIO
import os
from pathlib import Path
import sys
import threading
import types
from typing import Any

import httpx
import pytest

from assistant_core import cli
from assistant_core.cli_app import client as cli_client_module
from assistant_core.cli_app import stream_control


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

    async def list_conversations(self, *, limit: int = 20):
        self.calls.append(("list_conversations", {"limit": limit}))
        return {
            "conversations": [
                {
                    "conversation_id": "conversation-1",
                    "title": "First session",
                    "active_project_namespace": "project.personal_assistant",
                    "status": "active",
                    "updated_at": "2026-05-29T10:00:00Z",
                },
            ],
        }

    async def get_conversation(self, conversation_id: str):
        self.calls.append(("get_conversation", conversation_id))
        return {
            "conversation_id": conversation_id,
            "title": "Resumed session",
            "active_project_namespace": "project.personal_assistant",
            "status": "active",
            "updated_at": "2026-05-29T10:00:00Z",
        }

    async def submit_message(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        content: str,
        sensitivity: str,
        loop_strategy: str | None = None,
        working_directory: str | None = None,
    ):
        payload = {
            "conversation_id": conversation_id,
            "client_message_id": client_message_id,
            "content": content,
            "sensitivity": sensitivity,
        }
        if loop_strategy is not None:
            payload["loop_strategy"] = loop_strategy
        if working_directory is not None:
            payload["working_directory"] = working_directory
        self.calls.append(
            (
                "submit_message",
                payload,
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
        self.calls.append(("list_memories", {"query": None}))
        return {
            "memories": [
                {
                    "memory_id": "memory-1",
                    "namespace": "project.personal_assistant",
                    "memory_type": "fact",
                    "sensitivity": "project",
                    "status": "active",
                    "indexing_status": "indexed",
                    "content": "saved",
                },
            ],
        }

    async def search_memories(self, query: str):
        self.calls.append(("list_memories", {"query": query}))
        return {
            "memories": [
                {
                    "memory_id": "memory-2",
                    "namespace": "project.personal_assistant",
                    "memory_type": "fact",
                    "sensitivity": "project",
                    "status": "active",
                    "indexing_status": "indexed",
                    "content": "search hit",
                },
            ],
        }

    async def delete_memory(self, memory_id: str):
        self.calls.append(("delete_memory", memory_id))
        return {"memory_id": memory_id, "status": "archived"}

    async def cancel_request(self, request_id: str):
        self.calls.append(("cancel_request", request_id))
        return {"request_id": request_id, "status": "cancelled"}

    async def get_request_status(self, request_id: str):
        self.calls.append(("get_request_status", request_id))
        return {"request_id": request_id, "status": "running"}

    async def runtime_status(self):
        self.calls.append(("runtime_status", None))
        return {
            "default_model_profile": "local_main",
            "model_profiles": {
                "local_main": {
                    "provider": "ollama",
                    "model": "qwen3.5:9b",
                    "max_output_tokens": 1024,
                    "temperature": 0.3,
                },
            },
        }

    async def ingest_project_docs(self):
        self.calls.append(("ingest_project_docs", None))
        return {"seen_sources": 2, "created_sources": 1, "updated_sources": 1, "created_chunks": 3}

    async def reindex_project_docs(self):
        self.calls.append(("reindex_project_docs", None))
        return {"seen_sources": 2, "updated_sources": 2, "created_chunks": 1}

    async def list_content_sources(self):
        self.calls.append(("list_content_sources", None))
        return {
            "sources": [
                {"path": "README.md", "status": "active", "title": "Readme"},
                {"path": "docs/guide.md", "status": "active", "title": "Guide"},
            ],
        }

    async def content_status(self):
        self.calls.append(("content_status", None))
        return {
            "sources": {"total": 2, "by_status": {"active": 2}},
            "chunks": {"total": 3, "by_status": {"active": 3}},
        }


class InterruptedStreamCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        raise asyncio.CancelledError
        yield


class ServerCancelledStreamCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        yield "request.processing.cancelled", {
            "error": {"code": "cancelled", "message": "request cancelled"},
        }


class SlowStreamCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        await asyncio.sleep(1.0)
        yield "token", {"delta": "late"}


class DelayedTokenStreamCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        await asyncio.sleep(0.2)
        yield "token", {"delta": "late"}
        yield "request.processing.completed", {"assistant_message_id": "message-1"}


class CleanupTrackingStreamCliClient(FakeCliClient):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.cleaned = False

    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        try:
            await asyncio.sleep(1.0)
            yield "token", {"delta": "late"}
        finally:
            await asyncio.sleep(0)
            self.cleaned = True


class ToolEventStreamCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        yield "tool.shell.started", {
            "tool_name": "tool.shell.read.project",
            "argv": ["rg", "ToolGatewayPort", "docs"],
        }
        yield "tool.shell.completed", {
            "tool_name": "tool.shell.read.project",
            "exit_code": 0,
            "output_bytes": 42,
            "truncated": False,
        }
        yield "token", {"delta": "ToolGatewayPort is documented."}
        yield "request.processing.completed", {"assistant_message_id": "message-1"}


class ToolUnavailableCliClient(FakeCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        yield "request.processing.failed", {
            "error": {
                "code": "working_directory_required",
                "message": "tool loop is rejected by policy",
                "details": {"raw": "hidden"},
            },
        }


class DegradedHealthCliClient(FakeCliClient):
    async def health(self):
        self.calls.append(("health", None))
        return {
            "status": "not_ready",
            "readiness": {
                "status": "failed",
                "checks": {"conversation_store": "ok", "inference": "failed"},
                "reasons": {"inference": "missing providers: local_main"},
            },
        }


class ApprovalPromptCliClient(FakeCliClient):
    def __init__(
        self,
        base_url: str,
        *,
        approval_status: str = "pending",
        deny_error: cli.CliUserError | None = None,
    ) -> None:
        super().__init__(base_url)
        self.approval_status = approval_status
        self.deny_error = deny_error

    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        yield "approval.required", {
            "approval_id": "approval-1",
            "capability": "tool.safe",
            "redacted_summary": "fake.echo(message)",
            "status": self.approval_status,
        }
        yield "request.processing.failed", {
            "error": {"code": "approval_denied", "message": "approval denied"},
        }

    async def get_approval(self, approval_id: str):
        self.calls.append(("get_approval", approval_id))
        return {
            "approval_id": approval_id,
            "status": self.approval_status,
            "capability": "tool.safe",
            "redacted_payload": {"summary": "fake.echo(message)"},
        }

    async def grant_approval(self, approval_id: str):
        self.calls.append(("grant_approval", approval_id))
        return {"approval_id": approval_id, "status": "granted"}

    async def deny_approval(self, approval_id: str):
        self.calls.append(("deny_approval", approval_id))
        if self.deny_error is not None:
            raise self.deny_error
        return {"approval_id": approval_id, "status": "denied"}


class ApprovalOnlyStreamCliClient(ApprovalPromptCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        yield "approval.required", {
            "approval_id": "approval-1",
            "capability": "tool.safe",
            "redacted_summary": "fake.echo(message)",
            "status": self.approval_status,
        }
        yield "token", {"delta": "approved path"}
        yield "request.processing.completed", {"assistant_message_id": "message-1"}


class DelayedApprovalOnlyStreamCliClient(ApprovalOnlyStreamCliClient):
    async def stream_request(self, request_id: str):
        self.calls.append(("stream_request", request_id))
        await asyncio.sleep(0.1)
        yield "approval.required", {
            "approval_id": "approval-1",
            "capability": "tool.safe",
            "redacted_summary": "fake.echo(message)",
            "status": self.approval_status,
        }
        yield "token", {"delta": "approved path"}
        yield "request.processing.completed", {"assistant_message_id": "message-1"}


class InterruptingInput:
    def readline(self) -> str:
        raise KeyboardInterrupt


class FailingHealthCliClient(FakeCliClient):
    async def health(self):
        self.calls.append(("health", None))
        raise cli.CliUserError("daemon unavailable")


class FailingResumeCliClient(FakeCliClient):
    async def get_conversation(self, conversation_id: str):
        self.calls.append(("get_conversation", conversation_id))
        raise cli.CliUserError("conversation not found")


class EmptyCliClient(FakeCliClient):
    async def list_conversations(self, *, limit: int = 20):
        self.calls.append(("list_conversations", {"limit": limit}))
        return {"conversations": []}

    async def list_memories(self):
        self.calls.append(("list_memories", {"query": None}))
        return {"memories": []}

    async def search_memories(self, query: str):
        self.calls.append(("list_memories", {"query": query}))
        return {"memories": []}


class MalformedListCliClient(FakeCliClient):
    async def list_conversations(self, *, limit: int = 20):
        self.calls.append(("list_conversations", {"limit": limit}))
        return {"conversations": [{"title": "broken"}]}

    async def list_memories(self):
        self.calls.append(("list_memories", {"query": None}))
        return {"memories": [{"content": "broken"}]}


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


def test_slash_command_menu_filters_by_prefix() -> None:
    stdout = StringIO()

    cli.write_slash_command_menu(stdout, prefix="/m")

    output = stdout.getvalue()
    assert "/memory add" in output
    assert "/memory list" in output
    assert "/new" not in output


def test_slash_command_registry_filters_by_partial_prefix() -> None:
    registry = cli.SlashCommandRegistry.from_commands(cli.SLASH_COMMANDS)

    matches = registry.matches("/memory s")
    usages = [command.usage for command in matches]

    assert "/memory search TEXT" in usages
    assert "/memory add TEXT" not in usages
    assert all(command.description for command in matches)


def test_slash_command_completion_includes_descriptions_and_argument_hints() -> None:
    registry = cli.SlashCommandRegistry.from_commands(cli.SLASH_COMMANDS)

    completion = registry.best_completion("/memory se")

    assert completion is not None
    assert completion.text == "/memory search "
    assert completion.display == "/memory search TEXT"
    assert completion.description == "Search manual memories."
    assert completion.argument_hint == "TEXT"


class TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("test stream has no fd")


class PipeTtyInput:
    def __init__(self, fd: int) -> None:
        self._file = os.fdopen(fd, "r", encoding="utf-8")

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._file.fileno()

    def readline(self) -> str:
        return self._file.readline()

    def close(self) -> None:
        self._file.close()


def test_prompt_toolkit_reader_is_selected_for_tty() -> None:
    reader = cli.create_interactive_line_reader(
        stdin=TtyStringIO(),
        stdout=TtyStringIO(),
    )

    assert isinstance(reader, cli.PromptToolkitLineReader)


def test_prompt_toolkit_reader_binds_injected_stdio(monkeypatch) -> None:
    created_kwargs: dict[str, Any] = {}
    stdin = TtyStringIO()
    stdout = TtyStringIO()

    class FakePromptSession:
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)

    class FakeCompleter:
        pass

    class FakeCompletion:
        def __init__(self, *args, **kwargs):
            pass

    class FakeHistory:
        pass

    prompt_toolkit_module = types.ModuleType("prompt_toolkit")
    prompt_toolkit_module.PromptSession = FakePromptSession
    completion_module = types.ModuleType("prompt_toolkit.completion")
    completion_module.Completer = FakeCompleter
    completion_module.Completion = FakeCompletion
    history_module = types.ModuleType("prompt_toolkit.history")
    history_module.History = FakeHistory
    input_module = types.ModuleType("prompt_toolkit.input")
    input_defaults_module = types.ModuleType("prompt_toolkit.input.defaults")
    output_module = types.ModuleType("prompt_toolkit.output")
    output_defaults_module = types.ModuleType("prompt_toolkit.output.defaults")
    input_defaults_module.create_input = lambda *, stdin=None: ("input", stdin)
    output_defaults_module.create_output = lambda *, stdout=None: ("output", stdout)

    monkeypatch.setitem(sys.modules, "prompt_toolkit", prompt_toolkit_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.completion", completion_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.history", history_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.input", input_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.input.defaults", input_defaults_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.output", output_module)
    monkeypatch.setitem(sys.modules, "prompt_toolkit.output.defaults", output_defaults_module)

    reader = cli.PromptToolkitLineReader(
        stdin=stdin,
        stdout=stdout,
        command_registry=cli.SlashCommandRegistry.from_commands(cli.SLASH_COMMANDS),
    )
    reader._ensure_session()

    assert created_kwargs["input"] == ("input", stdin)
    assert created_kwargs["output"] == ("output", stdout)


def test_plain_flag_forces_line_reader_without_prompt_toolkit() -> None:
    args = cli._parser().parse_args(["--plain", "chat"])

    assert args.plain is True

    reader = cli.create_interactive_line_reader(
        stdin=TtyStringIO("hello\n"),
        stdout=TtyStringIO(),
        plain=True,
    )

    assert isinstance(reader, cli.InteractiveLineReader)
    assert not isinstance(reader, cli.PromptToolkitLineReader)


def test_chat_subcommand_accepts_plain_after_subcommand() -> None:
    args = cli._parser().parse_args(["chat", "--plain"])

    assert args.plain is True


def test_status_line_renders_mode_readiness_conversation_phase_model_and_cwd_scope() -> None:
    line = cli.render_status_line(
        mode="auto",
        readiness="ready",
        conversation_id="conversation-1234567890",
        phase="tool running",
        model="local_main/qwen3.5:9b",
        cwd="/Users/alex/Jarvis",
        width=120,
    )

    assert "mode=auto" in line
    assert "ready" in line
    assert "conversation-1234567890" in line
    assert "tool running" in line
    assert "local_main/qwen3.5:9b" in line
    assert "Jarvis" in line
    assert len(line) <= 120


def test_status_line_redacts_secret_like_paths_and_truncates_long_values() -> None:
    line = cli.render_status_line(
        mode="tools",
        readiness="ready",
        conversation_id="conversation-" + ("x" * 80),
        phase="streaming",
        model="local_main",
        cwd="/Users/alex/Jarvis/sk-live-secret-token/project",
        width=88,
    )

    assert "sk-live-secret-token" not in line
    assert "project" in line
    assert len(line) <= 88


def test_shell_activity_transitions_follow_public_stream_events() -> None:
    activity = cli.ShellActivityState.idle().mark_submitting("request-1")

    assert activity.phase == "submitting"
    assert activity.request_id == "request-1"

    activity = activity.apply_stream_event("request.loop_selection.started", {})
    assert activity.phase == "selecting"
    activity = activity.apply_stream_event("context.assembly.started", {})
    assert activity.phase == "assembling context"
    activity = activity.apply_stream_event("memory.retrieved", {})
    assert activity.phase == "retrieving context"
    activity = activity.apply_stream_event("content.retrieved", {})
    assert activity.phase == "retrieving context"
    activity = activity.apply_stream_event("tool.shell.started", {"tool_name": "tool.shell.read.project"})
    assert activity.phase == "tool running"
    activity = activity.apply_stream_event("approval.required", {"approval_id": "approval-1"})
    assert activity.phase == "waiting approval"
    activity = activity.apply_stream_event("token", {"delta": "O"})
    assert activity.phase == "streaming"
    activity = activity.apply_stream_event("request.processing.completed", {})
    assert activity.phase == "done"


def test_shell_activity_never_renders_fake_percentages() -> None:
    activity = cli.ShellActivityState.idle().mark_submitting("request-1")

    for event_type in [
        "request.loop_selection.started",
        "context.assembly.started",
        "tool.shell.started",
        "token",
    ]:
        activity = activity.apply_stream_event(event_type, {})
        indicator = activity.render_indicator()
        assert "%" not in indicator
        assert "100" not in indicator


def test_tty_activity_writer_outputs_phase_without_percentages() -> None:
    stdout = TtyStringIO()
    activity = cli.ShellActivityState.idle().mark_submitting("request-1")

    cli.write_activity_indicator(stdout, activity, enabled=True)

    output = stdout.getvalue()
    assert "activity=submitting" in output
    assert "%" not in output
    assert "100" not in output


def test_status_and_model_renderers_return_payload_for_shell_state() -> None:
    client = FakeCliClient("http://test")
    stdout = StringIO()

    status_payload = asyncio.run(cli.write_status(client=client, stdout=stdout))
    model_payload = asyncio.run(cli.write_model_status(client=client, stdout=stdout))

    assert status_payload["status"] == "ready"
    assert model_payload["default_model_profile"] == "local_main"


def test_tty_cancel_command_can_cancel_active_stream() -> None:
    client = SlowStreamCliClient("http://test")
    stdout = TtyStringIO()
    read_fd, write_fd = os.pipe()
    stdin = PipeTtyInput(read_fd)

    async def scenario() -> int:
        async def write_cancel() -> None:
            await asyncio.sleep(0.05)
            os.write(write_fd, b"/cancel\n")
            os.close(write_fd)

        writer = asyncio.create_task(write_cancel())
        try:
            return await cli.submit_and_stream_message(
                client=client,
                stdout=stdout,
                conversation_id="conversation-1",
                content="slow answer",
                sensitivity="project",
                client_message_id="client-slow-cancel",
                assistant_prefix="assistant> ",
                stdin=stdin,
                allow_tty_cancel_command=True,
            )
        finally:
            await writer
            stdin.close()

    exit_code = asyncio.run(scenario())

    assert exit_code == 130
    assert ("cancel_request", "request-1") in client.calls
    assert "cancelled> request request-1" in stdout.getvalue()


def test_tty_cancel_monitor_does_not_consume_approval_input() -> None:
    client = ApprovalOnlyStreamCliClient("http://test")
    stdout = TtyStringIO()
    read_fd, write_fd = os.pipe()
    stdin = PipeTtyInput(read_fd)

    async def scenario() -> int:
        def write_approval() -> None:
            os.write(write_fd, b"y\n")
            os.close(write_fd)

        writer = threading.Timer(0.05, write_approval)
        writer.start()
        try:
            return await asyncio.wait_for(
                cli.submit_and_stream_message(
                    client=client,
                    stdout=stdout,
                    conversation_id="conversation-1",
                    content="needs approval",
                    sensitivity="project",
                    client_message_id="client-approval-tty",
                    assistant_prefix="assistant> ",
                    stdin=stdin,
                    allow_tty_cancel_command=True,
                ),
                timeout=1.0,
            )
        finally:
            writer.join(timeout=1.0)
            stdin.close()

    exit_code = asyncio.run(scenario())

    assert exit_code == 0
    assert ("grant_approval", "approval-1") in client.calls
    assert "approval> granted" in stdout.getvalue()


def test_tty_cancel_poll_does_not_reuse_typeahead_as_approval_input() -> None:
    client = DelayedApprovalOnlyStreamCliClient("http://test")
    stdout = TtyStringIO()
    read_fd, write_fd = os.pipe()
    stdin = PipeTtyInput(read_fd)
    os.write(write_fd, b"not approval\n")

    async def scenario() -> int:
        def write_approval() -> None:
            os.write(write_fd, b"y\n")
            os.close(write_fd)

        writer = threading.Timer(0.2, write_approval)
        writer.start()
        try:
            return await asyncio.wait_for(
                cli.submit_and_stream_message(
                    client=client,
                    stdout=stdout,
                    conversation_id="conversation-1",
                    content="needs delayed approval",
                    sensitivity="project",
                    client_message_id="client-delayed-approval-tty",
                    assistant_prefix="assistant> ",
                    stdin=stdin,
                    allow_tty_cancel_command=True,
                ),
                timeout=1.0,
            )
        finally:
            writer.join(timeout=1.0)
            stdin.close()

    exit_code = asyncio.run(scenario())

    assert exit_code == 0
    assert ("grant_approval", "approval-1") in client.calls
    assert ("deny_approval", "approval-1") not in client.calls
    assert "approval> granted" in stdout.getvalue()


def test_tty_cancel_poll_keeps_listening_after_non_cancel_typeahead() -> None:
    client = DelayedTokenStreamCliClient("http://test")
    stdout = TtyStringIO()
    read_fd, write_fd = os.pipe()
    stdin = PipeTtyInput(read_fd)
    os.write(write_fd, b"not a command\n/cancel\n")
    os.close(write_fd)

    async def scenario() -> int:
        try:
            return await cli.submit_and_stream_message(
                client=client,
                stdout=stdout,
                conversation_id="conversation-1",
                content="slow answer",
                sensitivity="project",
                client_message_id="client-typeahead-cancel",
                assistant_prefix="assistant> ",
                stdin=stdin,
                allow_tty_cancel_command=True,
            )
        finally:
            stdin.close()

    exit_code = asyncio.run(scenario())

    assert exit_code == 130
    assert ("cancel_request", "request-1") in client.calls


def test_tty_cancel_awaits_active_stream_cleanup() -> None:
    client = CleanupTrackingStreamCliClient("http://test")
    stdout = TtyStringIO()
    read_fd, write_fd = os.pipe()
    stdin = PipeTtyInput(read_fd)
    os.write(write_fd, b"/cancel\n")
    os.close(write_fd)

    async def scenario() -> int:
        try:
            return await cli.submit_and_stream_message(
                client=client,
                stdout=stdout,
                conversation_id="conversation-1",
                content="slow answer",
                sensitivity="project",
                client_message_id="client-cleanup-cancel",
                assistant_prefix="assistant> ",
                stdin=stdin,
                allow_tty_cancel_command=True,
            )
        finally:
            stdin.close()

    exit_code = asyncio.run(scenario())

    assert exit_code == 130
    assert client.cleaned is True


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


def test_terminal_line_reader_ctrl_c_on_prompt_returns_empty_line() -> None:
    stdin = StringIO("\x03/exit\n")
    stdout = StringIO()
    reader = cli.TerminalInteractiveLineReader(
        stdin=stdin,
        stdout=stdout,
        raw_mode=False,
    )

    assert reader.readline("jarvis> ") == ""
    assert reader.readline("jarvis> ") == "/exit"

    assert "^C" in stdout.getvalue()


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
                    "working_directory": str(Path.cwd()),
                },
            ),
        ("stream_request", "request-1"),
    ]


def test_http_submit_message_omits_loop_strategy_for_default_auto(monkeypatch) -> None:
    recorded: dict[str, Any] = {}

    class Response:
        status_code = 202

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"request_id": "request-1"}

    class RecordingAsyncClient:
        def __init__(self, *, base_url: str, timeout=None) -> None:
            self.base_url = base_url
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, path: str, *, json: dict[str, Any]) -> Response:
            recorded["path"] = path
            recorded["json"] = json
            return Response()

    monkeypatch.setattr(cli_client_module.httpx, "AsyncClient", RecordingAsyncClient)

    payload = asyncio.run(
        cli.HttpJarvisClient("http://testserver").submit_message(
            conversation_id="conversation-1",
            client_message_id="client-1",
            content="hello",
            sensitivity="project",
            loop_strategy=None,
        )
    )

    assert payload["request_id"] == "request-1"
    assert recorded["path"] == "/v1/conversations/conversation-1/messages"
    assert "loop_strategy" not in recorded["json"]


def test_http_submit_message_sends_working_directory_when_provided(monkeypatch) -> None:
    recorded: dict[str, Any] = {}

    class Response:
        status_code = 202

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"request_id": "request-1"}

    class RecordingAsyncClient:
        def __init__(self, *, base_url: str, timeout=None) -> None:
            self.base_url = base_url
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, path: str, *, json: dict[str, Any]) -> Response:
            recorded["json"] = json
            return Response()

    monkeypatch.setattr(cli_client_module.httpx, "AsyncClient", RecordingAsyncClient)

    asyncio.run(
        cli.HttpJarvisClient("http://testserver").submit_message(
            conversation_id="conversation-1",
            client_message_id="client-1",
            content="show cpu usage",
            sensitivity="project",
            working_directory="/tmp/project",
        )
    )

    assert recorded["json"]["working_directory"] == "/tmp/project"


def test_cli_chat_accepts_loop_strategy_override() -> None:
    stdout = StringIO()
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            [
                "chat",
                "--loop-strategy",
                "tools",
                "inspect",
                "project",
                "--client-message-id",
                "client-tools",
            ],
            client_factory=client_factory,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    assert clients[0].calls[1] == (
        "submit_message",
        {
            "conversation_id": "conversation-1",
            "client_message_id": "client-tools",
            "content": "inspect project",
            "sensitivity": "project",
            "working_directory": str(Path.cwd()),
            "loop_strategy": "tools",
        },
    )


def test_cli_auto_tool_intent_submits_caller_working_directory() -> None:
    stdout = StringIO()
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat", "show", "cpu", "usage"],
            client_factory=client_factory,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    submitted = clients[0].calls[1][1]
    assert "loop_strategy" not in submitted
    assert submitted["working_directory"] == str(Path.cwd())


def test_cli_project_docs_question_uses_auto_without_tool_override() -> None:
    stdout = StringIO()
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat", "where", "does", "ADR-035", "describe", "routing?"],
            client_factory=client_factory,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    submitted = clients[0].calls[1][1]
    assert "loop_strategy" not in submitted
    assert submitted["working_directory"] == str(Path.cwd())


def test_cli_tool_flow_renders_action_and_observation_without_raw_json_noise() -> None:
    stdout = StringIO()

    exit_code = asyncio.run(
        cli.run(
            ["chat", "inspect", "project"],
            client_factory=ToolEventStreamCliClient,
            stdout=stdout,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "tool> running tool.shell.read.project rg ToolGatewayPort docs" in output
    assert "tool> completed tool.shell.read.project exit=0 bytes=42" in output
    assert "ToolGatewayPort is documented." in output
    assert "{" not in output


def test_cli_tool_unavailable_message_is_clear_when_policy_denies() -> None:
    stdout = StringIO()

    exit_code = asyncio.run(
        cli.run(
            ["chat", "show", "cpu", "usage"],
            client_factory=ToolUnavailableCliClient,
            stdout=stdout,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 1
    assert "error> tool loop is rejected by policy (working_directory_required)" in output
    assert "{" not in output


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


def test_interactive_chat_continues_after_stream_interrupt() -> None:
    stdout = StringIO()
    stdin = StringIO("looping answer\n/status\n/exit\n")
    clients: list[InterruptedStreamCliClient] = []

    def client_factory(base_url: str) -> InterruptedStreamCliClient:
        client = InterruptedStreamCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "cancelled> request request-1" in output
    assert "status> ready" in output
    assert output.rstrip().endswith("bye")


def test_plain_interactive_chat_suppresses_tty_activity_indicator() -> None:
    stdout = TtyStringIO()
    stdin = StringIO("hello\n/exit\n")

    exit_code = asyncio.run(
        cli.run(
            ["chat", "--plain"],
            client_factory=FakeCliClient,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "assistant> OK" in output
    assert "activity=" not in output


def test_status_command_prints_degraded_reasons() -> None:
    stdout = StringIO()

    asyncio.run(cli.write_status(client=DegradedHealthCliClient("http://test"), stdout=stdout))

    output = stdout.getvalue()
    assert "status> not_ready" in output
    assert "reason> inference: missing providers: local_main" in output


def test_http_health_returns_not_ready_payload_from_503(monkeypatch) -> None:
    class DegradedResponse:
        status_code = 503

        def raise_for_status(self) -> None:
            request = httpx.Request("GET", "http://testserver/v1/health")
            response = httpx.Response(
                self.status_code,
                request=request,
                json=self.json(),
            )
            raise httpx.HTTPStatusError("service unavailable", request=request, response=response)

        def json(self) -> dict[str, Any]:
            return {
                "status": "not_ready",
                "readiness": {
                    "status": "failed",
                    "checks": {"inference": "failed"},
                    "reasons": {"inference": "missing providers: local_main"},
                },
            }

    class RecordingAsyncClient:
        def __init__(self, *, base_url: str, timeout=None) -> None:
            self.base_url = base_url
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, path: str) -> DegradedResponse:
            assert path == "/v1/health"
            return DegradedResponse()

    monkeypatch.setattr(cli_client_module.httpx, "AsyncClient", RecordingAsyncClient)

    payload = asyncio.run(cli.HttpJarvisClient("http://testserver").health())

    assert payload["status"] == "not_ready"
    assert payload["readiness"]["reasons"]["inference"] == "missing providers: local_main"


def test_content_ops_cli_commands_call_api() -> None:
    commands = [
        (["content", "ingest"], "content> ingested sources=2 chunks=3", ("ingest_project_docs", None)),
        (["content", "reindex"], "content> reindexed sources=2 chunks=1", ("reindex_project_docs", None)),
        (["content", "list"], "content> README.md active Readme", ("list_content_sources", None)),
        (["content", "status"], "content> sources=2 chunks=3", ("content_status", None)),
    ]

    for argv, expected_output, expected_call in commands:
        stdout = StringIO()
        clients: list[FakeCliClient] = []

        def client_factory(base_url: str) -> FakeCliClient:
            client = FakeCliClient(base_url)
            clients.append(client)
            return client

        exit_code = asyncio.run(
            cli.run(
                argv,
                client_factory=client_factory,
                stdout=stdout,
            ),
        )

        assert exit_code == 0
        assert expected_output in stdout.getvalue()
        assert clients[0].calls == [expected_call]


def test_stream_cancelled_event_returns_interrupt_status() -> None:
    stdout = StringIO()

    exit_code = asyncio.run(
        cli.run(
            ["chat", "cancelled", "server-side"],
            client_factory=ServerCancelledStreamCliClient,
            stdout=stdout,
        ),
    )

    assert exit_code == 130
    assert "cancelled> request request-1" in stdout.getvalue()


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
                    "working_directory": str(Path.cwd()),
                },
            ),
        ("stream_request", "request-1"),
    ]


def test_interactive_mode_defaults_to_auto() -> None:
    stdout = StringIO()
    stdin = StringIO("/status\n/exit\n")

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=FakeCliClient,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "mode> auto\n" in output
    assert "memory_augmented_answer" not in output
    assert "tool_react_loop" not in output


def test_interactive_mode_command_switches_between_auto_chat_tools() -> None:
    stdout = StringIO()
    stdin = StringIO(
        "/mode tools\n"
        "inspect project\n"
        "/mode chat\n"
        "explain design\n"
        "/mode auto\n"
        "hello auto\n"
        "/exit\n"
    )
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    submit_payloads = [
        payload for method, payload in clients[0].calls if method == "submit_message"
    ]
    assert exit_code == 0
    assert "mode> tools\n" in stdout.getvalue()
    assert "mode> chat\n" in stdout.getvalue()
    assert "mode> auto\n" in stdout.getvalue()
    assert submit_payloads[0]["loop_strategy"] == "tools"
    assert submit_payloads[1]["loop_strategy"] == "chat"
    assert "loop_strategy" not in submit_payloads[2]


def test_interactive_help_lists_mode_command() -> None:
    stdout = StringIO()

    cli.write_interactive_help(stdout)

    assert "/mode auto|chat|tools" in stdout.getvalue()


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
    assert "memory> memory-1 active fact project project.personal_assistant saved" in output
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
        ("list_memories", {"query": None}),
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
                    "working_directory": str(Path.cwd()),
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


def test_cli_memory_search_prints_rich_results() -> None:
    stdout = StringIO()
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["memory", "search", "search", "target"],
            client_factory=client_factory,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    assert stdout.getvalue() == (
        "memory> memory-2 active fact project project.personal_assistant search hit\n"
    )
    assert clients[0].calls == [("list_memories", {"query": "search target"})]


def test_cli_memory_delete_archives_memory() -> None:
    stdout = StringIO()

    exit_code = asyncio.run(
        cli.run(
            ["memory", "delete", "memory-2"],
            client_factory=FakeCliClient,
            stdout=stdout,
        ),
    )

    assert exit_code == 0
    assert stdout.getvalue() == "memory> memory-2 archived\n"


def test_cli_interactive_control_surface_and_sessions() -> None:
    stdout = StringIO()
    stdin = StringIO(
        "/status\n"
        "/model\n"
        "/sessions\n"
        "/resume conversation-1\n"
        "/clear\n"
        "hello after clear\n"
        "/cancel\n"
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

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status> ready" in output
    assert "model> local_main ollama qwen3.5:9b" in output
    assert "session> conversation-1 active First session" in output
    assert "conversation> resumed conversation-1 Resumed session" in output
    assert "conversation> cleared" in output
    assert "cancelled> no request" in output
    assert ("cancel_request", "request-1") not in clients[0].calls


def test_cli_interactive_keeps_running_after_command_error() -> None:
    stdout = StringIO()
    stdin = StringIO("/resume missing-conversation\n/status\n/exit\n")
    clients: list[FailingResumeCliClient] = []

    def client_factory(base_url: str) -> FailingResumeCliClient:
        client = FailingResumeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "error> conversation not found" in output
    assert "status> ready" in output
    assert output.rstrip().endswith("bye")


def test_cli_interactive_cancel_does_not_reuse_completed_request() -> None:
    stdout = StringIO()
    stdin = StringIO("hello\n/cancel\n/exit\n")
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    assert exit_code == 0
    assert "cancelled> no request" in stdout.getvalue()
    assert ("cancel_request", "request-1") not in clients[0].calls


def test_cli_memory_search_delete_and_rich_list_output() -> None:
    stdout = StringIO()
    stdin = StringIO(
        "/memory list\n"
        "/memory search search target\n"
        "/memory delete memory-2\n"
        "/exit\n"
    )
    clients: list[FakeCliClient] = []

    def client_factory(base_url: str) -> FakeCliClient:
        client = FakeCliClient(base_url)
        clients.append(client)
        return client

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=client_factory,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "memory> memory-1 active fact project project.personal_assistant saved" in output
    assert "memory> memory-2 active fact project project.personal_assistant search hit" in output
    assert "memory> memory-2 archived" in output
    assert ("list_memories", {"query": "search target"}) in clients[0].calls
    assert ("delete_memory", "memory-2") in clients[0].calls


def test_cli_interactive_empty_lists_are_explicit() -> None:
    stdout = StringIO()
    stdin = StringIO("/sessions\n/memory list\n/memory search none\n/exit\n")

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=EmptyCliClient,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "sessions> empty" in output
    assert "memory> empty" in output


def test_cli_interactive_reports_malformed_list_payloads_without_exit() -> None:
    stdout = StringIO()
    stdin = StringIO("/sessions\n/memory list\n/status\n/exit\n")

    exit_code = asyncio.run(
        cli.run(
            ["chat"],
            client_factory=MalformedListCliClient,
            stdout=stdout,
            stdin=stdin,
        ),
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert output.count("error> daemon response missing string field") == 2
    assert "status> ready" in output
    assert output.rstrip().endswith("bye")


def test_parse_sse_blocks() -> None:
    events = cli.parse_sse_blocks(
        'event: token\ndata: {"delta": "A"}\n\n'
        'event: request.processing.completed\ndata: {"request_id": "r"}\n\n',
    )

    assert events == [
        ("token", {"delta": "A"}),
        ("request.processing.completed", {"request_id": "r"}),
    ]


def test_cli_renders_approval_prompt() -> None:
    client = ApprovalPromptCliClient("http://test")
    stdout = StringIO()

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=stdout,
            stdin=StringIO("n\n"),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix="assistant> ",
        ),
    )

    output = stdout.getvalue()
    assert "approval> tool.safe wants to perform fake.echo(message)" in output
    assert "approve? [y/N]" in output


def test_empty_cli_approval_input_denies() -> None:
    client = ApprovalPromptCliClient("http://test")

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=StringIO(),
            stdin=StringIO("\n"),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix=None,
        ),
    )

    assert ("deny_approval", "approval-1") in client.calls
    assert ("grant_approval", "approval-1") not in client.calls


def test_cancel_cli_approval_input_marks_prompt_cancelled() -> None:
    client = ApprovalPromptCliClient("http://test")
    stdout = StringIO()

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=stdout,
            stdin=StringIO("cancel\n"),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix=None,
        ),
    )

    assert ("cancel_request", "request-1") in client.calls
    assert ("deny_approval", "approval-1") not in client.calls
    assert "approval> cancelled" in stdout.getvalue()


def test_yes_cli_approval_input_grants() -> None:
    client = ApprovalPromptCliClient("http://test")

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=StringIO(),
            stdin=StringIO("yes\n"),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix=None,
        ),
    )

    assert ("grant_approval", "approval-1") in client.calls
    assert ("deny_approval", "approval-1") not in client.calls


def test_cli_ctrl_c_denies_or_cancels_local_wait() -> None:
    client = ApprovalPromptCliClient("http://test")

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=StringIO(),
            stdin=InterruptingInput(),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix=None,
        ),
    )

    assert ("cancel_request", "request-1") in client.calls
    assert ("deny_approval", "approval-1") not in client.calls


def test_cli_reports_expired_approval() -> None:
    client = ApprovalPromptCliClient("http://test", approval_status="expired")
    stdout = StringIO()

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=stdout,
            stdin=StringIO("yes\n"),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix=None,
        ),
    )

    assert "approval> expired" in stdout.getvalue()
    assert ("grant_approval", "approval-1") not in client.calls


def test_cli_reports_expired_approval_when_empty_deny_conflicts() -> None:
    client = ApprovalPromptCliClient(
        "http://test",
        deny_error=cli.CliUserError("approval_expired: approval expired"),
    )
    stdout = StringIO()

    asyncio.run(
        cli.submit_and_stream_message(
            client=client,
            stdout=stdout,
            stdin=StringIO("\n"),
            conversation_id="conversation-1",
            content="use tool",
            sensitivity="project",
            client_message_id="client-approval",
            assistant_prefix=None,
        ),
    )

    assert "approval> expired" in stdout.getvalue()


def test_http_error_message_uses_api_error_payload() -> None:
    request = httpx.Request("GET", "http://testserver/v1/health")
    response = httpx.Response(
        503,
        request=request,
        json={"error": {"message": "daemon warming up"}},
    )
    exc = httpx.HTTPStatusError("service unavailable", request=request, response=response)

    assert cli._http_error_message(exc, "health") == "health failed: 503 daemon warming up"


def test_http_error_message_includes_validation_error_locations() -> None:
    request = httpx.Request("POST", "http://testserver/v1/conversations/id/messages")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "code": "invalid_request",
                "message": "request validation failed",
                "details": {
                    "errors": [
                        {
                            "loc": ["body", "working_directory"],
                            "msg": "Extra inputs are not permitted",
                            "type": "extra_forbidden",
                        },
                    ],
                },
            },
        },
    )
    exc = httpx.HTTPStatusError("bad request", request=request, response=response)

    assert cli._http_error_message(exc, "post message") == (
        "post message failed: 400 invalid_request: request validation failed "
        "(body.working_directory: extra_forbidden: Extra inputs are not permitted)"
    )


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

    monkeypatch.setattr(cli_client_module.httpx, "AsyncClient", RecordingAsyncClient)

    response = asyncio.run(cli.HttpJarvisClient("http://testserver").health())

    assert response == {"status": "ready"}
    assert timeouts == [cli.REQUEST_TIMEOUT_SECONDS]


def test_http_client_maps_control_surface_endpoints(monkeypatch) -> None:
    calls: list[tuple[str, str, Any]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class RecordingAsyncClient:
        def __init__(self, *, base_url: str, timeout=None) -> None:
            self.base_url = base_url
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, path: str, params=None) -> FakeResponse:
            calls.append(("GET", path, params))
            return FakeResponse({"ok": True})

        async def post(self, path: str, json=None) -> FakeResponse:
            calls.append(("POST", path, json))
            return FakeResponse({"ok": True})

    monkeypatch.setattr(cli_client_module.httpx, "AsyncClient", RecordingAsyncClient)
    client = cli.HttpJarvisClient("http://testserver")

    asyncio.run(client.list_conversations(limit=7))
    asyncio.run(client.get_conversation("conversation-1"))
    asyncio.run(client.runtime_status())
    asyncio.run(client.search_memories("needle"))
    asyncio.run(client.delete_memory("memory-1"))
    asyncio.run(client.cancel_request("request-1"))

    assert calls == [
        ("GET", "/v1/conversations", {"limit": 7}),
        ("GET", "/v1/conversations/conversation-1", None),
        ("GET", "/v1/runtime/status", None),
        ("GET", "/v1/memories", {"query": "needle"}),
        ("POST", "/v1/memories/memory-1/archive", {}),
        ("POST", "/v1/requests/request-1/cancel", {}),
    ]
