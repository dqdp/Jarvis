from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TextIO

from assistant_core.cli_app.client import CliUserError, HttpJarvisClient, JarvisClient
from assistant_core.cli_app.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MEMORY_TYPE,
    DEFAULT_PROJECT_NAMESPACE,
    DEFAULT_SENSITIVITY,
    LOOP_STRATEGY_CHOICES,
)
from assistant_core.cli_app.interactive import run_interactive_chat, submit_and_stream_message
from assistant_core.cli_app.renderers import (
    write_content_ingest,
    write_content_reindex,
    write_content_sources,
    write_content_status,
    write_memory_list,
)
from assistant_core.cli_app.utils import _display_text, _required_str


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
            loop_strategy=None,
            working_directory=str(Path.cwd()),
            plain=args.plain,
            developer_mode=args.developer,
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
                loop_strategy=args.loop_strategy,
                working_directory=args.working_directory or str(Path.cwd()),
                plain=args.plain,
                developer_mode=args.developer,
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
            loop_strategy=args.loop_strategy,
            working_directory=args.working_directory or str(Path.cwd()),
            client_message_id=args.client_message_id,
            assistant_prefix=None,
            stdin=stdin,
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
        await write_memory_list(client=client, stdout=stdout)
        return 0

    if args.command == "memory" and args.memory_command == "search":
        await write_memory_list(client=client, stdout=stdout, query=" ".join(args.query))
        return 0

    if args.command == "memory" and args.memory_command == "delete":
        memory = await client.delete_memory(args.memory_id)
        stdout.write(
            f"memory> {_required_str(memory, 'memory_id')} {_display_text(memory.get('status'))}\n"
        )
        return 0

    if args.command == "content" and args.content_command == "ingest":
        await write_content_ingest(client=client, stdout=stdout)
        return 0

    if args.command == "content" and args.content_command == "reindex":
        await write_content_reindex(client=client, stdout=stdout)
        return 0

    if args.command == "content" and args.content_command == "list":
        await write_content_sources(client=client, stdout=stdout)
        return 0

    if args.command == "content" and args.content_command == "status":
        await write_content_status(client=client, stdout=stdout)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    try:
        exit_code = asyncio.run(run(argv))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--plain", action="store_true", help="Use deterministic line-oriented CLI.")
    parser.add_argument(
        "--developer",
        action="store_true",
        help="Show developer diagnostics in interactive CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health")

    chat = subparsers.add_parser("chat")
    chat.add_argument("message", nargs="*")
    chat.add_argument("--conversation-id")
    chat.add_argument("--client-message-id")
    chat.add_argument("--project-namespace", default=DEFAULT_PROJECT_NAMESPACE)
    chat.add_argument("--sensitivity", default=DEFAULT_SENSITIVITY)
    chat.add_argument("--title")
    chat.add_argument("--loop-strategy", choices=LOOP_STRATEGY_CHOICES)
    chat.add_argument("--working-directory")
    chat.add_argument("--plain", action="store_true", default=argparse.SUPPRESS)
    chat.add_argument("--developer", action="store_true", default=argparse.SUPPRESS)

    memory = subparsers.add_parser("memory")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)
    memory_add = memory_subparsers.add_parser("add")
    memory_add.add_argument("content", nargs="+")
    memory_add.add_argument("--namespace", default=DEFAULT_PROJECT_NAMESPACE)
    memory_add.add_argument("--memory-type", default=DEFAULT_MEMORY_TYPE)
    memory_add.add_argument("--sensitivity", default=DEFAULT_SENSITIVITY)
    memory_subparsers.add_parser("list")
    memory_search = memory_subparsers.add_parser("search")
    memory_search.add_argument("query", nargs="+")
    memory_delete = memory_subparsers.add_parser("delete")
    memory_delete.add_argument("memory_id")

    content = subparsers.add_parser("content")
    content_subparsers = content.add_subparsers(dest="content_command", required=True)
    content_subparsers.add_parser("ingest")
    content_subparsers.add_parser("reindex")
    content_subparsers.add_parser("list")
    content_subparsers.add_parser("status")
    return parser

__all__ = ["main", "run", "_parser"]
