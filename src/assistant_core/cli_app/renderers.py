from __future__ import annotations

from typing import TextIO

from assistant_core.cli_app.client import JarvisClient
from assistant_core.cli_app.config import SLASH_COMMANDS
from assistant_core.cli_app.utils import _display_text, _required_str


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


async def write_status(*, client: JarvisClient, stdout: TextIO) -> dict:
    payload = await client.health()
    stdout.write(f"status> {_display_text(payload.get('status'))}\n")
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict):
        return payload
    reasons = readiness.get("reasons")
    if not isinstance(reasons, dict):
        return payload
    for component, reason in sorted(reasons.items()):
        stdout.write(f"reason> {_display_text(component)}: {_display_text(reason)}\n")
    return payload


async def write_model_status(*, client: JarvisClient, stdout: TextIO) -> dict:
    payload = await client.runtime_status()
    profile_name = _display_text(payload.get("default_model_profile"))
    profiles = payload.get("model_profiles", {})
    profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
    provider = _display_text(profile.get("provider")) if isinstance(profile, dict) else ""
    model = _display_text(profile.get("model")) if isinstance(profile, dict) else ""
    max_output_tokens = profile.get("max_output_tokens") if isinstance(profile, dict) else None
    temperature = profile.get("temperature") if isinstance(profile, dict) else None
    stdout.write(f"model> {profile_name} {provider} {model}")
    if max_output_tokens is not None:
        stdout.write(f" max_output_tokens={max_output_tokens}")
    if temperature is not None:
        stdout.write(f" temperature={temperature}")
    stdout.write("\n")
    return payload


async def write_content_ingest(*, client: JarvisClient, stdout: TextIO) -> None:
    payload = await client.ingest_project_docs()
    stdout.write(
        "content> ingested "
        f"sources={_display_text(payload.get('seen_sources'))} "
        f"chunks={_display_text(payload.get('created_chunks'))}\n",
    )


async def write_content_reindex(*, client: JarvisClient, stdout: TextIO) -> None:
    payload = await client.reindex_project_docs()
    stdout.write(
        "content> reindexed "
        f"sources={_display_text(payload.get('updated_sources'))} "
        f"chunks={_display_text(payload.get('created_chunks'))}\n",
    )


async def write_content_sources(*, client: JarvisClient, stdout: TextIO) -> None:
    payload = await client.list_content_sources()
    sources = payload.get("sources", [])
    if not sources:
        stdout.write("content> empty\n")
        return
    for source in sources:
        if not isinstance(source, dict):
            continue
        stdout.write(
            "content> "
            f"{_display_text(source.get('path'))} "
            f"{_display_text(source.get('status'))} "
            f"{_display_text(source.get('title'))}\n",
        )


async def write_content_status(*, client: JarvisClient, stdout: TextIO) -> None:
    payload = await client.content_status()
    sources = payload.get("sources", {})
    chunks = payload.get("chunks", {})
    source_total = sources.get("total") if isinstance(sources, dict) else ""
    chunk_total = chunks.get("total") if isinstance(chunks, dict) else ""
    stdout.write(f"content> sources={_display_text(source_total)} chunks={_display_text(chunk_total)}\n")


async def write_conversation_list(*, client: JarvisClient, stdout: TextIO) -> None:
    payload = await client.list_conversations(limit=20)
    conversations = payload.get("conversations", [])
    if not conversations:
        stdout.write("sessions> empty\n")
        return
    for conversation in conversations:
        conversation_id = _required_str(conversation, "conversation_id")
        stdout.write(
            "session> "
            f"{conversation_id} "
            f"{_display_text(conversation.get('status'))} "
            f"{_display_text(conversation.get('title'))}\n"
        )


async def write_memory_list(
    *,
    client: JarvisClient,
    stdout: TextIO,
    query: str | None = None,
) -> None:
    payload = await client.list_memories() if query is None else await client.search_memories(query)
    memories = payload.get("memories", [])
    if not memories:
        stdout.write("memory> empty\n")
        return
    for memory in memories:
        memory_id = _required_str(memory, "memory_id")
        stdout.write(
            "memory> "
            f"{memory_id} "
            f"{_display_text(memory.get('status'))} "
            f"{_display_text(memory.get('memory_type'))} "
            f"{_display_text(memory.get('sensitivity'))} "
            f"{_display_text(memory.get('namespace'))} "
            f"{_display_text(memory.get('content'))}\n"
        )


def is_tool_stream_event(event_type: str) -> bool:
    return event_type.startswith("tool.shell.") or event_type.startswith("tool.system.diagnostics.")


def write_tool_stream_event(stdout: TextIO, *, event_type: str, data: dict) -> None:
    tool_name = _display_text(data.get("tool_name"))
    if event_type.endswith(".started"):
        argv = _display_argv(data.get("argv"))
        suffix = f" {argv}" if argv else ""
        stdout.write(f"tool> running {tool_name}{suffix}\n")
        return
    if event_type.endswith(".completed"):
        details = []
        if data.get("exit_code") is not None:
            details.append(f"exit={_display_text(data.get('exit_code'))}")
        if data.get("output_bytes") is not None:
            details.append(f"bytes={_display_text(data.get('output_bytes'))}")
        suffix = f" {' '.join(details)}" if details else ""
        stdout.write(f"tool> completed {tool_name}{suffix}\n")
        return
    if event_type.endswith(".denied"):
        code = _display_text(data.get("error_code") or data.get("policy_outcome"))
        suffix = f" ({code})" if code else ""
        stdout.write(f"tool> denied {tool_name}{suffix}\n")
        return
    if event_type.endswith(".unavailable"):
        source = _display_text(data.get("source") or data.get("family"))
        suffix = f" {source}" if source else ""
        stdout.write(f"tool> unavailable {tool_name}{suffix}\n")
        return
    stdout.write(f"tool> {_display_text(event_type)} {tool_name}\n")


def write_request_error(stdout: TextIO, data: dict) -> None:
    error = data.get("error")
    if not isinstance(error, dict):
        stdout.write(f"error> {_display_text(data)}\n")
        return
    message = _display_text(error.get("message") or "request failed")
    code = _display_text(error.get("code"))
    if code:
        stdout.write(f"error> {message} ({code})\n")
        return
    stdout.write(f"error> {message}\n")


def _display_argv(value) -> str:
    if not isinstance(value, list):
        return ""
    return " ".join(_display_text(item) for item in value)


__all__ = [
    "is_tool_stream_event",
    "write_content_ingest",
    "write_content_reindex",
    "write_content_sources",
    "write_content_status",
    "write_conversation_list",
    "write_interactive_help",
    "write_memory_list",
    "write_model_status",
    "write_request_error",
    "write_slash_command_menu",
    "write_status",
    "write_tool_stream_event",
]
