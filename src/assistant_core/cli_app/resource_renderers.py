from __future__ import annotations

from typing import TextIO

from assistant_core.cli_app.client import JarvisClient
from assistant_core.cli_app.utils import _display_text, _required_str


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
    stdout.write(
        f"content> sources={_display_text(source_total)} "
        f"chunks={_display_text(chunk_total)}\n"
    )


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


__all__ = [
    "write_content_ingest",
    "write_content_reindex",
    "write_content_sources",
    "write_content_status",
    "write_conversation_list",
    "write_memory_list",
]
