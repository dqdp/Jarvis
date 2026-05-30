from __future__ import annotations

from dataclasses import dataclass


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
    SlashCommand("/status", "Show daemon readiness."),
    SlashCommand("/model", "Show active local model profile."),
    SlashCommand("/sessions", "List recent conversations."),
    SlashCommand("/resume ID", "Resume a conversation."),
    SlashCommand("/new [title]", "Start a new conversation."),
    SlashCommand("/clear", "Clear current conversation."),
    SlashCommand("/cancel [request_id]", "Cancel the last or selected request."),
    SlashCommand("/memory add TEXT", "Save manual memory."),
    SlashCommand("/memory list", "List manual memories."),
    SlashCommand("/memory search TEXT", "Search manual memories."),
    SlashCommand("/memory delete ID", "Archive a manual memory."),
    SlashCommand("/exit", "Quit."),
    SlashCommand("/quit", "Quit."),
)
