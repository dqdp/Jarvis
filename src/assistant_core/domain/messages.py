from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from assistant_core.domain.sensitivity import Sensitivity


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    DEVELOPER = "developer"


@dataclass(frozen=True)
class TextPart:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: list[TextPart]
    name: str | None = None
    sensitivity: Sensitivity = Sensitivity.PERSONAL
    metadata: dict[str, Any] = field(default_factory=dict)
