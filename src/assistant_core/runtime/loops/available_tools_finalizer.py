from __future__ import annotations

import re

from assistant_core.domain.loops import ToolRequestPlan
from assistant_core.runtime.loops.tool_catalog import allowed_tool_catalog


AVAILABLE_TOOLS_FINALIZER_SOURCE = "deterministic_available_tools"

_AVAILABLE_TOOLS_PATTERNS: tuple[str, ...] = (
    r"\bwhat\s+(?:local\s+)?tools\s+are\s+(?:currently\s+)?(?:available|enabled|allowed)(?:\s+(?:now|right\s+now|currently|for\s+this\s+request|to\s+you(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?))?\s*[?.!]*$",
    r"\bwhich\s+(?:local\s+)?tools\s+are\s+(?:currently\s+)?(?:available|enabled|allowed)(?:\s+(?:now|right\s+now|currently|for\s+this\s+request|to\s+you(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?))?\s*[?.!]*$",
    r"\bwhich\s+(?:local\s+)?tools\s+can\s+you\s+use(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?\s*[?.!]*$",
    r"\bwhat\s+(?:local\s+)?tools\s+can\s+you\s+use(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?\s*[?.!]*$",
    r"\bwhat\s+tools\s+do\s+you\s+have(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?\s*[?.!]*$",
    r"\bwhat\s+capabilities\s+(?:do\s+you\s+have|can\s+you\s+use)(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?\s*[?.!]*$",
    r"\bwhat\s+capabilities\s+are\s+(?:currently\s+)?(?:available|enabled|allowed)(?:\s+(?:now|right\s+now|currently|for\s+this\s+request|to\s+you(?:\s+(?:now|right\s+now|currently|for\s+this\s+request))?))?\s*[?.!]*$",
    r"(?:какие|что\s+за)\s+(?:инструменты|тулы|tools?|возможности)\s+(?:(?:тебе|вам)\s+)?(?:(?:сейчас|текущ\w*)\s+)?(?:доступ\w*|разреш\w*|есть)\s*[?.!]*$",
    r"(?:какие|что\s+за)\s+(?:инструменты|тулы|tools?|возможности)\s+(?:(?:тебе|вам)\s+)?(?:доступ\w*|разреш\w*)\s+(?:сейчас|текущ\w*)\s*[?.!]*$",
    r"(?:какие|что\s+за)\s+(?:инструменты|тулы|tools?|возможности)\s+(?:доступ\w*|разреш\w*)\s+(?:сейчас|текущ\w*|тебе|вам)\s*[?.!]*$",
    r"(?:какие|что\s+за)\s+(?:инструменты|тулы|tools?|возможности)\s+(?:доступ\w*|разреш\w*)\s+(?:(?:тебе|вам)\s+)(?:сейчас|текущ\w*)\s*[?.!]*$",
    r"(?:какими|какие)\s+(?:инструментами|инструменты|тулами|тулы)\s+(?:(?:ты|вы)\s+)?(?:можешь|можете)(?:\s+(?:использовать|пользоваться))?(?:\s+(?:сейчас|текущ\w*))?\s*[?.!]*$",
    r"(?:инструменты|тулы|tools?|возможности)\s+(?:сейчас|текущ\w*)\s+(?:доступ\w*|разреш\w*)\s*[?.!]*$",
    r"(?:инструменты|тулы|tools?|возможности)\s+(?:доступ\w*|разреш\w*)\s+(?:сейчас|текущ\w*)\s*[?.!]*$",
)
_TOOL_DOCUMENTATION_NEAR_MISS_PATTERNS: tuple[str, ...] = (
    r"\b(?:architecture|design|implementation|documentation|docs?|adr)\b.*\btools?\b",
    r"\btools?\b.*\b(?:architecture|design|implementation|documentation|docs?|adr)\b",
    r"\bhow\s+(?:do|does)\s+.*\btools?\b.*\b(?:work|implemented|built)\b",
    r"\bexplain\b.*\btools?\b.*\b(?:work|architecture|design|implementation)\b",
    r"(?:архитектур|документац|реализац|устройств).*(?:инструмент|тул)",
    r"(?:инструмент|тул).*(?:архитектур|документац|реализац|устроен)",
)
_COMPOUND_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\b(?:and|also|plus)\b",
    r"\b(?:и|а\s+также|ещ[её])\b",
    r"[,;]",
)


def deterministic_available_tools_response(
    user_input: str,
    request_plan: ToolRequestPlan,
) -> str | None:
    if not is_current_available_tools_request(user_input):
        return None
    allowed = request_plan.allowed_tool_names or frozenset()
    if request_plan.policy not in {"available", "required"} or not allowed:
        return "No local tools are available for this request."
    catalog = allowed_tool_catalog(
        request_plan.allowed_tool_summaries,
        allowed_tool_names=allowed,
    )
    if not catalog:
        catalog = [f"{tool_name}." for tool_name in sorted(allowed)]
    return "Available local tools for this request:\n" + "\n".join(
        f"- {item}" for item in catalog
    )


def is_current_available_tools_request(user_input: str) -> bool:
    value = " ".join(user_input.split())
    if not value:
        return False
    if _matches_any(_TOOL_DOCUMENTATION_NEAR_MISS_PATTERNS, value):
        return False
    if _matches_any(_COMPOUND_REQUEST_PATTERNS, value):
        return False
    return _full_matches_any(_AVAILABLE_TOOLS_PATTERNS, value)


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _full_matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.fullmatch(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


__all__ = [
    "AVAILABLE_TOOLS_FINALIZER_SOURCE",
    "deterministic_available_tools_response",
    "is_current_available_tools_request",
]
