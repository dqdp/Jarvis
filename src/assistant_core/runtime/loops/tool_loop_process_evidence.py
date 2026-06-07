from __future__ import annotations

import re

from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.tools import ToolObservationStatus, ToolParseStatus
from assistant_core.runtime.loops.tool_loop_process_resource_evidence import (
    PROCESS_RESOURCE_SCHEMA,
    process_resource_payload_matches_request,
    requested_process_resource_names,
    required_process_resource_metrics,
    requires_process_resource_payload,
)


PROCESS_TOOL_NAMES = frozenset({"tool.system.read.process"})
_PROCESS_NAME_SEARCH_SCHEMA = "system.process_name_search"

PROCESS_SHARED_SUFFIX_TOPIC_PATTERNS: tuple[str, ...] = (
    r"^(?:process(?:es)?|service(?:s)?|pid|pids|pgrep|ps)$",
    r"^(?:ollama|postgres(?:ql)?|redis|python|node|java|docker|nginx|uvicorn|gunicorn|jarvis)$",
    r"^(?:процесс(?:ы)?|служб\w*|пид|pid)$",
)

PROCESS_TASK_CONTINUATION_CLAUSE_PATTERNS: tuple[str, ...] = (
    r"^(?:is|are|does|do|check|show)\s+(?:the\s+|my\s+)?[\w.-]{2,}(?:\s+process)?\s+(?:running|active|up)\??$",
    r"^(?:is|are|does|do|check|show)\s+(?:the\s+)?(?:process(?:es)?|service(?:s)?)\s+(?:running|active|up)\??$",
    r"^(?:запущ|работ|проверь|покажи).*(?:процесс|служб|ollama|postgres|redis|python|node|java|docker|nginx|uvicorn|gunicorn|jarvis)$",
)

_PROCESS_STATUS_PATTERNS: tuple[str, ...] = (
    r"\b(?:process(?:es)?|pid|pids|pgrep|ps)\b",
    r"\b(?:status|state)\b.*\b(?:process(?:es)?|service(?:s)?)\b",
    r"\b(?:process(?:es)?|service(?:s)?)\b.*\b(?:running|active|up|status|state)\b",
    r"\b(?:is|are)\s+(?:the\s+)?(?:ollama|postgres(?:ql)?|redis|python|node|java|docker|nginx|uvicorn|gunicorn|jarvis)\b.*\b(?:running|active|up)\b",
    r"\b(?:is|are)\s+(?:the\s+)?(?!(?:the|my|wi[- ]?fi|wifi|internet|network|vpn)\b)[\w.-]{2,}\s+(?:process\s+)?(?:running|active|up)\b",
    r"\b(?:is|are)\s+(?:the\s+|my\s+)?(?!(?:the|my|wi[- ]?fi|wifi|internet|network|vpn)\b)[\w.-]{2,}(?:\s+[\w.-]{2,}){1,3}\s+(?:running|active|up)\b",
    r"\b(?:процесс(?:ы)?|пид|pid|pgrep|ps)\b",
    r"(?:статус|состояни).*(?:процесс|служб)",
    r"(?:процесс|служб).*(?:запущ|работ|актив|статус|состояни)",
    r"(?:запущ|работ).*(?:ollama|postgres(?:ql)?|redis|python|node|java|docker|nginx|uvicorn|gunicorn|jarvis)",
)

_PROCESS_NAME_TOKEN = r"[\w.-]{2,}(?:\s+[\w.-]{2,}){0,3}"
_PROCESS_NAME_PATTERNS: tuple[str, ...] = (
    rf"\b(?:is|are|does|do|check|show)\s+(?:the\s+|my\s+)?(?P<name>{_PROCESS_NAME_TOKEN})(?:\s+process)?\s+(?:running|active|up)\b",
    rf"\b(?:is|are|does|do|check|show)\s+(?:the\s+|my\s+)?(?P<name>{_PROCESS_NAME_TOKEN})\s+service\s+(?:running|active|up)\b",
    rf"\b(?:is|are|does|do|check|show)\s+(?:the\s+)?(?:process|service)\s+(?P<name>{_PROCESS_NAME_TOKEN})\s+(?:running|active|up)\b",
    r"\b(?:the\s+)?(?P<name>(?!(?:the|is|are|what)\b)[\w.-]{2,}(?:\s+[\w.-]{2,}){0,2})\s+daemon\s+(?:(?:is|are)\s+)?(?:status|health|running|active|up)\b",
    r"\b(?P<name>[\w.-]{2,})\s+process\b",
    r"\b(?P<name>[\w.-]{2,})\s+service\b",
    r"\b(?:process|service)\s+(?:named\s+|called\s+)(?P<name>[\w.-]{2,})\b",
)
_PROCESS_PID_PATTERNS: tuple[str, ...] = (
    r"\bpid\s+(?P<pid>\d+)\b", r"\bprocess\s+(?P<pid>\d+)\b", r"\bпид\s+(?P<pid>\d+)\b",
)
_PROCESS_NAME_STOPWORDS = frozenset({
    "active", "are", "check", "code", "computer", "cpu", "current", "do", "does", "is",
    "load", "local", "mac", "machine", "info", "list", "memory", "my", "process", "processor", "ram", "running",
    "script", "service", "show", "state", "status", "system", "table", "the",
    "usage", "utilisation", "utilization",
})

_PROCESS_SCOPED_RESOURCE_PATTERNS: tuple[str, ...] = (r"\b(?:process|service|pid)\b", r"\b(?:процесс|служб|пид)\b")
def matches_process_live_state_intent(value: str) -> bool:
    return _matches_any(_PROCESS_STATUS_PATTERNS, value)


def is_process_scoped_resource_request(request_text: str) -> bool:
    if _matches_any(_PROCESS_SCOPED_RESOURCE_PATTERNS, request_text):
        return True
    return requires_process_resource_payload(request_text) and bool(
        requested_process_resource_names(request_text),
    )


def process_observations_match_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    requested_names = _requested_process_names(request_text)
    requires_resource_payload = requires_process_resource_payload(request_text)
    if requires_resource_payload:
        requested_names = requested_names | requested_process_resource_names(request_text)
    requested_pids = _requested_process_pids(request_text)
    required_resource_metrics = required_process_resource_metrics(request_text)
    return any(
        _is_completed_observation(ref)
        and ref.tool_name == "tool.system.read.process"
        and _process_ref_can_satisfy_request(
            ref,
            requested_names=requested_names,
            requested_pids=requested_pids,
            requires_resource_payload=requires_resource_payload,
            required_resource_metrics=required_resource_metrics,
        )
        for ref in tool_observation_refs
    )


def _process_ref_can_satisfy_request(
    ref: ToolObservationRef,
    *,
    requested_names: frozenset[str],
    requested_pids: frozenset[int],
    requires_resource_payload: bool,
    required_resource_metrics: frozenset[str],
) -> bool:
    if ref.metadata.get("unavailable") is True:
        return False
    schema = ref.structured_schema
    if schema is not None:
        if not isinstance(ref.structured_content, dict) or not _parse_status_is_parsed(ref):
            return False
        if schema == _PROCESS_NAME_SEARCH_SCHEMA:
            return (
                not requires_resource_payload
                and _structured_process_name_search_matches_request(
                    ref.structured_content,
                    requested_names=requested_names,
                    requested_pids=requested_pids,
                    truncated=ref.truncated or ref.metadata.get("stdout_truncated") is True,
                )
            )
        if schema == PROCESS_RESOURCE_SCHEMA:
            return process_resource_payload_matches_request(
                ref.structured_content,
                requested_names=requested_names,
                requested_pids=requested_pids,
                required_resource_metrics=required_resource_metrics,
            )
        return False
    return False


def _structured_process_name_search_matches_request(
    content: dict,
    *,
    requested_names: frozenset[str],
    requested_pids: frozenset[int],
    truncated: bool,
) -> bool:
    if "error" in content:
        return False
    if not requested_names and not requested_pids:
        return False
    raw_matches = content.get("matches")
    if not isinstance(raw_matches, list):
        return False
    matches = tuple(item for item in raw_matches if isinstance(item, dict))
    if matches:
        return any(_process_item_matches_request(item, requested_names, requested_pids) for item in matches)
    if truncated:
        return False
    if requested_pids:
        return False
    query = content.get("query")
    return isinstance(query, str) and _any_name_matches(requested_names, (query,))


def _process_item_matches_request(
    item: dict,
    requested_names: frozenset[str],
    requested_pids: frozenset[int],
) -> bool:
    pid = item.get("pid")
    if requested_pids and (not isinstance(pid, int) or pid not in requested_pids):
        return False
    if requested_names and not _any_name_matches(requested_names, _process_item_names(item)):
        return False
    return True


def _process_item_names(item: dict) -> tuple[str, ...]:
    return tuple(
        value
        for key in ("name", "comm", "process", "process_name", "command_name")
        if isinstance((value := item.get(key)), str)
    )


def _requested_process_names(request_text: str) -> frozenset[str]:
    return _requested_process_names_for_patterns(request_text, _PROCESS_NAME_PATTERNS)


def _requested_process_names_for_patterns(
    request_text: str,
    patterns: tuple[str, ...],
) -> frozenset[str]:
    names: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, request_text, flags=re.IGNORECASE):
            raw_name = match.group("name")
            if re.fullmatch(r"(?:(?:pid|пид|process)\s*)?\d+", raw_name.strip(), flags=re.IGNORECASE):
                continue
            raw_name = re.sub(r"\s+(?:daemon|process|service)\s*$", "", raw_name, flags=re.IGNORECASE)
            name = _normalize_process_name(raw_name)
            if name and name not in _PROCESS_NAME_STOPWORDS:
                names.add(name)
    return frozenset(names)


def _requested_process_pids(request_text: str) -> frozenset[int]:
    pids: set[int] = set()
    for pattern in _PROCESS_PID_PATTERNS:
        for match in re.finditer(pattern, request_text, flags=re.IGNORECASE):
            pids.add(int(match.group("pid")))
    return frozenset(pids)


def _any_name_matches(requested_names: frozenset[str], values: tuple[str, ...] | list[str]) -> bool:
    value_variants = tuple(variant for value in values for variant in _process_name_variants(value))
    return any(
        _process_name_matches(requested, value)
        for requested in requested_names
        for value in value_variants
    )


def _process_name_matches(requested: str, value: str) -> bool:
    return requested == value or any(
        value.startswith(f"{requested}{separator}")
        for separator in ("-", "_", ".")
    )


def _normalize_process_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "", value.casefold())
    return normalized.removesuffix(".app")


def _process_name_variants(value: str) -> frozenset[str]:
    normalized = _normalize_process_name(value)
    variants = {normalized} if normalized else set()
    tokens = [token for token in (_normalize_process_name(part) for part in value.split()) if token]
    if len(tokens) > 1:
        variants.add(tokens[-1])
    first_segment = re.split(r"[-_.]", normalized, maxsplit=1)[0]
    if len(first_segment) >= 3:
        variants.add(first_segment)
    versioned = re.match(r"([a-z]{3,})\d", normalized)
    if versioned:
        variants.add(versioned.group(1))
    return frozenset(variants)


def _parse_status_is_parsed(ref: ToolObservationRef) -> bool:
    return ref.parse_status is ToolParseStatus.PARSED


def _is_completed_observation(ref: ToolObservationRef) -> bool:
    return ref.status in {ToolObservationStatus.COMPLETED, ToolObservationStatus.COMPLETED.value}


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


__all__ = [
    "PROCESS_SHARED_SUFFIX_TOPIC_PATTERNS", "PROCESS_TASK_CONTINUATION_CLAUSE_PATTERNS",
    "PROCESS_TOOL_NAMES", "is_process_scoped_resource_request", "matches_process_live_state_intent",
    "process_observations_match_request",
]
