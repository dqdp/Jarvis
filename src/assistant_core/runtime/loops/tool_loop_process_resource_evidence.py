from __future__ import annotations

import re


PROCESS_RESOURCE_SCHEMA = "system.process_resource_snapshot"
_PROCESS_RESOURCE_PATTERNS: tuple[str, ...] = (
    r"\b(?:cpu|processor|memory|ram|load|usage|utili[sz]ation|percent|%)\b.*\b(?:process|service|pid)\b",
    r"\b(?:process|service|pid)\b.*\b(?:cpu|processor|memory|ram|load|usage|utili[sz]ation|percent|%)\b",
    r"(?:процесс(?!ор)|служб|пид).*(?:cpu|процессор|памят|нагруз|использ|утилизац)",
    r"(?:cpu|процессор|памят|нагруз|использ|утилизац).*(?:процесс(?!ор)|служб|пид)",
)
_PROCESS_NAME_WORD = r"(?!(?:process|service|app|application|cpu|processor|memory|ram|usage)\b)[\w.-]{2,}"
_PROCESS_NAME_TOKEN = rf"{_PROCESS_NAME_WORD}(?:\s+{_PROCESS_NAME_WORD}){{0,3}}"
_PROCESS_SUBJECT_PREFIX = r"\b(?:the\s+|my\s+)?(?!(?:what|how|is|are|compare|current)\b)"
_PROCESS_RESOURCE_NAME_PATTERNS: tuple[str, ...] = (
    rf"\b(?:cpu|processor|memory|ram|load|usage|utili[sz]ation|percent|%)\b.{{0,80}}\b(?:of|for|by)\s+(?:the\s+|my\s+)?(?P<name>{_PROCESS_NAME_TOKEN})(?:\s+(?:process|service|app|application))?\b",
    rf"{_PROCESS_SUBJECT_PREFIX}(?P<name>{_PROCESS_NAME_TOKEN})(?:\s+(?:process|service|app|application))?\s+(?:cpu|processor|memory|ram)\s+(?:usage|load|utili[sz]ation|percent|%)\b",
    rf"\b(?:how\s+much\s+)?(?:cpu|processor|memory|ram)\s+(?:is|are)\s+(?:the\s+|my\s+)?(?P<name>{_PROCESS_NAME_TOKEN})\s+using\b",
    rf"\b(?:is|are)\s+(?:the\s+|my\s+)?(?P<name>{_PROCESS_NAME_TOKEN})\s+using\b.*\b(?:cpu|processor|memory|ram)\b",
)
_PROCESS_CPU_RESOURCE_PATTERNS: tuple[str, ...] = (r"\b(?:cpu|processor|load|utili[sz]ation)\b", r"\b(?:процессор|цп|нагруз|утилизац)\b")
_PROCESS_MEMORY_RESOURCE_PATTERNS: tuple[str, ...] = (r"\b(?:memory|ram)\b", r"\b(?:памят|оператив)\b")
_PROCESS_RESOURCE_METRIC_PATTERNS: tuple[str, ...] = (
    *_PROCESS_CPU_RESOURCE_PATTERNS,
    *_PROCESS_MEMORY_RESOURCE_PATTERNS,
    r"\b(?:usage|percent|%)\b",
    r"\b(?:использ|процент)\b",
)
_PROCESS_RESOURCE_PID_PATTERNS: tuple[str, ...] = (
    r"\bpid\s+(?P<pid>\d+)\b", r"\bprocess\s+(?P<pid>\d+)\b", r"\bпид\s+(?P<pid>\d+)\b",
)
_PROCESS_CPU_KEYS = frozenset({"cpu", "cpu_percent", "cpu_usage", "cpu_usage_percent", "load_percent"})
_PROCESS_MEMORY_KEYS = frozenset({"memory", "memory_percent", "memory_usage", "memory_usage_percent", "rss", "rss_bytes"})
_PROCESS_NAME_STOPWORDS = frozenset(
    {
        "add", "and", "are", "average", "avg", "battery", "computer", "cpu", "current", "daemon", "device", "disk",
        "divide", "divided", "global", "how", "is", "laptop", "load", "local", "mac", "macbook",
        "machine", "mean", "memory", "minus", "much", "multiply", "my", "network", "or", "overall",
        "of", "percent", "plus", "process", "processor", "ram", "runtime", "service", "show", "subtract",
        "sum", "system", "than", "the", "times", "to", "total", "usage", "vpn", "what",
    },
)
_NON_PROCESS_RESOURCE_NAME_TOKENS = frozenset({
    "add", "and", "average", "avg", "calculate", "compute", "derive", "divide",
    "divided", "evaluate", "find", "mean", "minus", "multiply", "or", "percent",
    "greater", "of", "plus", "subtract", "sum", "than", "times", "to",
})
_LIVE_SUFFIX_PATTERN = re.compile(r"\s+(?:right\s+now|now|currently|сейчас|в\s+данн\w*\s+момент)\s*$", re.IGNORECASE)


def requires_process_resource_payload(request_text: str) -> bool:
    return _matches_any(_PROCESS_RESOURCE_PATTERNS, request_text) or (
        _matches_any(_PROCESS_RESOURCE_METRIC_PATTERNS, request_text)
        and bool(requested_process_resource_names(request_text))
    )


def required_process_resource_metrics(request_text: str) -> frozenset[str]:
    metrics: set[str] = set()
    if _matches_any(_PROCESS_CPU_RESOURCE_PATTERNS, request_text):
        metrics.add("cpu")
    if _matches_any(_PROCESS_MEMORY_RESOURCE_PATTERNS, request_text):
        metrics.add("memory")
    return frozenset(metrics)


def requested_process_resource_names(request_text: str) -> frozenset[str]:
    names: set[str] = set()
    for pattern in _PROCESS_RESOURCE_NAME_PATTERNS:
        for match in re.finditer(pattern, request_text, flags=re.IGNORECASE):
            raw_name = _LIVE_SUFFIX_PATTERN.sub("", match.group("name"))
            name = _normalize_process_name(raw_name)
            if _is_valid_process_resource_name(raw_name, name):
                names.add(name)
    return frozenset(names)


def requested_process_resource_pids(request_text: str) -> frozenset[int]:
    pids: set[int] = set()
    for pattern in _PROCESS_RESOURCE_PID_PATTERNS:
        for match in re.finditer(pattern, request_text, flags=re.IGNORECASE):
            pids.add(int(match.group("pid")))
    return frozenset(pids)


def _is_valid_process_resource_name(raw_name: str, name: str) -> bool:
    tokens = frozenset(
        token
        for token in (_normalize_process_name(part) for part in raw_name.split())
        if token
    )
    if not name or name in _PROCESS_NAME_STOPWORDS or not tokens:
        return False
    if tokens.issubset(_PROCESS_NAME_STOPWORDS):
        return False
    if tokens.intersection(_NON_PROCESS_RESOURCE_NAME_TOKENS):
        return False
    if any(token.isdecimal() for token in tokens):
        return False
    return True


def process_resource_payload_matches_request(
    content: dict,
    *,
    requested_names: frozenset[str],
    requested_pids: frozenset[int],
    required_resource_metrics: frozenset[str],
) -> bool:
    processes = _process_items(content, "processes") or _process_items(content, "matches")
    return any(
        _process_item_matches_request(item, requested_names, requested_pids)
        and (
            not required_resource_metrics
            or _process_item_has_resource_metrics(item, required_resource_metrics)
        )
        for item in processes
    )


def process_resource_items_matching_request(
    content: dict,
    *,
    requested_names: frozenset[str],
    requested_pids: frozenset[int],
) -> tuple[dict, ...]:
    processes = _process_items(content, "processes") or _process_items(content, "matches")
    if not requested_names and not requested_pids:
        return ()
    return tuple(item for item in processes if _process_item_matches_request(item, requested_names, requested_pids))


def _process_items(content: dict, key: str) -> tuple[dict, ...]:
    values = content.get(key)
    if not isinstance(values, list):
        return ()
    return tuple(item for item in values if isinstance(item, dict))


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


def _process_item_has_resource_metrics(item: dict, required_metrics: frozenset[str]) -> bool:
    keys = frozenset(key for key, value in item.items() if value is not None)
    if "cpu" in required_metrics and not keys.intersection(_PROCESS_CPU_KEYS):
        return False
    if "memory" in required_metrics and not keys.intersection(_PROCESS_MEMORY_KEYS):
        return False
    return bool(required_metrics) or bool(keys.intersection(_PROCESS_CPU_KEYS | _PROCESS_MEMORY_KEYS))


def _any_name_matches(requested_names: frozenset[str], values: tuple[str, ...]) -> bool:
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


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


__all__ = [
    "PROCESS_RESOURCE_SCHEMA", "process_resource_payload_matches_request",
    "process_resource_items_matching_request", "requested_process_resource_names",
    "requested_process_resource_pids", "required_process_resource_metrics",
    "requires_process_resource_payload",
]
