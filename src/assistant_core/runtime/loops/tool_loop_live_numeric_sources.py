from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.tools import ToolObservationStatus, ToolParseStatus
from assistant_core.runtime.loops.tool_loop_datetime_diff_evidence import (
    datetime_diff_observation_matches_request,
)
from assistant_core.runtime.loops.tool_loop_live_state_tools import (
    is_known_live_state_tool_name,
)
from assistant_core.runtime.loops.tool_loop_numeric_literals import (
    number_literal_variants,
    number_literals_from_text,
)
from assistant_core.runtime.loops.tool_loop_process_resource_evidence import (
    process_resource_items_matching_request,
    requested_process_resource_names,
    requested_process_resource_pids,
)
from assistant_core.runtime.loops.tool_loop_time_delta_units import (
    GENERIC_TIME_DELTA_UNIT_PATTERN,
    TIME_DELTA_UNIT_PATTERN,
    time_delta_unit_source,
)


_CPU_PATTERNS = (r"\b(?:cpu|processor)\b", r"(?:процессор|цп)")
_MEMORY_PATTERNS = (r"\b(?:memory|ram)\b", r"(?:памят|оператив)")
_DISK_PATTERNS = (r"\b(?:disk|storage)\b", r"(?:диск|накопител|хранилищ)")
_BATTERY_PATTERNS = (r"\bbattery\b", r"(?:батар|аккумулятор)")
_NETWORK_PATTERNS = (r"\bnetwork\b", r"(?:сет\w*)")
_TIME_DELTA_PATTERNS = (
    rf"(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN}).*(?:до\b|остав|остал)",
    rf"(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN}).*(?:прошед|прошл).*(?:\bс\b|\bсо\b)",
    rf"(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN}).*(?:\bмежду\b|\bbetween\b)",
    rf"\b(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN})\b.*\b(?:until|remaining|left)\b",
    rf"\b(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN})\b.*\b(?:elapsed|passed)\b.*\b(?:since|from)\b",
)

_CPU_USAGE_KEYS = frozenset({"cpu", "cpu_percent", "cpu_usage", "cpu_usage_percent", "used_percent"})
_CPU_LOAD_KEYS = _CPU_USAGE_KEYS | frozenset({"load", "load_1m", "load_5m", "load_15m", "load_average", "load_percent"})
_CPU_IDLE_KEYS = frozenset({"idle_percent"})
_CPU_ACTIVITY_KEYS = frozenset({"system_percent", "user_percent"})
_CPU_KEYS = _CPU_USAGE_KEYS | _CPU_LOAD_KEYS | _CPU_IDLE_KEYS | _CPU_ACTIVITY_KEYS
_MEMORY_USED_KEYS = frozenset({
    "memory",
    "memory_percent",
    "memory_usage",
    "memory_usage_percent",
    "rss",
    "rss_bytes",
    "swap_used",
    "used",
    "used_bytes",
    "used_percent",
})
_MEMORY_AVAILABLE_KEYS = frozenset({"available", "available_bytes", "available_percent", "free", "free_bytes"})
_MEMORY_TOTAL_KEYS = frozenset({"total", "total_bytes"})
_MEMORY_KEYS = _MEMORY_USED_KEYS | _MEMORY_AVAILABLE_KEYS | _MEMORY_TOTAL_KEYS
_DISK_USED_KEYS = frozenset({"used", "used_bytes", "used_percent_value"})
_DISK_AVAILABLE_KEYS = frozenset({"available", "available_bytes", "available_percent", "free", "free_bytes"})
_DISK_KEYS = _DISK_USED_KEYS | _DISK_AVAILABLE_KEYS
_BATTERY_KEYS = frozenset({"percent", "percentage", "level", "charge_percent"})
_NETWORK_KEYS = frozenset({"bytes_in", "bytes_out", "packets_in", "packets_out", "rx_bytes", "tx_bytes"})
_PROCESS_KEYS = _CPU_KEYS | _MEMORY_KEYS
_TIME_DELTA_KEYS = frozenset({"value", "microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "weeks", "months", "quarters", "decades"})
_USED_PATTERNS = (r"\b(?:usage|used|utili[sz]ation|percent|percentage)\b", r"(?:занят|использ|процент)")
_AVAILABLE_PATTERNS = (r"\b(?:available|free)\b", r"(?:свобод|доступ)")
_LOAD_PATTERNS = (r"\bload\b", r"нагруз")


@dataclass(frozen=True)
class LiveNumericOperandGroup:
    scope: str
    values: frozenset[str]


def completed_live_numeric_literals_for_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> frozenset[str]:
    groups = completed_live_numeric_operand_groups_for_request(request_text, tool_observation_refs)
    return frozenset(value for group in groups for value in group.values)


def completed_live_numeric_operand_groups_for_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> tuple[LiveNumericOperandGroup, ...]:
    required_sources = _request_numeric_sources(request_text)
    if not required_sources:
        return ()
    groups: list[LiveNumericOperandGroup] = []
    for source in sorted(required_sources):
        values: set[str] = set()
        for ref in tool_observation_refs:
            if not _is_completed(ref) or not _is_live_tool_name(ref.tool_name):
                continue
            for item in _iter_source_numeric_values(
                ref,
                source,
                request_text,
                tool_observation_refs,
            ):
                values.update(number_literal_variants(item))
        if values:
            groups.append(LiveNumericOperandGroup(scope=source, values=frozenset(values)))
    return tuple(groups)


def _request_numeric_sources(request_text: str) -> frozenset[str]:
    normalized = _normalize(request_text)
    scopes: set[str] = set()
    if _matches_any(_TIME_DELTA_PATTERNS, normalized):
        source = _time_delta_source(normalized)
        if source is not None:
            scopes.add(source)
    if _matches_any(_CPU_PATTERNS, normalized):
        scopes.add("cpu.load" if _matches_any(_LOAD_PATTERNS, normalized) else "cpu.used")
    if _matches_any(_MEMORY_PATTERNS, normalized):
        scopes.add("memory.available" if _matches_any(_AVAILABLE_PATTERNS, normalized) else "memory.used")
    if _matches_any(_DISK_PATTERNS, normalized):
        scopes.add("disk.available" if _matches_any(_AVAILABLE_PATTERNS, normalized) else "disk.used")
    if _matches_any(_BATTERY_PATTERNS, normalized):
        scopes.add("battery.percent")
    if _matches_any(_NETWORK_PATTERNS, normalized):
        scopes.add("network.traffic")
    return frozenset(scopes)


def _time_delta_source(normalized: str) -> str | None:
    return time_delta_unit_source(normalized)


def _iter_source_numeric_values(
    ref: ToolObservationRef,
    source: str,
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> Iterator[int | float]:
    content = ref.structured_content
    if not isinstance(content, dict) or not _has_parsed_typed_payload(ref):
        return
    schema = ref.structured_schema
    base_source = source.split(".", maxsplit=1)[0]
    if schema in {"calendar.diff", "datetime.diff"}:
        if not datetime_diff_observation_matches_request(
            request_text,
            tool_observation_refs,
            ref,
            full_request_text=request_text,
            expected_tool_name=ref.tool_name,
        ):
            return
        if base_source == "time_delta":
            yield from _iter_time_delta_numbers(content, source)
        return
    if schema == "datetime.until":
        if base_source == "time_delta":
            yield from _iter_time_delta_numbers(content, source)
        return
    if schema == "system.resource_overview":
        yield from _iter_resource_overview_values(content, source)
        return
    if schema == "system.cpu_overview" and base_source == "cpu":
        yield from _iter_matching_key_numbers(content, _keys_for_source(source))
        return
    if schema == "system.memory_overview" and base_source == "memory":
        yield from _iter_matching_key_numbers(content, _keys_for_source(source))
        return
    if schema == "system.disk_free" and base_source == "disk":
        yield from _iter_matching_key_numbers(content, _keys_for_source(source))
        return
    if schema == "system.battery_charge" and base_source == "battery":
        yield from _iter_matching_key_numbers(content, _BATTERY_KEYS)
        return
    if schema == "system.network_interfaces" and base_source == "network":
        yield from _iter_matching_key_numbers(content, _NETWORK_KEYS)
        return
    if schema == "system.process_resource_snapshot":
        yield from _iter_process_resource_values(content, source, request_text)


def _iter_resource_overview_values(
    content: dict,
    source: str,
) -> Iterator[int | float]:
    if source.startswith("cpu.") and isinstance(content.get("cpu"), dict):
        yield from _iter_matching_key_numbers(content["cpu"], _keys_for_source(source))
    if source.startswith("memory.") and isinstance(content.get("memory"), dict):
        yield from _iter_matching_key_numbers(content["memory"], _keys_for_source(source))
    if source.startswith("disk.") and isinstance(content.get("disk"), dict):
        yield from _iter_matching_key_numbers(content["disk"], _keys_for_source(source))


def _iter_process_resource_values(
    content: dict,
    source: str,
    request_text: str,
) -> Iterator[int | float]:
    values = process_resource_items_matching_request(
        content,
        requested_names=requested_process_resource_names(request_text),
        requested_pids=requested_process_resource_pids(request_text),
    )
    keys = _keys_for_source(source) if source.startswith(("cpu.", "memory.")) else frozenset()
    for item in values:
        yield from _iter_matching_key_numbers(item, keys)


def _iter_time_delta_numbers(content: dict, source: str) -> Iterator[int | float]:
    unit = source.removeprefix("time_delta.")
    keys = {unit}
    if content.get("unit") == unit:
        keys.add("value")
    yield from _iter_matching_key_numbers(content, frozenset(keys & _TIME_DELTA_KEYS))


def _keys_for_source(source: str) -> frozenset[str]:
    if source == "cpu.used":
        return _CPU_USAGE_KEYS
    if source == "cpu.load":
        return _CPU_LOAD_KEYS
    if source == "memory.used":
        return _MEMORY_USED_KEYS
    if source == "memory.available":
        return _MEMORY_AVAILABLE_KEYS
    if source == "disk.used":
        return _DISK_USED_KEYS
    if source == "disk.available":
        return _DISK_AVAILABLE_KEYS
    return frozenset()


def _has_parsed_typed_payload(ref: ToolObservationRef) -> bool:
    return ref.parse_status in {ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}


def _iter_matching_key_numbers(value: object, allowed_keys: frozenset[str]) -> Iterator[int | float]:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in allowed_keys:
                yield from _iter_numeric_values(item)
            elif isinstance(item, dict | list | tuple):
                yield from _iter_matching_key_numbers(item, allowed_keys)
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _iter_matching_key_numbers(item, allowed_keys)


def _iter_numeric_values(value: object) -> Iterator[int | float]:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_numeric_values(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _iter_numeric_values(item)


def _is_completed(ref: ToolObservationRef) -> bool:
    return ref.status in {ToolObservationStatus.COMPLETED, ToolObservationStatus.COMPLETED.value}


def _is_live_tool_name(tool_name: str) -> bool:
    return is_known_live_state_tool_name(tool_name)


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _normalize(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"(?<!\d),|,(?!\d)|[:;]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


__all__ = [
    "LiveNumericOperandGroup", "completed_live_numeric_literals_for_request",
    "completed_live_numeric_operand_groups_for_request", "number_literals_from_text",
    "number_literal_variants",
]
