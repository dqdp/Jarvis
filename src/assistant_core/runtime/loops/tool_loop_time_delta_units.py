from __future__ import annotations

from datetime import datetime
import re


DATETIME_DIFF_UNITS = frozenset(
    {"microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "weeks"}
)
CALENDAR_DIFF_UNITS = frozenset(
    {*DATETIME_DIFF_UNITS, "months", "quarters", "decades"}
)
CALENDAR_ONLY_DIFF_UNITS = CALENDAR_DIFF_UNITS - DATETIME_DIFF_UNITS
TIME_DELTA_UNIT_PATTERN = (
    r"(?:"
    r"микросекунд\w*|миллисекунд\w*|секунд\w*|минут\w*|час(?:ов|а|ы)?|"
    r"дн(?:ей|я|и|ю|ем|ям|ями|ях)?|недел\w*|месяц\w*|квартал\w*|декад\w*|"
    r"microseconds?|usecs?|us|µs|milliseconds?|msecs?|ms|seconds?|secs?|minutes?|mins?|"
    r"hours?|hrs?|days?|weeks?|months?|quarters?|decades?"
    r")"
)
GENERIC_TIME_DELTA_UNIT_PATTERN = r"(?:единиц\w*\s+времени|time\s+units?)"

_UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("microseconds", r"микросекунд\w*|microseconds?|usecs?|\bus\b|µs"),
    ("milliseconds", r"миллисекунд\w*|milliseconds?|msecs?|\bms\b"),
    ("seconds", r"секунд\w*|seconds?|secs?"),
    ("minutes", r"минут\w*|minutes?|mins?"),
    ("hours", r"час(?:ов|а|ы)?|hours?|hrs?"),
    ("days", r"дн(?:ей|я|и|ю|ем|ям|ями|ях)?|days?"),
    ("weeks", r"недел\w*|weeks?"),
    ("months", r"месяц\w*|months?"),
    ("quarters", r"квартал\w*|quarters?"),
    ("decades", r"декад\w*|decades?"),
)
_MEASURED_UNIT_CONTEXT_PATTERNS: tuple[str, ...] = (
    rf"(?:количеств\w*|числ\w*|сколько)\s+(?P<unit>{TIME_DELTA_UNIT_PATTERN})",
    rf"\b(?:how\s+many|number\s+of|amount\s+of|count\s+of|total)\s+"
    rf"(?P<unit>{TIME_DELTA_UNIT_PATTERN})\b",
)
_TIME_DELTA_DIRECTION_PATTERN = re.compile(
    r"(?:прошед|прошл|остав|остал|\bдо\b|\bк\b|"
    r"\buntil\b|\bsince\b|\bfrom\b|\bbetween\b|\belapsed\b|\bpassed\b|"
    r"\bremaining\b|\bleft\b)",
    flags=re.IGNORECASE,
)
_FIXED_INTERVAL_ENDPOINT_PATTERN = re.compile(
    r"(?:\bс\b|\bсо\b).+\bдо\b|\bмежду\b.+\bи\b|\bbetween\b.+\b(?:and|to)\b",
    flags=re.IGNORECASE,
)
_CURRENT_ENDPOINT_PATTERN = re.compile(
    r"сейчас|текущ|в\s+данн\w*\s+момент|"
    r"\b(?:now|current|right\s+now)\b|"
    r"прошед|остав|остал|послед|прошл|следующ|"
    r"\b(?:elapsed|passed|remaining|left|until|last|previous|next)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_INTERVAL_ENDPOINT_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[t\s]\d{2}(?::|\s+)\d{2}(?:(?::|\s+)\d{2}(?:\.\d+)?)?(?:z|[+-]\d{2}(?::|\s+)\d{2})?)?\b",
    flags=re.IGNORECASE,
)


def expected_time_delta_unit(value: str) -> str | None:
    normalized = _normalize(value)
    for pattern in _MEASURED_UNIT_CONTEXT_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            unit = _unit_name_for_text(match.group("unit"))
            if unit is not None:
                return unit
    direction = _TIME_DELTA_DIRECTION_PATTERN.search(normalized)
    before_direction = normalized[: direction.start()] if direction else normalized
    unit = _first_unit_name_by_position(before_direction)
    if unit is not None:
        return unit
    return _single_unit_name(normalized)


def time_delta_unit_source(value: str) -> str | None:
    unit = expected_time_delta_unit(value)
    return f"time_delta.{unit}" if unit is not None else None


def explicit_time_delta_endpoint_values(value: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _EXPLICIT_INTERVAL_ENDPOINT_PATTERN.finditer(value):
        normalized = normalize_timezone_aware_iso_endpoint_value(match.group(0))
        if normalized is not None:
            values.append(normalized)
    return tuple(values)


def is_self_contained_time_delta_interval(value: str) -> bool:
    normalized = _normalize(value)
    if expected_time_delta_unit(normalized) is None or _FIXED_INTERVAL_ENDPOINT_PATTERN.search(normalized) is None:
        return False
    if len(_EXPLICIT_INTERVAL_ENDPOINT_PATTERN.findall(normalized)) >= 2:
        return True
    return _CURRENT_ENDPOINT_PATTERN.search(normalized) is None


def _normalize(value: str) -> str:
    lowered = value.casefold()
    lowered = re.sub(r"(?<!\d),|,(?!\d)|[:;]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_iso_endpoint_value(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return value
    return parsed.isoformat()


def normalize_timezone_aware_iso_endpoint_value(value: str) -> str | None:
    normalized = normalize_iso_endpoint_value(value)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.isoformat()


def _unit_name_for_text(value: str) -> str | None:
    for unit, pattern in _UNIT_PATTERNS:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return unit
    return None


def _first_unit_name_by_position(value: str) -> str | None:
    matches = sorted(
        (
            (match.start(), unit)
            for unit, pattern in _UNIT_PATTERNS
            for match in re.finditer(pattern, value, flags=re.IGNORECASE)
        ),
        key=lambda item: item[0],
    )
    return matches[0][1] if matches else None


def _single_unit_name(value: str) -> str | None:
    units = {
        unit
        for unit, pattern in _UNIT_PATTERNS
        if re.search(pattern, value, flags=re.IGNORECASE)
    }
    if len(units) == 1:
        return next(iter(units))
    return None


__all__ = [
    "CALENDAR_DIFF_UNITS",
    "CALENDAR_ONLY_DIFF_UNITS",
    "DATETIME_DIFF_UNITS",
    "GENERIC_TIME_DELTA_UNIT_PATTERN",
    "TIME_DELTA_UNIT_PATTERN",
    "explicit_time_delta_endpoint_values",
    "expected_time_delta_unit",
    "is_self_contained_time_delta_interval",
    "normalize_iso_endpoint_value",
    "normalize_timezone_aware_iso_endpoint_value",
    "time_delta_unit_source",
]
