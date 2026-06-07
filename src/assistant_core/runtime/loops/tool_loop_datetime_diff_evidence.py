from __future__ import annotations

from typing import Any

from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.tools import ToolObservationStatus, ToolParseStatus
from assistant_core.runtime.loops.tool_loop_time_delta_units import (
    explicit_time_delta_endpoint_values,
    expected_time_delta_unit,
    normalize_timezone_aware_iso_endpoint_value,
)


def datetime_diff_observations_match_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
    *,
    full_request_text: str | None = None,
    expected_tool_name: str | None = None,
) -> bool:
    return any(
        datetime_diff_observation_matches_request(
            request_text,
            tool_observation_refs,
            ref,
            full_request_text=full_request_text,
            expected_tool_name=expected_tool_name,
        )
        for ref in tool_observation_refs
    )


def datetime_diff_observation_matches_request(
    request_text: str,
    tool_observation_refs: tuple[ToolObservationRef, ...],
    ref: ToolObservationRef,
    *,
    full_request_text: str | None = None,
    expected_tool_name: str | None = None,
) -> bool:
    expected_unit = expected_time_delta_unit(request_text)
    if expected_unit is None:
        return False
    observed_sources = completed_datetime_now_iso_values(tool_observation_refs)
    endpoint_source_text = full_request_text or request_text
    explicit_endpoints = explicit_time_delta_endpoint_values(endpoint_source_text)
    return _datetime_diff_ref_matches(
        ref,
        expected_unit=expected_unit,
        expected_tool_name=expected_tool_name,
        observed_sources=observed_sources,
        explicit_endpoints=explicit_endpoints,
    )


def _datetime_diff_ref_matches(
    ref: ToolObservationRef,
    *,
    expected_unit: str,
    expected_tool_name: str | None,
    observed_sources: frozenset[str],
    explicit_endpoints: tuple[str, ...],
) -> bool:
    if (
        ref.status is not ToolObservationStatus.COMPLETED
        or ref.tool_name not in {"calendar.diff", "datetime.diff"}
        or ref.structured_schema != ref.tool_name
        or ref.parse_status not in {ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}
        or not isinstance(ref.structured_content, dict)
    ):
        return False
    if expected_tool_name is not None and ref.tool_name != expected_tool_name:
        return False
    if ref.structured_content.get("unit") != expected_unit:
        return False
    endpoints = _datetime_diff_endpoint_values(ref.structured_content)
    if len(endpoints) != 2:
        return False
    endpoint_set = frozenset(endpoints)
    if len(explicit_endpoints) >= 2:
        return endpoints == explicit_endpoints[:2]
    if len(explicit_endpoints) == 1:
        return bool(observed_sources) and explicit_endpoints[0] in endpoint_set and bool(
            endpoint_set & observed_sources
        )
    return False


def _datetime_diff_endpoint_values(payload: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for value in (payload.get("from_iso"), payload.get("to_iso")):
        if isinstance(value, str) and value.strip():
            normalized = normalize_timezone_aware_iso_endpoint_value(value)
            if normalized is not None:
                values.append(normalized)
    return tuple(values)


def completed_datetime_now_iso_values(
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> frozenset[str]:
    values: set[str] = set()
    for ref in tool_observation_refs:
        if (
            ref.status is ToolObservationStatus.COMPLETED
            and ref.tool_name == "datetime.now"
            and ref.structured_schema == "datetime.now"
            and ref.parse_status in {ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}
            and isinstance(ref.structured_content, dict)
        ):
            iso_value = ref.structured_content.get("iso")
            if isinstance(iso_value, str) and iso_value.strip():
                normalized = normalize_timezone_aware_iso_endpoint_value(iso_value)
                if normalized is not None:
                    values.add(normalized)
    return frozenset(values)


__all__ = [
    "completed_datetime_now_iso_values",
    "datetime_diff_observation_matches_request",
    "datetime_diff_observations_match_request",
    "expected_time_delta_unit",
]
