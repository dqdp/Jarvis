from __future__ import annotations

from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.tools import ToolObservationStatus, ToolParseStatus


def datetime_now_observations_match_request(
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    return any(
        _is_completed_observation(ref)
        and ref.tool_name == "datetime.now"
        and _datetime_now_ref_has_iso(ref)
        for ref in tool_observation_refs
    )


def daemon_status_observations_match_request(
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    return any(
        _is_completed_observation(ref)
        and ref.tool_name == "daemon.status"
        and _daemon_status_ref_has_status(ref)
        for ref in tool_observation_refs
    )


def _datetime_now_ref_has_iso(ref: ToolObservationRef) -> bool:
    if _system_ref_is_unavailable(ref):
        return False
    schema = _system_ref_schema(ref)
    if schema != "datetime.now":
        return False
    return isinstance(ref.structured_content, dict) and isinstance(
        ref.structured_content.get("iso"),
        str,
    ) and ref.parse_status in {ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}


def _daemon_status_ref_has_status(ref: ToolObservationRef) -> bool:
    if _system_ref_is_unavailable(ref):
        return False
    schema = _system_ref_schema(ref)
    if schema != "daemon.status":
        return False
    return isinstance(ref.structured_content, dict) and isinstance(
        ref.structured_content.get("status"),
        str,
    ) and ref.parse_status in {ToolParseStatus.PARSED, ToolParseStatus.PARTIAL}


def _is_completed_observation(ref: ToolObservationRef) -> bool:
    return ref.status in {ToolObservationStatus.COMPLETED, ToolObservationStatus.COMPLETED.value}


def _system_ref_is_unavailable(ref: ToolObservationRef) -> bool:
    return ref.metadata.get("unavailable") is True


def _system_ref_schema(ref: ToolObservationRef) -> str | None:
    if isinstance(ref.structured_schema, str) and ref.structured_schema:
        return ref.structured_schema
    return None


__all__ = [
    "daemon_status_observations_match_request",
    "datetime_now_observations_match_request",
]
