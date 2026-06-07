from __future__ import annotations

import json

from assistant_core.domain.loops import ToolObservationRef


def system_ref_argv(ref: ToolObservationRef) -> tuple[str, ...]:
    argv = ref.arguments.get("argv")
    if not isinstance(argv, (list, tuple)):
        return ()
    if not all(isinstance(arg, str) for arg in argv):
        return ()
    return tuple(argv)


def system_ref_argv_command(ref: ToolObservationRef) -> str | None:
    argv = system_ref_argv(ref)
    return argv[0] if argv else None


def system_ref_has_usable_raw_diagnostics(ref: ToolObservationRef) -> bool:
    if system_ref_is_unavailable(ref):
        return False
    if not system_ref_argv(ref):
        return False
    if not system_ref_has_raw_diagnostic_payload(ref):
        return False
    exit_code = ref.metadata.get("exit_code")
    if isinstance(exit_code, int):
        return exit_code == 0
    content_exit_code = system_ref_content_exit_code(ref)
    if content_exit_code is not None:
        return content_exit_code == 0
    return False


def system_ref_has_raw_diagnostic_payload(ref: ToolObservationRef) -> bool:
    if ref.content_type == "application/json":
        try:
            content = json.loads(ref.content)
        except (TypeError, ValueError):
            return False
        return raw_diagnostic_payload_has_data(content)
    return isinstance(ref.content, str) and bool(ref.content.strip())


def raw_diagnostic_payload_has_data(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in ("stdout", "stderr")
    )


def system_ref_content_exit_code(ref: ToolObservationRef) -> int | None:
    if ref.content_type != "application/json":
        return None
    try:
        content = json.loads(ref.content)
    except (TypeError, ValueError):
        return None
    if not isinstance(content, dict):
        return None
    exit_code = content.get("exit_code")
    return exit_code if isinstance(exit_code, int) else None


def system_ref_is_unavailable(ref: ToolObservationRef) -> bool:
    return ref.metadata.get("unavailable") is True
