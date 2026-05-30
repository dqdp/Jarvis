from __future__ import annotations

import hashlib
import json
from typing import Any

from assistant_core.domain.approvals import ApprovalScope
from assistant_core.domain.tools import ToolCallRequest, ToolSpec


def approval_scope(spec: ToolSpec, request: ToolCallRequest) -> ApprovalScope:
    return ApprovalScope(
        capability=spec.capability,
        risk_classes=spec.risk_classes,
        tool_name=spec.name,
        user_id=request.user_id,
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        step_id=request.step_id,
        project_namespace=request.project_namespace,
        working_directory=request.working_directory,
        sensitivity=request.sensitivity,
        permission_mode=request.permission_mode,
        argument_keys=tuple(sorted(request.arguments)),
        arguments_hash=arguments_hash(request.arguments),
    )


def approval_payload(spec: ToolSpec, request: ToolCallRequest) -> dict[str, Any]:
    return {
        "tool_name": spec.name,
        "capability": spec.capability.value,
        "risk_classes": sorted(risk.value for risk in spec.risk_classes),
        "argument_keys": sorted(request.arguments),
    }


def arguments_hash(arguments: dict[str, Any]) -> str:
    encoded = json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
