from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from assistant_core.config.settings import Settings


def conversation_payload(conversation) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "title": conversation.title,
        "active_project_namespace": conversation.active_project_namespace,
        "status": conversation.status.value,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "metadata": conversation.metadata,
    }


def message_payload(message) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "conversation_id": message.conversation_id,
        "request_id": message.request_id,
        "role": message.role.value,
        "content": message.content,
        "sensitivity": message.sensitivity.value,
        "created_at": message.created_at,
        "metadata": message.metadata,
    }


def request_payload(request) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "conversation_id": request.conversation_id,
        "user_message_id": request.user_message_id,
        "assistant_message_id": request.assistant_message_id,
        "status": request.status.value,
        "created_at": request.created_at,
        "started_at": request.started_at,
        "completed_at": request.completed_at,
        "error": (
            None
            if request.error_code is None
            else {"code": request.error_code, "message": request.error_message}
        ),
    }


def approval_payload(approval) -> dict[str, Any]:
    return {
        "approval_id": approval.approval_id,
        "status": approval.status.value,
        "capability": approval.capability.value,
        "risk_classes": sorted(risk.value for risk in approval.risk_classes),
        "scope": {
            "tool_name": approval.scope.tool_name,
            "request_id": approval.scope.request_id,
            "conversation_id": approval.scope.conversation_id,
            "step_id": approval.scope.step_id,
            "project_namespace": approval.scope.project_namespace,
            "working_directory": redacted_scope_value(approval.scope.working_directory),
            "sensitivity": approval.scope.sensitivity.value,
            "permission_mode": approval.scope.permission_mode,
            "argument_keys": list(approval.scope.argument_keys),
        },
        "redacted_payload": approval.redacted_payload,
        "created_at": approval.created_at,
        "expires_at": approval.expires_at,
        "granted_at": approval.granted_at,
        "denied_at": approval.denied_at,
        "cancelled_at": approval.cancelled_at,
        "used_at": approval.used_at,
    }


def redacted_scope_value(value: str | None) -> str | None:
    if value is None:
        return None
    return "<redacted>"


def memory_payload(memory) -> dict[str, Any]:
    return {
        "memory_id": memory.id,
        "namespace": memory.namespace,
        "memory_type": memory.memory_type.value,
        "content": memory.content,
        "summary": memory.summary,
        "content_hash": memory.content_hash,
        "sensitivity": memory.sensitivity.value,
        "status": memory.status.value,
        "indexing_status": memory.indexing_status.value,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "archived_at": memory.archived_at,
        "archive_reason": memory.archive_reason,
        "metadata": memory.metadata,
    }


def memory_lifecycle_payload(memory) -> dict[str, Any]:
    return {
        "memory_id": memory.id,
        "namespace": memory.namespace,
        "memory_type": memory.memory_type.value,
        "sensitivity": memory.sensitivity.value,
        "status": memory.status.value,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "archived_at": memory.archived_at,
        "archive_reason": memory.archive_reason,
    }


def content_ingestion_payload(result) -> dict[str, Any]:
    return {
        "seen_sources": result.seen_sources,
        "created_sources": result.created_sources,
        "updated_sources": result.updated_sources,
        "deleted_sources": result.deleted_sources,
        "created_chunks": result.created_chunks,
        "stale_chunks": result.stale_chunks,
        "deleted_chunks": result.deleted_chunks,
    }


def content_source_payload(source) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_type": source.source_type.value,
        "path": source.path.as_posix(),
        "uri": source.uri,
        "title": source.title,
        "content_hash": source.content_hash,
        "status": source.status.value,
        "sensitivity": source.sensitivity.value,
        "last_seen_at": source.last_seen_at,
        "indexed_at": source.indexed_at,
        "metadata": source.metadata,
    }


def content_status_summary(records) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for record in records:
        status = record.status.value
        by_status[status] = by_status.get(status, 0) + 1
    return {"total": len(records), "by_status": by_status}


def runtime_status_payload(settings: Settings) -> dict[str, Any]:
    return {
        "status": "ready",
        "default_model_profile": "local_main",
        "model_profiles": {
            name: {
                "purpose": profile.purpose,
                "provider": profile.provider,
                "enabled": profile.enabled,
                "cloud": profile.cloud,
                "model": profile.model,
                "endpoint": public_endpoint(profile.endpoint),
                "max_input_tokens": profile.max_input_tokens,
                "max_output_tokens": profile.max_output_tokens,
                "temperature": profile.temperature,
                "supports_streaming": profile.supports_streaming,
            }
            for name, profile in settings.model_profiles.items()
        },
        "runtime_budgets": {
            name: {
                "max_model_calls": budget.max_model_calls,
                "max_tool_calls": budget.max_tool_calls,
                "max_wall_time_seconds": budget.max_wall_time_seconds,
                "max_output_tokens": budget.max_output_tokens,
                "allow_cloud": budget.allow_cloud,
                "allow_tools": budget.allow_tools,
                "allow_autonomous_memory_write": budget.allow_autonomous_memory_write,
            }
            for name, budget in settings.runtime_budgets.items()
        },
    }


def public_endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    parsed = urlsplit(endpoint)
    if not parsed.netloc:
        return endpoint
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
