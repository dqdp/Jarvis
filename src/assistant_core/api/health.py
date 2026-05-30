from __future__ import annotations

from typing import Any


async def health_payload(
    conversation_store,
    memory_store,
    *,
    content_store=None,
    inference_health=None,
) -> dict[str, Any]:
    checks: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for name, component in (
        ("conversation_store", conversation_store),
        ("memory_store", memory_store),
        ("content_store", content_store),
        ("inference", inference_health),
    ):
        if component is None:
            continue
        status, reason = await component_health(component)
        checks[name] = status
        if reason:
            reasons[name] = reason
    ready = all(value == "ok" for value in checks.values())
    return {
        "status": "ready" if ready else "not_ready",
        "liveness": {"status": "ok"},
        "readiness": {
            "status": "ok" if ready else "failed",
            "checks": checks,
            "reasons": reasons,
        },
    }


async def component_health(component) -> tuple[str, str | None]:
    health_status = getattr(component, "health_status", None)
    if health_status is not None:
        try:
            result = health_status()
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:
            return "failed", type(exc).__name__
        if isinstance(result, dict):
            status = result.get("status")
            reason = result.get("reason")
            if status in {"ok", "ready"}:
                return "ok", None
            return "failed", str(reason) if reason else None

    health_check = getattr(component, "health_check", None)
    if health_check is None:
        return "ok", None
    try:
        result = health_check()
        if hasattr(result, "__await__"):
            result = await result
    except Exception as exc:
        return "failed", type(exc).__name__
    return ("ok", None) if result else ("failed", "health check returned false")
