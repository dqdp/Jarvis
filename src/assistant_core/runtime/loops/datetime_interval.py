from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from assistant_core.domain.loops import LoopExecutionRequest, ToolObservationRef, ToolProposal, ToolRequestPlan
from assistant_core.domain.tools import ToolObservationStatus


def required_datetime_until_followup(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> ToolProposal | None:
    allowed = request_plan.allowed_tool_names
    if not allowed or not {"datetime.now", "datetime.until"}.issubset(allowed):
        return None
    if not _looks_like_datetime_interval_question(request.user_input):
        return None
    if any(
        ref.tool_name == "datetime.until" and ref.status == ToolObservationStatus.COMPLETED
        for ref in tool_observation_refs
    ):
        return None
    datetime_now_ref = next(
        (
            ref
            for ref in reversed(tool_observation_refs)
            if ref.tool_name == "datetime.now" and ref.status == ToolObservationStatus.COMPLETED
        ),
        None,
    )
    from_iso = (
        _datetime_now_observation_iso(datetime_now_ref)
        if datetime_now_ref is not None
        else None
    )
    target = _datetime_until_target(request.user_input)
    if target is None:
        return None
    arguments = {
        "target": target,
        "unit": _datetime_until_unit(request.user_input),
    }
    if from_iso:
        arguments["from_iso"] = from_iso
    return ToolProposal(
        action="tool_call",
        tool_name="datetime.until",
        arguments=arguments,
    )


def datetime_until_deterministic_response(
    request: LoopExecutionRequest,
    observation_ref: ToolObservationRef,
) -> str | None:
    if observation_ref.tool_name != "datetime.until":
        return None
    if observation_ref.status != ToolObservationStatus.COMPLETED:
        return None
    if not _looks_like_datetime_interval_question(request.user_input):
        return None
    payload = _tool_observation_payload(observation_ref)
    if payload is None or payload.get("target") != "next_new_year":
        return None
    unit = payload.get("unit")
    value = payload.get("value")
    if unit == "seconds":
        value = payload.get("seconds", value)
    if not isinstance(unit, str) or not isinstance(value, int | float):
        return None
    target_year = _datetime_until_target_year(payload.get("target_iso"))
    value_text = _format_interval_value(value)
    unit_text = _russian_interval_unit(unit)
    if target_year is None:
        return f"До Нового года осталось {value_text} {unit_text}."
    return f"До Нового года {target_year} года осталось {value_text} {unit_text}."


def _looks_like_datetime_interval_question(user_input: str) -> bool:
    text = user_input.casefold()
    if _datetime_until_target(text) is None:
        return False
    return any(
        marker in text
        for marker in (
            "до",
            "через",
            "остал",
            "сколько",
            "until",
            "countdown",
            "remain",
        )
    )


def _datetime_now_observation_iso(observation_ref: ToolObservationRef) -> str | None:
    payload = _tool_observation_payload(observation_ref)
    if payload is None:
        return None
    value = payload.get("iso")
    if isinstance(value, str) and value:
        return value
    return None


def _tool_observation_payload(observation_ref: ToolObservationRef) -> dict[str, Any] | None:
    if isinstance(observation_ref.structured_content, dict):
        return observation_ref.structured_content
    try:
        payload = json.loads(observation_ref.content)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _datetime_until_target(user_input: str) -> str | None:
    text = user_input.casefold()
    if (
        "нового года" in text
        or "новый год" in text
        or "new year" in text
        or "new year's" in text
    ):
        return "next_new_year"
    return None


def _datetime_until_unit(user_input: str) -> str:
    text = user_input.casefold()
    if "сек" in text or "second" in text:
        return "seconds"
    if "мин" in text or "minute" in text:
        return "minutes"
    if "час" in text or "hour" in text:
        return "hours"
    if "дн" in text or "day" in text:
        return "days"
    return "seconds"


def _datetime_until_target_year(value: Any) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).year
    except ValueError:
        return None


def _format_interval_value(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}".replace(",", " ")
    return f"{int(value):,}".replace(",", " ")


def _russian_interval_unit(unit: str) -> str:
    return {
        "seconds": "секунд",
        "minutes": "минут",
        "hours": "часов",
        "days": "дней",
    }.get(unit, unit)
