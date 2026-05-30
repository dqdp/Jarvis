from __future__ import annotations

from typing import Any

from assistant_core.config.settings import Settings
from assistant_core.domain.loops import LoopStrategyName


def runtime_request_metadata(body: Any, settings: Settings) -> dict[str, str]:
    loop_strategy = resolve_loop_strategy(body.loop_strategy, settings)
    return {
        "loop_strategy": loop_strategy.value,
        "model_profile": resolve_model_profile(body.model_profile, loop_strategy, settings),
    }


def resolve_loop_strategy(
    requested: str | None,
    settings: Settings,
) -> LoopStrategyName:
    value = requested or LoopStrategyName.MEMORY_AUGMENTED_ANSWER.value
    try:
        loop_strategy = LoopStrategyName(value)
    except ValueError as exc:
        raise ValueError("loop strategy is not configured") from exc
    if loop_strategy.value not in settings.runtime_budgets:
        raise ValueError("loop strategy is not configured")
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP and not settings.policy.tools_enabled:
        raise ValueError("tool loop is disabled by policy")
    return loop_strategy


def resolve_model_profile(
    requested: str | None,
    loop_strategy: LoopStrategyName,
    settings: Settings,
) -> str:
    profile_name = requested or default_model_profile(loop_strategy)
    profile = settings.model_profiles.get(profile_name)
    if profile is None or not profile.enabled or profile.cloud:
        raise ValueError("model profile is not available for this request")
    if profile.purpose != required_model_profile_purpose(loop_strategy):
        raise ValueError("model profile purpose is not valid for loop strategy")
    return profile_name


def default_model_profile(loop_strategy: LoopStrategyName) -> str:
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP:
        return "local_structured"
    return "local_main"


def required_model_profile_purpose(loop_strategy: LoopStrategyName) -> str:
    if loop_strategy is LoopStrategyName.TOOL_REACT_LOOP:
        return "structured"
    return "chat"
