from __future__ import annotations

import re

from assistant_core.domain.loops import (
    LoopBudget,
    LoopExecutionRequest,
    ToolObservationRef,
    ToolRequestPlan,
)
from assistant_core.domain.tools import ToolObservationStatus


TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS = 8.0
_ARITHMETIC_EXPRESSION_PATTERN = re.compile(
    r"(?i)(?:\d+(?:[.,]\d+)?|\be\b|\bpi\b|π)\s*(?:\*\*|[*/^×÷+\-])\s*"
    r"(?:\d+(?:[.,]\d+)?|\be\b|\bpi\b|π)"
)
_ARITHMETIC_TOKEN_PATTERN = re.compile(
    r"(?i)\d+(?:[.,]\d+)?|π|\bpi\b|\be\b|\*\*|[+\-*/^×÷()]"
)
_LIVE_STATE_INTENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?:"
    r"\bcpu\b|\bprocessor\b|\bmemory\b|\bram\b|\bload\b|\busage\b|"
    r"\bbattery\b|\bvpn\b|\bnetwork\b|\bexternal\s+ip\b|\bpublic\s+ip\b|"
    r"\bdaemon\b|\bsystem\s+status\b|\bhardware\b|\bdisk\b|"
    r"\bcurrent\s+(?:time|date)\b|\blocal\s+(?:time|date)\b|"
    r"\bwhat\s+time\s+is\s+it\b|"
    r"процессор|нагрузк|памят|оператив|батар|аккумулятор|сеть|"
    r"внешн\w*\s+(?:ip|айпи)|публичн\w*\s+(?:ip|айпи)|"
    r"\bцп\b|vpn|впн|желез|оборудован|диск|"
    r"сколько\s+врем(?:я|ени)|который\s+час|текущ\w*\s+(?:время|дата)"
    r")"
)


def tool_proposal_model_call_timeout(
    budget: LoopBudget,
    *,
    completed_observations: int,
    request: LoopExecutionRequest | None = None,
    request_plan: ToolRequestPlan | None = None,
    initial_model_call_cap_seconds: float = TOOL_PROPOSAL_MAX_MODEL_CALL_SECONDS,
) -> float:
    if completed_observations > 0:
        return float(budget.max_model_call_seconds)
    if (
        request is not None
        and request_plan is not None
        and request_requires_initial_tool_evidence(request, request_plan)
    ):
        return float(budget.max_model_call_seconds)
    return min(float(budget.max_model_call_seconds), float(initial_model_call_cap_seconds))


def request_requires_initial_tool_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
) -> bool:
    allowed = request_plan.allowed_tool_names or frozenset()
    if not allowed:
        return False
    if request_plan.final_answer_requires_observation():
        return True
    return request_needs_live_state_math_evidence(
        request,
        request_plan,
        tool_observation_refs=(),
    )


def should_defer_final_answer_for_calculator_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: list[ToolObservationRef],
    used_tool_calls: int,
) -> bool:
    allowed = request_plan.allowed_tool_names or frozenset()
    if "calculator.evaluate" not in allowed:
        return False
    if not request_needs_live_state_math_evidence(
        request,
        request_plan,
        tool_observation_refs=tuple(tool_observation_refs),
    ):
        return False
    has_live_state_observation = any(
        is_completed_observation(ref)
        and (
            ref.tool_name in request_plan.live_state_tool_names
            or is_live_state_tool_name(ref.tool_name)
        )
        for ref in tool_observation_refs
    )
    has_calculator_observation = any(
        is_completed_observation(ref)
        and ref.tool_name == "calculator.evaluate"
        and calculator_observation_matches_request(ref, request)
        for ref in tool_observation_refs
    )
    if has_live_state_observation and has_calculator_observation:
        return False
    if used_tool_calls >= request.budget.max_tool_calls:
        raise RuntimeError("required_tool_evidence_missing")
    return True


def request_needs_live_state_math_evidence(
    request: LoopExecutionRequest,
    request_plan: ToolRequestPlan,
    *,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    if "calculator.evaluate" not in (request_plan.allowed_tool_names or frozenset()):
        return False
    if not contains_arithmetic_expression(request.user_input):
        return False
    has_live_state_tool = bool(request_plan.live_state_tool_names) or any(
        is_live_state_tool_name(tool_name)
        for tool_name in (request_plan.allowed_tool_names or frozenset())
    )
    has_live_state_observation = any(
        is_live_state_tool_name(ref.tool_name) for ref in tool_observation_refs
    )
    if not (has_live_state_tool or has_live_state_observation):
        return False
    return contains_live_state_intent(request.user_input) or has_live_state_observation


def calculator_observation_matches_request(
    ref: ToolObservationRef,
    request: LoopExecutionRequest,
) -> bool:
    expected = expected_calculator_expression(request.user_input)
    if expected is None:
        return False
    actual = ref.arguments.get("expression")
    if not isinstance(actual, str):
        return False
    return normalize_calculator_expression(actual) == normalize_calculator_expression(expected)


def expected_calculator_expression(value: str) -> str | None:
    candidates = arithmetic_expression_candidates(value)
    if not candidates:
        return None
    return max(candidates, key=len)


def arithmetic_expression_candidates(value: str) -> list[str]:
    matches = list(_ARITHMETIC_TOKEN_PATTERN.finditer(value))
    if not matches:
        return []
    groups: list[list[re.Match[str]]] = []
    current = [matches[0]]
    for match in matches[1:]:
        gap = value[current[-1].end() : match.start()]
        if gap.strip():
            groups.append(current)
            current = [match]
        else:
            current.append(match)
    groups.append(current)

    candidates: list[str] = []
    for group in groups:
        text = value[group[0].start() : group[-1].end()].strip()
        if is_arithmetic_expression_candidate(text):
            candidates.append(text)
    return candidates


def is_arithmetic_expression_candidate(value: str) -> bool:
    if not re.search(r"(?:\*\*|[+\-*/^×÷])", value):
        return False
    balance = 0
    for char in value:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        if balance < 0:
            return False
    return balance == 0


def normalize_calculator_expression(value: str) -> str:
    return (
        value.replace("×", "*")
        .replace("÷", "/")
        .replace("π", "pi")
        .replace(",", ".")
        .replace(" ", "")
        .lower()
    )


def contains_arithmetic_expression(value: str) -> bool:
    return _ARITHMETIC_EXPRESSION_PATTERN.search(value) is not None


def contains_live_state_intent(value: str) -> bool:
    return _LIVE_STATE_INTENT_PATTERN.search(value) is not None


def is_completed_observation(ref: ToolObservationRef) -> bool:
    return ref.status in {ToolObservationStatus.COMPLETED, ToolObservationStatus.COMPLETED.value}


def is_live_state_tool_name(tool_name: str) -> bool:
    return (
        tool_name in {"datetime.now", "datetime.until", "daemon.status"}
        or tool_name.startswith("tool.system.read.")
    )
