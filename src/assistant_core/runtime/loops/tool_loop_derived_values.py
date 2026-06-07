from __future__ import annotations

import re

from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.tools import ToolObservationStatus
from assistant_core.runtime.loops.tool_loop_derived_value_operations import (
    operation_families_match_request,
    operation_families_from_expression,
    operation_families_from_request,
)
from assistant_core.runtime.loops.tool_loop_live_numeric_sources import (
    completed_live_numeric_operand_groups_for_request,
    completed_live_numeric_literals_for_request,
    number_literal_variants,
    number_literals_from_text,
)
from assistant_core.runtime.loops.tool_loop_time_delta_units import GENERIC_TIME_DELTA_UNIT_PATTERN, TIME_DELTA_UNIT_PATTERN


_CALCULATION_INTENT_PATTERNS: tuple[str, ...] = (
    r"\b(?:calculate|compute|evaluate|find|derive)\b",
    r"(?:посчитай|рассчитай|вычисли|найди)",
)
_ARITHMETIC_WORD_OPERATOR_PATTERNS: tuple[str, ...] = (
    r"\b(?:plus|add|minus|subtract|times|multiply|multiplied|divide|divided|ratio|quotient|power|squared|cubed)\b",
    r"(?:плюс|прибав|добав|минус|вычт|умнож|раздел|подели|отношен|частн|степен|квадрат|куб)",
)
_SYMBOLIC_LIVE_OPERATOR_PATTERNS: tuple[str, ...] = (
    r"(?:\b(?:cpu|processor|memory|ram|load|disk|battery|network)\b|(?:процессор|цп|памят|оператив|нагрузк|диск|хранилищ|батар|аккумулятор|сет\w*)).{0,80}(?:\*\*|[+\-*/^×÷]).{0,80}\d",
    r"\d.{0,80}(?:\*\*|[+\-*/^×÷]).{0,80}(?:\b(?:cpu|processor|memory|ram|load|disk|battery|network)\b|(?:процессор|цп|памят|оператив|нагрузк|диск|хранилищ|батар|аккумулятор|сет\w*))",
)
_AVERAGE_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\b(?:average|mean)\b",
    r"средн",
)
_DIRECT_LIVE_AVERAGE_METRIC_PATTERNS: tuple[str, ...] = (
    r"\bload\s+average\b",
)
_THRESHOLD_COMPARISON_PATTERNS: tuple[str, ...] = (
    r"\b(?:over|above|under|below|greater\s+than|less\s+than|at\s+least|at\s+most|"
    r"more\s+than|not\s+more\s+than|higher\s+than|lower\s+than|between|equal\s+to)\b",
    r"[<>]=?",
    r"(?:больше|меньше|выше|ниже|превыш|как\s+минимум|как\s+максимум|не\s+менее|не\s+более)",
)
_THRESHOLD_COMPARISON_SPLIT_PATTERN = re.compile(
    r"\b(?:over|above|under|below|greater\s+than|less\s+than|at\s+least|at\s+most|"
    r"more\s+than|not\s+more\s+than|higher\s+than|lower\s+than|between|equal\s+to)\b|"
    r"[<>]=?|"
    r"(?:больше|меньше|выше|ниже|превыш|как\s+минимум|как\s+максимум|не\s+менее|не\s+более)",
    flags=re.IGNORECASE,
)
_COMPARISON_CLAUSE_BOUNDARY_PATTERN = re.compile(r"\b(?:and|or|и|или)\b", flags=re.IGNORECASE)
_TRANSFORM_INDICATOR_PATTERNS: tuple[str, ...] = (
    r"\b(?:math(?:ematical)?|arithmetic|numeric)\s+(?:operation|function|transform|conversion)\b",
    r"\b(?:function|formula|transform|convert|ratio|sum|difference|product|quotient|power|root|logarithm|log|ln|sqrt)\b",
    r"\b(?:average|mean)\b",
    r"(?:математическ|арифметическ|числов).*(?:операц|функц|преобраз|трансформ|конвер)",
    r"(?:функц|формул|преобраз|трансформ|конвер|отношен|сумм|разниц|произвед|частн|степен|корен|логарифм)",
    r"средн\w+.*(?:от|из|для|между|по)\b",
    r"с\s+точностью\s+до\s+\d+",
    *_ARITHMETIC_WORD_OPERATOR_PATTERNS,
    *_SYMBOLIC_LIVE_OPERATOR_PATTERNS,
)
_TIME_DELTA_VALUE_REFERENCE_PATTERNS: tuple[str, ...] = (
    rf"(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN}).*(?:до\b|остав|остал)",
    rf"(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN}).*(?:прошед|прошл).*(?:\bс\b|\bсо\b)",
    rf"(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN}).*(?:\bмежду\b|\bbetween\b)",
    rf"\b(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN})\b.*\b(?:until|remaining|left)\b",
    rf"\b(?:{TIME_DELTA_UNIT_PATTERN}|{GENERIC_TIME_DELTA_UNIT_PATTERN})\b.*\b(?:elapsed|passed)\b.*\b(?:since|from)\b",
)
_DIRECT_TIME_DELTA_REQUEST_PATTERNS: tuple[str, ...] = (
    rf"^(?:пожалуйста\s+)?(?:(?:сколько|посчитай|вычисли|рассчитай|найди|покажи|дай)\s+)?(?:точн\w*\s+)?(?:количеств\w*\s+)?{TIME_DELTA_UNIT_PATTERN}(?:\s+(?:остал\w*|остав\w*))?\s+(?:до|к)\b",
    rf"^(?:пожалуйста\s+)?(?:(?:сколько|посчитай|вычисли|рассчитай|найди|покажи|дай)\s+)?(?:точн\w*\s+)?(?:количеств\w*\s+)?{TIME_DELTA_UNIT_PATTERN}.*(?:прошед|прошл).*(?:\bс\b|\bсо\b)",
    rf"^(?:please\s+)?(?:(?:how\s+many|calculate|compute|find|show|give\s+me)\s+)?(?:exact\s+|total\s+)?(?:number\s+of\s+)?{TIME_DELTA_UNIT_PATTERN}(?:\s+(?:remaining|left))?\s+until\b",
    rf"^(?:please\s+)?(?:(?:how\s+many|calculate|compute|find|show|give\s+me)\s+)?(?:exact\s+|total\s+)?(?:number\s+of\s+)?{TIME_DELTA_UNIT_PATTERN}.*\b(?:elapsed|passed)\b.*\b(?:since|from)\b",
)
_LIVE_NUMERIC_REFERENCE_PATTERNS: tuple[str, ...] = (
    *_TIME_DELTA_VALUE_REFERENCE_PATTERNS,
    r"\b(?:cpu|processor|memory|ram|load|disk|battery|network)\b",
    r"(?:процессор|цп|памят|оператив|нагрузк|диск|хранилищ|батар|аккумулятор|сет\w*)",
    r"\b(?:value|number|quantity|percent|percentage)\b",
    r"(?:значен|числ|количеств|процент)",
)
_LEADING_LIVE_NUMERIC_REFERENCE_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:current\s+|the\s+current\s+)?(?:cpu|processor|memory|ram|load|disk|battery|network)\b",
    r"^\s*(?:текущ\w+\s+)?(?:процессор|цп|памят|оператив|нагрузк|диск|хранилищ|батар|аккумулятор|сет\w*)",
    rf"^\s*{TIME_DELTA_UNIT_PATTERN}\b.*\b(?:until|remaining|left)\b",
    rf"^\s*{TIME_DELTA_UNIT_PATTERN}\b.*\b(?:elapsed|passed)\b.*\b(?:since|from)\b",
    rf"^\s*{TIME_DELTA_UNIT_PATTERN}.*(?:до\b|остав|остал)",
    rf"^\s*{TIME_DELTA_UNIT_PATTERN}.*(?:прошед|прошл).*(?:\bс\b|\bсо\b)",
)
def request_requires_derived_live_value_calculation(value: str,
    *,
    live_state_detected: bool,
    live_time_delta_detected: bool,
) -> bool:
    if not live_state_detected:
        return False
    normalized = _normalize(value)
    if live_time_delta_detected:
        return _time_delta_transform_requested(normalized)
    if _matches_any(_DIRECT_LIVE_AVERAGE_METRIC_PATTERNS, normalized):
        return False
    if _matches_any(_THRESHOLD_COMPARISON_PATTERNS, normalized):
        return _threshold_live_transform_text(normalized) is not None
    return _matches_any(_TRANSFORM_INDICATOR_PATTERNS, normalized) and _matches_any(
        _LIVE_NUMERIC_REFERENCE_PATTERNS,
        normalized,
    )


def calculator_expression_matches_derived_request(
    request_text: str,
    ref: ToolObservationRef,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> bool:
    if not _is_completed(ref) or ref.tool_name != "calculator.evaluate":
        return False
    expression = ref.arguments.get("expression")
    if not isinstance(expression, str):
        return False
    expression_numbers = number_literals_from_text(expression)
    if not expression_numbers:
        return False
    live_numbers = completed_live_numeric_literals_for_request(
        request_text,
        _prior_observation_refs(ref, tool_observation_refs),
    )
    if not expression_numbers & live_numbers:
        return False
    request_numbers = _request_number_literals_for_derived_expression(request_text)
    live_operand_groups = completed_live_numeric_operand_groups_for_request(
        request_text,
        _prior_observation_refs(ref, tool_observation_refs),
    )
    implicit_numbers = _implicit_operation_number_literals(request_text, live_operand_groups)
    if not expression_numbers <= live_numbers | request_numbers | implicit_numbers:
        return False
    if not _expression_covers_required_live_operand_groups(
        request_text,
        expression_numbers,
        live_operand_groups,
    ):
        return False
    expression_operations = operation_families_from_expression(expression)
    if not expression_operations:
        return False
    request_operations = operation_families_from_request(request_text)
    if not request_operations:
        return False
    return operation_families_match_request(request_operations, expression_operations)


def _expression_covers_required_live_operand_groups(
    request_text: str,
    expression_numbers: frozenset[str],
    live_operand_groups,
) -> bool:
    if not live_operand_groups:
        return False
    return all(expression_numbers & group.values for group in live_operand_groups)


def _implicit_operation_number_literals(request_text: str, live_operand_groups) -> frozenset[str]:
    if not _matches_any(_AVERAGE_REQUEST_PATTERNS, request_text):
        return frozenset()
    return number_literal_variants(len(live_operand_groups))


def _request_number_literals_for_derived_expression(request_text: str) -> frozenset[str]:
    threshold_transform_text = _threshold_live_transform_text(_normalize(request_text))
    if threshold_transform_text is not None:
        return number_literals_from_text(threshold_transform_text)
    return number_literals_from_text(request_text)


def _prior_observation_refs(
    ref: ToolObservationRef,
    tool_observation_refs: tuple[ToolObservationRef, ...],
) -> tuple[ToolObservationRef, ...]:
    prior: list[ToolObservationRef] = []
    for item in tool_observation_refs:
        if item is ref:
            return tuple(prior)
        if item == ref:
            return tuple(prior)
        prior.append(item)
    return tuple(prior)


def _time_delta_transform_requested(value: str) -> bool:
    if not _matches_any(_TIME_DELTA_VALUE_REFERENCE_PATTERNS, value):
        return False
    if _matches_any(_DIRECT_TIME_DELTA_REQUEST_PATTERNS, value):
        return False
    return _matches_any(_CALCULATION_INTENT_PATTERNS, value) or _matches_any(
        _TRANSFORM_INDICATOR_PATTERNS,
        value,
    )


def _threshold_live_transform_text(value: str) -> str | None:
    matches = tuple(_THRESHOLD_COMPARISON_SPLIT_PATTERN.finditer(value))
    for index, match in enumerate(matches):
        previous_end = matches[index - 1].end() if index else 0
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        lhs = _rightmost_comparison_operand(value[previous_end : match.start()])
        if _text_has_live_transform(lhs):
            return lhs
        rhs = _leftmost_comparison_operand(value[match.end() : next_start])
        if _text_has_live_transform(rhs):
            return rhs
    return None


def _rightmost_comparison_operand(value: str) -> str:
    boundary_matches = _live_clause_boundary_matches(value)
    if not boundary_matches:
        return value
    return value[boundary_matches[-1].end() :]


def _leftmost_comparison_operand(value: str) -> str:
    boundary_matches = _live_clause_boundary_matches(value)
    if not boundary_matches:
        return value
    return value[: boundary_matches[0].start()]


def _live_clause_boundary_matches(value: str) -> tuple[re.Match[str], ...]:
    return tuple(
        match
        for match in _COMPARISON_CLAUSE_BOUNDARY_PATTERN.finditer(value)
        if _matches_any(_LEADING_LIVE_NUMERIC_REFERENCE_PATTERNS, value[match.end() :])
    )


def _text_has_live_transform(value: str) -> bool:
    return _matches_any(_LIVE_NUMERIC_REFERENCE_PATTERNS, value) and _matches_any(
        _TRANSFORM_INDICATOR_PATTERNS,
        value,
    )


def _is_completed(ref: ToolObservationRef) -> bool:
    return ref.status in {ToolObservationStatus.COMPLETED, ToolObservationStatus.COMPLETED.value}


def _matches_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _normalize(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r";+", " and ", lowered)
    lowered = re.sub(r"(?<!\d),|,(?!\d)|:+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()
