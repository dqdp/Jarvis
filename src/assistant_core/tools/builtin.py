from __future__ import annotations

import ast
import math
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolInvocationResult, ToolParseStatus, ToolSpec


@dataclass
class BuiltinToolAdapter:
    spec: ToolSpec
    handler: object
    content_type: str = "application/json"

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.handler(arguments)


_CALCULATOR_MAX_EXPRESSION_LENGTH = 256
_CALCULATOR_MAX_AST_NODES = 96
_CALCULATOR_MAX_ABS_RESULT = 1e308
_CALCULATOR_MAX_EXPONENT = 256
_CALCULATOR_MAX_FACTORIAL = 100
_CALCULATOR_MAX_ROUND_DIGITS = 12
_TIME_DELTA_UNITS = ["microseconds", "milliseconds", "seconds", "minutes", "hours", "days", "weeks"]
_CALENDAR_DIFF_UNITS = [*_TIME_DELTA_UNITS, "months", "quarters", "decades"]

_CALCULATOR_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_CALCULATOR_UNARY_FUNCTIONS = {
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "exp": math.exp,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
}

_CALCULATOR_BINARY_FUNCTIONS = {
    "atan2": math.atan2,
    "hypot": math.hypot,
}


def datetime_now_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="datetime.now",
            display_name="Current Time",
            description="Returns the current local date and time.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            adapter_name="builtin.datetime.now",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
        handler=_datetime_now,
    )


def datetime_until_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="datetime.until",
            display_name="Time Until",
            description=(
                "Use for countdowns to supported targets. If from_iso is omitted, "
                "the tool uses its current local timestamp; otherwise from_iso must "
                "be a timezone-aware ISO timestamp."
            ),
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {
                    "from_iso": {"type": "string"},
                    "target": {"type": "string", "enum": ["next_new_year"]},
                    "unit": {
                        "type": "string",
                        "enum": _TIME_DELTA_UNITS,
                    },
                },
                "required": ["target", "unit"],
                "additionalProperties": False,
            },
            adapter_name="builtin.datetime.until",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
        handler=_datetime_until,
    )


def datetime_diff_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="datetime.diff",
            display_name="Timestamp Difference",
            description=(
                "Use for elapsed time between two known timezone-aware ISO timestamps "
                "in microseconds through weeks. Provide explicit from_iso/to_iso; this "
                "tool does not resolve event names or holidays."
            ),
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {
                    "from_iso": {
                        "type": "string",
                        "description": "Start timestamp as timezone-aware ISO datetime.",
                    },
                    "to_iso": {
                        "type": "string",
                        "description": "End timestamp as timezone-aware ISO datetime.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": _TIME_DELTA_UNITS,
                        "description": "Requested fixed-duration output unit.",
                    },
                    "absolute": {
                        "type": "boolean",
                        "description": "Return an absolute value instead of a signed interval.",
                    },
                },
                "required": ["from_iso", "to_iso", "unit"],
                "additionalProperties": False,
            },
            adapter_name="builtin.datetime.diff",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
        handler=_datetime_diff,
    )


def calendar_diff_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="calendar.diff",
            display_name="Calendar Difference",
            description=(
                "Use for differences between two known timezone-aware ISO timestamps, "
                "including calendar units such as months, quarters and decades. Provide "
                "explicit from_iso/to_iso; this tool does not resolve event names or holidays."
            ),
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {
                    "from_iso": {
                        "type": "string",
                        "description": "Start timestamp as timezone-aware ISO datetime.",
                    },
                    "to_iso": {
                        "type": "string",
                        "description": "End timestamp as timezone-aware ISO datetime.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": _CALENDAR_DIFF_UNITS,
                        "description": "Requested fixed-duration or calendar output unit.",
                    },
                    "absolute": {
                        "type": "boolean",
                        "description": "Return an absolute value instead of a signed interval.",
                    },
                },
                "required": ["from_iso", "to_iso", "unit"],
                "additionalProperties": False,
            },
            adapter_name="builtin.calendar.diff",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
        handler=_calendar_diff,
    )


def calculator_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="calculator.evaluate",
            display_name="Scientific Calculator",
            description=(
                "Evaluates a bounded scientific expression. Supports +, -, *, /, //, %, "
                "^ or ** powers, constants pi/e/tau, and common math functions such as "
                "sqrt, sin, cos, tan, log, exp, round, min, max and factorial."
            ),
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "maxLength": _CALCULATOR_MAX_EXPRESSION_LENGTH,
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            adapter_name="builtin.calculator.evaluate",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
        handler=lambda arguments: _evaluate_expression(str(arguments["expression"])),
        content_type="text/plain",
    )


def daemon_status_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    payload = {"status": "ok"}
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="daemon.status",
            display_name="Daemon Status",
            description="Returns local daemon status.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            adapter_name="builtin.daemon.status",
            sensitivity_ceiling=Sensitivity.PUBLIC,
            enabled=enabled,
        ),
        handler=lambda _arguments: _typed_json_result(
            payload,
            structured_schema="daemon.status",
            structured_content=payload,
        ),
    )


def _datetime_now(_arguments: dict[str, Any]) -> ToolInvocationResult:
    payload = {"iso": datetime.now().astimezone().isoformat()}
    return _typed_json_result(
        payload,
        structured_schema="datetime.now",
        structured_content=payload,
    )


def _datetime_until(arguments: dict[str, Any]) -> ToolInvocationResult:
    raw_source = arguments.get("from_iso")
    source = (
        datetime.fromisoformat(str(raw_source))
        if raw_source
        else datetime.now().astimezone()
    )
    target_name = str(arguments["target"])
    if target_name != "next_new_year":
        raise ValueError("unsupported datetime target")
    target = source.replace(
        year=source.year + 1,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    unit = str(arguments["unit"])
    values = _fixed_duration_values(source, target, absolute=False)
    if values["microseconds"] < 0:
        values = _fixed_duration_values(source, source, absolute=False)
    payload = {
        "from_iso": source.isoformat(),
        "target": target_name,
        "target_iso": target.isoformat(),
        **values,
        "unit": unit,
        "value": values[unit],
    }
    return _typed_json_result(
        payload,
        structured_schema="datetime.until",
        structured_content=payload,
    )


def _datetime_diff(arguments: dict[str, Any]) -> ToolInvocationResult:
    source = _timezone_aware_iso_datetime(arguments["from_iso"], argument_name="from_iso")
    target = _timezone_aware_iso_datetime(arguments["to_iso"], argument_name="to_iso")
    absolute = bool(arguments.get("absolute", False))
    unit = str(arguments["unit"])
    values = _fixed_duration_values(source, target, absolute=absolute)
    payload = {
        "from_iso": source.isoformat(),
        "to_iso": target.isoformat(),
        "microseconds": values["microseconds"],
        "milliseconds": values["milliseconds"],
        "seconds": values["seconds"],
        "minutes": values["minutes"],
        "hours": values["hours"],
        "days": values["days"],
        "weeks": values["weeks"],
        "unit": unit,
        "value": values[unit],
        "absolute": absolute,
    }
    return _typed_json_result(
        payload,
        structured_schema="datetime.diff",
        structured_content=payload,
    )


def _calendar_diff(arguments: dict[str, Any]) -> ToolInvocationResult:
    source = _timezone_aware_iso_datetime(arguments["from_iso"], argument_name="from_iso")
    target = _timezone_aware_iso_datetime(arguments["to_iso"], argument_name="to_iso")
    absolute = bool(arguments.get("absolute", False))
    unit = str(arguments["unit"])
    values = _fixed_duration_values(source, target, absolute=absolute)
    month_count = _whole_months_between(source, target)
    if absolute:
        month_count = abs(month_count)
    values.update({
        "months": month_count,
        "quarters": _whole_calendar_units(month_count, 3),
        "decades": _whole_calendar_units(month_count, 120),
    })
    payload = {
        "from_iso": source.isoformat(),
        "to_iso": target.isoformat(),
        **values,
        "unit": unit,
        "value": values[unit],
        "absolute": absolute,
    }
    return _typed_json_result(
        payload,
        structured_schema="calendar.diff",
        structured_content=payload,
    )


def _fixed_duration_values(
    source: datetime,
    target: datetime,
    *,
    absolute: bool,
) -> dict[str, int | float]:
    delta = target - source
    total_microseconds = (
        delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
    )
    if absolute:
        total_microseconds = abs(total_microseconds)
    total_seconds_float = total_microseconds / 1_000_000
    total_seconds: int | float = (
        int(total_seconds_float)
        if total_seconds_float.is_integer()
        else total_seconds_float
    )
    return {
        "microseconds": total_microseconds,
        "milliseconds": total_microseconds / 1000,
        "seconds": total_seconds,
        "minutes": total_seconds_float / 60,
        "hours": total_seconds_float / 3600,
        "days": total_seconds_float / 86400,
        "weeks": total_seconds_float / 604800,
    }


def _whole_months_between(source: datetime, target: datetime) -> int:
    if target < source:
        return -_whole_months_between(target, source)
    months = (target.year - source.year) * 12 + target.month - source.month
    if _add_months(source, months) > target:
        months -= 1
    return months


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(day=min(value.day, monthrange(year, month)[1]), year=year, month=month)


def _whole_calendar_units(month_count: int, months_per_unit: int) -> int:
    sign = -1 if month_count < 0 else 1
    return sign * (abs(month_count) // months_per_unit)


def _timezone_aware_iso_datetime(value: object, *, argument_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{argument_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{argument_name} must include a timezone offset")
    return parsed


def _typed_json_result(
    payload: dict[str, Any],
    *,
    structured_schema: str,
    structured_content: dict[str, Any],
) -> ToolInvocationResult:
    import json

    content = json.dumps(payload, sort_keys=True)
    return ToolInvocationResult(
        content=content,
        content_type="application/json",
        output_bytes=len(content.encode("utf-8")),
        structured_content=structured_content,
        structured_schema=structured_schema,
        structured_schema_version=1,
        parse_status=ToolParseStatus.PARSED,
    )


def _evaluate_expression(expression: str) -> str:
    if len(expression) > _CALCULATOR_MAX_EXPRESSION_LENGTH:
        raise ValueError("calculator expression is too long")
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid calculator expression") from exc
    if sum(1 for _node in ast.walk(tree)) > _CALCULATOR_MAX_AST_NODES:
        raise ValueError("calculator expression is too complex")
    result = _ensure_bounded_number(_eval_node(tree.body))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    if isinstance(result, float):
        return format(result, ".15g")
    return str(result)


def _eval_node(node: ast.AST) -> int | float:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return _ensure_bounded_number(node.value)
    if isinstance(node, ast.Name):
        if node.id in _CALCULATOR_CONSTANTS:
            return _CALCULATOR_CONSTANTS[node.id]
        raise ValueError("unknown calculator name")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return _ensure_bounded_number(-operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError("unsupported calculator unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return _ensure_bounded_number(left + right)
        if isinstance(node.op, ast.Sub):
            return _ensure_bounded_number(left - right)
        if isinstance(node.op, ast.Mult):
            return _ensure_bounded_number(left * right)
        if isinstance(node.op, ast.Div):
            return _ensure_bounded_number(left / right)
        if isinstance(node.op, ast.FloorDiv):
            return _ensure_bounded_number(left // right)
        if isinstance(node.op, ast.Mod):
            return _ensure_bounded_number(left % right)
        if isinstance(node.op, ast.Pow):
            return _checked_power(left, right)
        raise ValueError("unsupported calculator binary operator")
    if isinstance(node, ast.Call):
        return _eval_call(node)
    raise ValueError("unsupported calculator expression")


def _eval_call(node: ast.Call) -> int | float:
    if not isinstance(node.func, ast.Name):
        raise ValueError("unsupported calculator function")
    if node.keywords:
        raise ValueError("calculator functions do not accept keyword arguments")
    name = node.func.id
    args = [_eval_node(argument) for argument in node.args]

    if name in _CALCULATOR_UNARY_FUNCTIONS:
        _require_arity(name, args, {1})
        return _call_math_function(name, _CALCULATOR_UNARY_FUNCTIONS[name], args)
    if name in _CALCULATOR_BINARY_FUNCTIONS:
        _require_arity(name, args, {2})
        return _call_math_function(name, _CALCULATOR_BINARY_FUNCTIONS[name], args)
    if name in {"log", "ln"}:
        _require_arity(name, args, {1, 2} if name == "log" else {1})
        return _call_log(name, args)
    if name == "pow":
        _require_arity(name, args, {2})
        return _checked_power(args[0], args[1])
    if name == "round":
        _require_arity(name, args, {1, 2})
        return _call_round(args)
    if name == "min":
        if not args:
            raise ValueError("calculator function expects at least one argument")
        return _ensure_bounded_number(min(args))
    if name == "max":
        if not args:
            raise ValueError("calculator function expects at least one argument")
        return _ensure_bounded_number(max(args))
    if name in {"factorial", "fact"}:
        _require_arity(name, args, {1})
        value = _integer_argument(
            args[0],
            name,
            minimum=0,
            maximum=_CALCULATOR_MAX_FACTORIAL,
        )
        return _ensure_bounded_number(math.factorial(value))
    raise ValueError("unknown calculator function")


def _call_log(name: str, args: list[int | float]) -> int | float:
    try:
        if name == "ln":
            result = math.log(args[0])
        elif len(args) == 1:
            result = math.log(args[0])
        else:
            result = math.log(args[0], args[1])
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid calculator function arguments") from exc
    return _ensure_bounded_number(result)


def _call_round(args: list[int | float]) -> int | float:
    if len(args) == 1:
        return _ensure_bounded_number(round(args[0]))
    digits = _integer_argument(
        args[1],
        "round",
        minimum=-_CALCULATOR_MAX_ROUND_DIGITS,
        maximum=_CALCULATOR_MAX_ROUND_DIGITS,
    )
    return _ensure_bounded_number(round(args[0], digits))


def _call_math_function(
    name: str,
    function: object,
    args: list[int | float],
) -> int | float:
    try:
        result = function(*args)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid calculator function arguments") from exc
    return _ensure_bounded_number(result)


def _checked_power(left: int | float, right: int | float) -> int | float:
    if abs(right) > _CALCULATOR_MAX_EXPONENT:
        raise ValueError("calculator exponent is too large")
    try:
        result = left**right
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid calculator power arguments") from exc
    return _ensure_bounded_number(result)


def _require_arity(name: str, args: list[int | float], allowed: set[int]) -> None:
    if len(args) not in allowed:
        raise ValueError(f"calculator function has invalid arity: {name}")


def _integer_argument(
    value: int | float,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"calculator function expects an integer argument: {name}")
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise ValueError(f"calculator integer argument is too small: {name}")
    if maximum is not None and integer > maximum:
        raise ValueError(f"calculator integer argument is too large: {name}")
    return integer


def _ensure_bounded_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("calculator result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("calculator result is not finite")
    if abs(value) > _CALCULATOR_MAX_ABS_RESULT:
        raise ValueError("calculator result is too large")
    return value
