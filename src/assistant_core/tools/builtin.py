from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolSpec


@dataclass
class BuiltinToolAdapter:
    spec: ToolSpec
    handler: object
    content_type: str = "application/json"

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.handler(arguments)


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
        handler=lambda _arguments: {"iso": datetime.now().astimezone().isoformat()},
    )


def datetime_until_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="datetime.until",
            display_name="Time Until",
            description="Computes a deterministic time interval from a source timestamp to a supported target.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {
                    "from_iso": {"type": "string"},
                    "target": {"type": "string", "enum": ["next_new_year"]},
                    "unit": {
                        "type": "string",
                        "enum": ["seconds", "minutes", "hours", "days"],
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


def calculator_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
    return BuiltinToolAdapter(
        spec=ToolSpec(
            name="calculator.evaluate",
            display_name="Calculator",
            description="Evaluates a bounded arithmetic expression.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
                "maxLength": 128,
            },
            adapter_name="builtin.calculator.evaluate",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
        handler=lambda arguments: _evaluate_expression(str(arguments["expression"])),
        content_type="text/plain",
    )


def daemon_status_tool(*, enabled: bool = True) -> BuiltinToolAdapter:
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
        handler=lambda _arguments: {"status": "ok"},
    )


def _datetime_until(arguments: dict[str, Any]) -> dict[str, Any]:
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
    total_seconds = max(0, int((target - source).total_seconds()))
    unit = str(arguments["unit"])
    values: dict[str, int | float] = {
        "seconds": total_seconds,
        "minutes": total_seconds / 60,
        "hours": total_seconds / 3600,
        "days": total_seconds / 86400,
    }
    return {
        "from_iso": source.isoformat(),
        "target": target_name,
        "target_iso": target.isoformat(),
        "seconds": total_seconds,
        "unit": unit,
        "value": values[unit],
    }


def _evaluate_expression(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _node in ast.walk(tree)) > 64:
        raise ValueError("calculator expression is too complex")
    result = _eval_node(tree.body)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported calculator expression")
