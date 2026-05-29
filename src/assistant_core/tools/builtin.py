from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime
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
            description="Returns the current UTC time.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            adapter_name="builtin.datetime.now",
            sensitivity_ceiling=Sensitivity.PUBLIC,
            enabled=enabled,
        ),
        handler=lambda _arguments: {"iso": datetime.now(UTC).isoformat()},
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
            sensitivity_ceiling=Sensitivity.PUBLIC,
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
