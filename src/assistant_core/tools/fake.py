from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from assistant_core.domain.policy import Capability, RiskClass
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.domain.tools import ToolSpec


@dataclass
class FakeToolAdapter:
    spec: ToolSpec
    response: Any | None = None
    content_type: str = "text/plain"
    call_count: int = 0

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        self.call_count += 1
        if self.response is not None:
            return self.response
        return arguments.get("message", "")


@dataclass
class FailingToolAdapter:
    spec: ToolSpec
    content_type: str = "text/plain"
    call_count: int = 0

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        self.call_count += 1
        raise RuntimeError("fake tool failed")


@dataclass
class TimeoutToolAdapter:
    spec: ToolSpec
    content_type: str = "text/plain"
    call_count: int = 0

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        self.call_count += 1
        delay = float(arguments.get("delay_seconds", 1.0))
        await asyncio.sleep(delay)
        return "completed"


def fake_echo_tool(*, enabled: bool = True) -> FakeToolAdapter:
    return FakeToolAdapter(
        spec=ToolSpec(
            name="fake.echo",
            display_name="Fake Echo",
            description="Echoes a test message.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
            adapter_name="fake.echo",
            sensitivity_ceiling=Sensitivity.PROJECT,
            enabled=enabled,
        ),
    )


def fake_fail_tool(*, enabled: bool = True) -> FailingToolAdapter:
    return FailingToolAdapter(
        spec=ToolSpec(
            name="fake.fail",
            display_name="Fake Failure",
            description="Raises a deterministic fake failure.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            adapter_name="fake.fail",
            enabled=enabled,
        ),
    )


def fake_timeout_tool(*, enabled: bool = True) -> TimeoutToolAdapter:
    return TimeoutToolAdapter(
        spec=ToolSpec(
            name="fake.timeout",
            display_name="Fake Timeout",
            description="Sleeps for a requested duration.",
            capability=Capability.TOOL_SAFE,
            risk_classes=frozenset({RiskClass.SAFE}),
            input_schema={
                "type": "object",
                "properties": {"delay_seconds": {"type": "number"}},
                "required": ["delay_seconds"],
                "additionalProperties": False,
            },
            adapter_name="fake.timeout",
            default_timeout_seconds=0.1,
            enabled=enabled,
        ),
    )
