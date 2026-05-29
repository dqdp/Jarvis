from __future__ import annotations

from typing import Protocol

from assistant_core.domain.tools import ToolCallRequest, ToolObservation, ToolSpec


class ToolGatewayPort(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def get_tool(self, tool_name: str) -> ToolSpec | None: ...

    async def invoke(self, request: ToolCallRequest) -> ToolObservation: ...
