from __future__ import annotations

from datetime import UTC, datetime

from assistant_core.domain.tools import ToolCallRequest, ToolObservation, ToolObservationStatus, ToolSpec


class NoopToolGateway:
    async def list_tools(self) -> list[ToolSpec]:
        return []

    async def get_tool(self, tool_name: str) -> ToolSpec | None:
        del tool_name
        return None

    async def invoke(self, request: ToolCallRequest) -> ToolObservation:
        now = datetime.now(UTC)
        return ToolObservation.empty(
            tool_name=request.tool_name,
            status=ToolObservationStatus.FAILED,
            sensitivity=request.sensitivity,
            started_at=now,
            completed_at=now,
            error={"code": "unknown_tool", "message": "tool is not registered"},
        )
