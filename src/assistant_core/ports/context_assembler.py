from __future__ import annotations

from typing import Protocol

from assistant_core.domain.context import AssembledContext, ContextAssemblyRequest


class ContextAssemblerPort(Protocol):
    async def assemble(self, request: ContextAssemblyRequest) -> AssembledContext: ...
