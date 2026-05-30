from __future__ import annotations

from assistant_core.domain.memory import CreateMemoryCommand, MemoryRecord


class MemoryWriteService:
    """Application boundary for memory write workflows."""

    def __init__(self, store) -> None:
        self._store = store

    async def create_memory(self, command: CreateMemoryCommand) -> MemoryRecord:
        return await self._store.create_memory(command)
