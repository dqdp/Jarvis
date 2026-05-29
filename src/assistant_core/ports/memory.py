from __future__ import annotations

from typing import Protocol

from assistant_core.domain.memory import (
    ArchiveMemoryCommand,
    CreateMemoryCommand,
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    SupersedeMemoryCommand,
    UpdateMemoryCommand,
)


class MemoryStoreError(Exception):
    """Base error for memory store contract violations."""


class UnknownMemoryNamespace(MemoryStoreError):
    """Raised when a memory namespace is not registered."""


class InvalidMemoryType(MemoryStoreError):
    """Raised when a memory type is outside the Phase 1 type set."""


class MemoryTypeNotAllowed(MemoryStoreError):
    """Raised when a memory type is not allowed for a namespace."""


class MemoryPolicyDenied(MemoryStoreError):
    """Raised when PolicyPort denies a memory write."""


class MemoryRetrievalError(MemoryStoreError):
    """Raised when memory retrieval cannot complete."""


class MemoryWritePort(Protocol):
    async def create_memory(self, command: CreateMemoryCommand) -> MemoryRecord: ...

    async def update_memory(self, command: UpdateMemoryCommand) -> MemoryRecord: ...

    async def archive_memory(self, command: ArchiveMemoryCommand) -> None: ...

    async def supersede_memory(self, command: SupersedeMemoryCommand) -> MemoryRecord: ...


class MemoryReadPort(Protocol):
    async def retrieve(self, query: MemoryQuery) -> list[MemoryHit]: ...

    async def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    async def list_memories(
        self,
        limit: int = 100,
        query: str | None = None,
    ) -> list[MemoryRecord]: ...
