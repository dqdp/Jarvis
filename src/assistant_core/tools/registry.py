from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from assistant_core.domain.tools import ToolSpec


class ToolRegistryError(ValueError):
    """Raised when the tool registry violates the ToolGateway contract."""


class ToolAdapter(Protocol):
    spec: ToolSpec
    content_type: str

    async def invoke(self, arguments: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class ToolClassificationResult:
    allowed: bool
    code: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolExecutionDenied(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = metadata or {}


@dataclass(frozen=True)
class ToolRegistry:
    _adapters: dict[str, ToolAdapter]

    def __init__(self, adapters: list[ToolAdapter]) -> None:
        by_name: dict[str, ToolAdapter] = {}
        for adapter in adapters:
            name = adapter.spec.name
            if name in by_name:
                raise ToolRegistryError(f"duplicate tool name: {name}")
            by_name[name] = adapter
        object.__setattr__(self, "_adapters", by_name)

    def list_specs(self, *, include_disabled: bool = False) -> list[ToolSpec]:
        specs = [
            adapter.spec
            for adapter in self._adapters.values()
            if include_disabled or adapter.spec.enabled
        ]
        return sorted(specs, key=lambda spec: spec.name)

    def get_spec(self, tool_name: str, *, include_disabled: bool = False) -> ToolSpec | None:
        adapter = self._adapters.get(tool_name)
        if adapter is None or (not include_disabled and not adapter.spec.enabled):
            return None
        return adapter.spec

    def get_adapter(self, tool_name: str, *, include_disabled: bool = False) -> ToolAdapter | None:
        adapter = self._adapters.get(tool_name)
        if adapter is None or (not include_disabled and not adapter.spec.enabled):
            return None
        return adapter
