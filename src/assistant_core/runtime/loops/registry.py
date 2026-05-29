from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from assistant_core.domain.loops import (
    LoopExecutionRequest,
    LoopExecutionResult,
    LoopStreamEvent,
    LoopStrategyName,
    UnknownLoopStrategy,
)


class LoopStrategy(Protocol):
    strategy_name: LoopStrategyName

    async def run_turn(self, request: LoopExecutionRequest) -> LoopExecutionResult:
        ...

    def stream_turn(self, request: LoopExecutionRequest) -> AsyncIterator[LoopStreamEvent]:
        ...


class LoopStrategyRegistry:
    def __init__(self, strategies: Iterable[LoopStrategy]) -> None:
        self._strategies: dict[LoopStrategyName, LoopStrategy] = {}
        for strategy in strategies:
            strategy_name = LoopStrategyName(strategy.strategy_name)
            if strategy_name in self._strategies:
                raise ValueError(f"duplicate loop strategy: {strategy_name.value}")
            self._strategies[strategy_name] = strategy

    def default_strategy(self) -> LoopStrategy:
        return self.get(LoopStrategyName.MEMORY_AUGMENTED_ANSWER)

    def get(self, strategy_name: LoopStrategyName | str) -> LoopStrategy:
        try:
            normalized = LoopStrategyName(strategy_name)
        except ValueError as exc:
            raise UnknownLoopStrategy(f"unknown loop strategy: {strategy_name}") from exc
        try:
            return self._strategies[normalized]
        except KeyError as exc:
            raise UnknownLoopStrategy(f"unknown loop strategy: {normalized.value}") from exc
