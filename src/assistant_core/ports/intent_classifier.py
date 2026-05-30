from __future__ import annotations

from typing import Protocol

from assistant_core.domain.loop_selection import IntentClassification, LoopSelectionRequest


class IntentClassifierPort(Protocol):
    async def classify(self, request: LoopSelectionRequest) -> IntentClassification: ...
