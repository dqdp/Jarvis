from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assistant_core.domain.models import EmbeddingResponse
from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class GenerateEmbeddingCommand:
    texts: list[str]
    sensitivity: Sensitivity


class EmbeddingPort(Protocol):
    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse: ...
