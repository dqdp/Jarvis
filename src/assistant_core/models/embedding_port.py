from __future__ import annotations

from assistant_core.domain.models import EmbeddingRequest, EmbeddingResponse
from assistant_core.ports.embedding import GenerateEmbeddingCommand
from assistant_core.ports.model_router import ModelRouterPort


class ModelRouterEmbeddingPort:
    def __init__(self, *, router: ModelRouterPort, profile: str) -> None:
        self._router = router
        self._profile = profile

    async def embed(self, command: GenerateEmbeddingCommand) -> EmbeddingResponse:
        return await self._router.embed(
            EmbeddingRequest(
                profile=self._profile,
                texts=command.texts,
                sensitivity=command.sensitivity,
            ),
        )
