from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from assistant_core.domain.model_invocations import ModelInvocationRecord
from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class StartModelInvocationCommand:
    request_id: str | None
    conversation_id: str | None
    profile: str
    provider: str
    model: str
    purpose: str
    sensitivity: Sensitivity
    streaming: bool
    input_token_estimate: int | None = None
    context_manifest_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinishModelInvocationCommand:
    model_invocation_id: str
    status: str
    input_tokens_reported: int | None = None
    output_tokens_reported: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class ModelInvocationRepositoryPort(Protocol):
    async def start(
        self,
        command: StartModelInvocationCommand,
    ) -> ModelInvocationRecord: ...

    async def finish(
        self,
        command: FinishModelInvocationCommand,
    ) -> ModelInvocationRecord: ...
