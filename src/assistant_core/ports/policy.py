from __future__ import annotations

from typing import Protocol

from assistant_core.domain.policy import (
    ContextPolicyRequest,
    MemoryWritePolicyRequest,
    ModelPolicyRequest,
    PolicyDecision,
)


class PolicyPort(Protocol):
    async def evaluate_model_request(
        self,
        request: ModelPolicyRequest,
    ) -> PolicyDecision: ...

    async def evaluate_memory_write(
        self,
        request: MemoryWritePolicyRequest,
    ) -> PolicyDecision: ...

    async def evaluate_context_inclusion(
        self,
        request: ContextPolicyRequest,
    ) -> PolicyDecision: ...
