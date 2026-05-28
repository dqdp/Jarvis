from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class ModelInvocationRecord:
    model_invocation_id: str
    request_id: str | None
    conversation_id: str | None
    profile: str
    provider: str
    model: str
    purpose: str
    sensitivity: Sensitivity
    status: str
    started_at: datetime
    finished_at: datetime | None
    latency_ms: int | None
    input_token_estimate: int | None
    input_tokens_reported: int | None
    output_tokens_reported: int | None
    streaming: bool
    error_type: str | None
    error_message: str | None
    context_manifest_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
