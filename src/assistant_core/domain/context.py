from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.messages import ChatMessage
from assistant_core.domain.policy import PermissionMode
from assistant_core.domain.sensitivity import Sensitivity


@dataclass(frozen=True)
class ContextAssemblyRequest:
    request_id: str
    conversation_id: str
    user_id: str
    current_user_message: str
    active_project_namespace: str | None
    loop_strategy: str
    model_profile: str
    current_message_sensitivity: Sensitivity = Sensitivity.PROJECT
    current_user_message_id: str | None = None
    causation_event_id: str | None = None
    permission_mode: PermissionMode | str | None = None
    max_messages: int | None = None
    max_input_tokens: int | None = None
    tool_observation_refs: tuple[ToolObservationRef, ...] = ()

    def __post_init__(self) -> None:
        if self.permission_mode is not None and not isinstance(
            self.permission_mode,
            PermissionMode,
        ):
            object.__setattr__(self, "permission_mode", PermissionMode(self.permission_mode))


@dataclass(frozen=True)
class ContextDroppedRef:
    kind: str
    ref_id: str
    reason: str


@dataclass(frozen=True)
class ContextSection:
    name: str
    content: str
    token_estimate: int
    source_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextContentHitRef:
    source_id: str
    chunk_id: str
    citation: str
    score: float
    sensitivity: Sensitivity
    content_hash: str


@dataclass(frozen=True)
class ContextManifest:
    context_manifest_id: str
    request_id: str
    conversation_id: str
    loop_strategy: str
    model_profile: str
    section_names: list[str]
    used_message_ids: list[str]
    used_memory_ids: list[str]
    dropped_refs: list[ContextDroppedRef]
    token_estimate: int
    active_namespaces: list[str]
    retrieval_parameters: dict[str, Any]
    max_sensitivity: Sensitivity
    sources_by_sensitivity: dict[str, list[str]]
    degraded: bool
    full_prompt_stored: bool = False
    used_content_refs: list[ContextContentHitRef] = field(default_factory=list)


@dataclass(frozen=True)
class AssembledContext:
    messages: list[ChatMessage]
    sections: list[ContextSection]
    manifest: ContextManifest
    token_estimate: int
