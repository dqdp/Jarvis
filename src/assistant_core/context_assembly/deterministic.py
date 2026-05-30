from __future__ import annotations

from assistant_core.config.settings import Settings
from assistant_core.context_assembly.audit import ContextAssemblyAuditRecorder
from assistant_core.context_assembly.manifest import (
    manifest as _manifest,
)
from assistant_core.context_assembly.policy_filter import dropped_reason as _dropped_reason
from assistant_core.context_assembly.rendering import (
    build_sections as _build_sections,
    context_token_estimate as _context_token_estimate,
    prompt_messages as _prompt_messages,
)
from assistant_core.context_assembly.retrieval import active_namespaces as _active_namespaces
from assistant_core.context_assembly.trimming import (
    apply_message_count_limit as _apply_message_count_limit,
    apply_token_budget as _apply_token_budget,
    current_conversation_messages as _current_conversation_messages,
    exclude_current_user_message as _exclude_current_user_message,
)
from assistant_core.domain.context import (
    AssembledContext,
    ContextAssemblyRequest,
    ContextDroppedRef,
)
from assistant_core.domain.conversations import ConversationMessage, RecentMessagesQuery
from assistant_core.domain.content_retrieval import ContentHit, ContentRetrievalQuery
from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.memory import MemoryHit, MemoryQuery
from assistant_core.domain.policy import (
    Capability,
    CapabilityPolicyRequest,
    ContextPolicyRequest,
    RiskClass,
)
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.memory import MemoryRetrievalError
from assistant_core.ports.policy import PolicyPort


class ContextPolicyDenied(Exception):
    """Raised when policy rejects the current user message before retrieval."""


class DeterministicContextAssembler:
    def __init__(
        self,
        *,
        conversation_store,
        memory_read,
        content_retrieval=None,
        event_log: EventLogPort,
        policy: PolicyPort,
        settings: Settings | None = None,
    ) -> None:
        self._conversation_store = conversation_store
        self._memory_read = memory_read
        self._content_retrieval = content_retrieval
        self._policy = policy
        self._audit = ContextAssemblyAuditRecorder(event_log)
        self._memory_hit_limit = (
            int(settings.memory.retrieval.max_hits_total)
            if settings is not None
            else 8
        )
        context_budget = settings.context_assembly.context_budget if settings is not None else {}
        conversation_window = (
            settings.context_assembly.conversation_window if settings is not None else {}
        )
        self._content_hit_limit = int(context_budget.get("content_hits_max", 4))
        self._conversation_message_limit = int(conversation_window.get("max_messages", 12))
        self._conversation_source_limit = int(
            conversation_window.get("source_limit", max(self._conversation_message_limit, 1000)),
        )
        self._retrieval_parameters = {
            "max_hits_total": self._memory_hit_limit,
            "content_hits_max": self._content_hit_limit,
            "query_source": "current_user_message",
        }

    async def assemble(self, request: ContextAssemblyRequest) -> AssembledContext:
        dropped_refs: list[ContextDroppedRef] = []
        degraded = False
        active_namespaces = _active_namespaces(request)
        await self._authorize_current_message(request)

        try:
            raw_memory_hits = await self._memory_read.retrieve(
                MemoryQuery(
                    text=request.current_user_message,
                    namespaces=active_namespaces,
                    limit=self._memory_hit_limit,
                ),
            )
        except MemoryRetrievalError:
            raw_memory_hits = []
            degraded = True
            dropped_refs.append(
                ContextDroppedRef(kind="memory", ref_id="*", reason="retrieval_failed"),
            )
            memory_event = await self._audit.record_memory_retrieval_failed(request)
        else:
            memory_event = None

        memory_hits = await self._filter_memory_hits_by_policy(
            request,
            raw_memory_hits,
            dropped_refs,
        )
        if memory_event is None:
            memory_event = await self._audit.record_memory_retrieved(request, memory_hits)

        raw_content_hits: list[ContentHit] = []
        if self._content_retrieval is not None:
            raw_content_hits = await self._retrieve_content_hits(request, dropped_refs)
        content_hits = await self._filter_content_hits_by_policy(
            request,
            raw_content_hits,
            dropped_refs,
        )

        recent_messages = await self._conversation_store.load_recent_messages(
            RecentMessagesQuery(
                conversation_id=request.conversation_id,
                limit=self._conversation_source_limit,
            ),
        )
        recent_messages = await self._filter_messages_by_policy(
            request,
            recent_messages,
            dropped_refs,
        )
        recent_messages = _exclude_current_user_message(request, recent_messages)
        recent_messages = _apply_message_count_limit(
            recent_messages,
            request.max_messages or self._conversation_message_limit,
            dropped_refs,
        )
        tool_observation_refs = await self._filter_tool_observation_refs_by_policy(
            request,
            list(request.tool_observation_refs),
            dropped_refs,
        )

        sections = _build_sections(
            request,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
        )
        recent_messages, content_hits, sections, tool_observation_refs = _apply_token_budget(
            request,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
            sections,
            dropped_refs,
        )

        conversation_messages = _current_conversation_messages(request, recent_messages)

        token_estimate = _context_token_estimate(sections, conversation_messages)
        manifest = _manifest(
            request,
            sections,
            recent_messages,
            memory_hits,
            content_hits,
            tool_observation_refs,
            dropped_refs,
            token_estimate,
            active_namespaces,
            self._retrieval_parameters,
            degraded,
        )
        messages = _prompt_messages(
            sections,
            conversation_messages,
            sensitivity=manifest.max_sensitivity,
            context_manifest_id=manifest.context_manifest_id,
        )
        context = AssembledContext(
            messages=messages,
            sections=sections,
            manifest=manifest,
            token_estimate=token_estimate,
        )
        await self._audit.record_context_assembled(context, causation_id=memory_event.event_id)
        return context

    async def _filter_tool_observation_refs_by_policy(
        self,
        request: ContextAssemblyRequest,
        refs: list[ToolObservationRef],
        dropped_refs: list[ContextDroppedRef],
    ) -> list[ToolObservationRef]:
        kept: list[ToolObservationRef] = []
        for ref in refs:
            source_ref = f"tool_observation:{ref.tool_call_id}"
            decision = await self._policy.evaluate_context_inclusion(
                ContextPolicyRequest(source_ref=source_ref, sensitivity=ref.sensitivity),
            )
            await self._audit.record_policy_decision(
                request,
                source_ref=source_ref,
                decision=decision,
                sensitivity=ref.sensitivity,
            )
            if not decision.allowed:
                dropped_refs.append(
                    ContextDroppedRef(
                        kind="tool_observation",
                        ref_id=ref.tool_call_id,
                        reason=_dropped_reason(ref.sensitivity, decision),
                    ),
                )
                continue
            kept.append(ref)
        return kept

    async def _filter_content_hits_by_policy(
        self,
        request: ContextAssemblyRequest,
        hits: list[ContentHit],
        dropped_refs: list[ContextDroppedRef],
    ) -> list[ContentHit]:
        kept: list[ContentHit] = []
        for hit in hits:
            source_ref = f"content:{hit.chunk_id}"
            decision = await self._policy.evaluate_context_inclusion(
                ContextPolicyRequest(source_ref=source_ref, sensitivity=hit.sensitivity),
            )
            await self._audit.record_policy_decision(
                request,
                source_ref=source_ref,
                decision=decision,
                sensitivity=hit.sensitivity,
            )
            if not decision.allowed:
                dropped_refs.append(
                    ContextDroppedRef(
                        kind="content",
                        ref_id=hit.chunk_id,
                        reason=_dropped_reason(hit.sensitivity, decision),
                    ),
                )
                continue
            kept.append(hit)
        return kept

    async def _retrieve_content_hits(
        self,
        request: ContextAssemblyRequest,
        dropped_refs: list[ContextDroppedRef],
    ) -> list[ContentHit]:
        decision = await self._policy.evaluate_capability_request(
            CapabilityPolicyRequest(
                capability=Capability.CONTENT_RETRIEVE,
                risk_classes=frozenset({RiskClass.READ_ONLY}),
                sensitivity=request.current_message_sensitivity,
                permission_mode=request.permission_mode,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                project_namespace=request.active_project_namespace,
                redacted_payload={"query_source": "current_user_message"},
            ),
        )
        if not decision.allowed:
            dropped_refs.append(
                ContextDroppedRef(
                    kind="content",
                    ref_id="*",
                    reason=decision.code,
                ),
            )
            return []
        return await self._content_retrieval.retrieve(
            ContentRetrievalQuery(
                text=request.current_user_message,
                limit=self._content_hit_limit,
                sensitivity=request.current_message_sensitivity,
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                correlation_id=request.request_id,
                causation_id=request.causation_event_id,
            ),
        )

    async def _filter_memory_hits_by_policy(
        self,
        request: ContextAssemblyRequest,
        hits: list[MemoryHit],
        dropped_refs: list[ContextDroppedRef],
    ) -> list[MemoryHit]:
        kept: list[MemoryHit] = []
        for hit in hits:
            source_ref = f"memory:{hit.memory.id}"
            decision = await self._policy.evaluate_context_inclusion(
                ContextPolicyRequest(source_ref=source_ref, sensitivity=hit.memory.sensitivity),
            )
            await self._audit.record_policy_decision(
                request,
                source_ref=source_ref,
                decision=decision,
                sensitivity=hit.memory.sensitivity,
            )
            if not decision.allowed:
                dropped_refs.append(
                    ContextDroppedRef(
                        kind="memory",
                        ref_id=hit.memory.id,
                        reason=_dropped_reason(hit.memory.sensitivity, decision),
                    ),
                )
                continue
            kept.append(hit)
        return kept

    async def _filter_messages_by_policy(
        self,
        request: ContextAssemblyRequest,
        messages: list[ConversationMessage],
        dropped_refs: list[ContextDroppedRef],
    ) -> list[ConversationMessage]:
        kept: list[ConversationMessage] = []
        for message in messages:
            source_ref = f"message:{message.message_id}"
            decision = await self._policy.evaluate_context_inclusion(
                ContextPolicyRequest(source_ref=source_ref, sensitivity=message.sensitivity),
            )
            await self._audit.record_policy_decision(
                request,
                source_ref=source_ref,
                decision=decision,
                sensitivity=message.sensitivity,
            )
            if not decision.allowed:
                dropped_refs.append(
                    ContextDroppedRef(
                        kind="message",
                        ref_id=message.message_id,
                        reason=_dropped_reason(message.sensitivity, decision),
                    ),
                )
                continue
            kept.append(message)
        return kept

    async def _authorize_current_message(self, request: ContextAssemblyRequest) -> None:
        decision = await self._policy.evaluate_context_inclusion(
            ContextPolicyRequest(
                source_ref="current_user_message",
                sensitivity=request.current_message_sensitivity,
            ),
        )
        await self._audit.record_policy_decision(
            request,
            source_ref="current_user_message",
            decision=decision,
            sensitivity=request.current_message_sensitivity,
        )
        if decision.allowed:
            return
        raise ContextPolicyDenied(decision.reason)
