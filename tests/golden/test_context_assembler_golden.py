from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assistant_core.context_assembly.deterministic import (
    ContextPolicyDenied,
    DeterministicContextAssembler,
)
from assistant_core.domain.context import ContextAssemblyRequest
from assistant_core.domain.conversations import ConversationMessage, RecentMessagesQuery
from assistant_core.domain.events import EventType
from assistant_core.domain.memory import (
    IndexingStatus,
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
)
from assistant_core.domain.messages import MessageRole
from assistant_core.domain.policy import ContextPolicyRequest, PolicyDecision
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.events.in_memory import InMemoryEventLog
from assistant_core.ports.memory import MemoryRetrievalError
from assistant_core.ports.event_log import EventFilter


pytestmark = pytest.mark.golden


class FakeConversationStore:
    def __init__(self, messages: list[ConversationMessage]) -> None:
        self.messages = messages

    async def load_recent_messages(self, query: RecentMessagesQuery) -> list[ConversationMessage]:
        return self.messages[-query.limit :]


class FakeMemoryRead:
    def __init__(self, hits: list[MemoryHit], *, fail: bool = False) -> None:
        self.hits = hits
        self.fail = fail
        self.queries: list[MemoryQuery] = []

    async def retrieve(self, query: MemoryQuery) -> list[MemoryHit]:
        self.queries.append(query)
        if self.fail:
            raise MemoryRetrievalError("boom")
        return self.hits


class FakePolicy:
    def __init__(self, *, deny_sensitivity: set[Sensitivity] | None = None) -> None:
        self.deny_sensitivity = deny_sensitivity or {Sensitivity.SECRET}
        self.context_requests: list[ContextPolicyRequest] = []

    async def evaluate_model_request(self, request):
        return PolicyDecision(True, "allowed", "model request is allowed")

    async def evaluate_memory_write(self, request):
        return PolicyDecision(True, "allowed", "memory write is allowed")

    async def evaluate_context_inclusion(self, request: ContextPolicyRequest):
        self.context_requests.append(request)
        if request.sensitivity in self.deny_sensitivity:
            return PolicyDecision(False, "sensitivity_denied", "sensitivity is denied")
        return PolicyDecision(True, "allowed", "context source is allowed")


def _request(**overrides) -> ContextAssemblyRequest:
    base = ContextAssemblyRequest(
        request_id="11111111-1111-1111-1111-111111111111",
        conversation_id="22222222-2222-2222-2222-222222222222",
        user_id="user-1",
        current_user_message="current question",
        active_project_namespace="project.personal_assistant",
        loop_strategy="memory_augmented_answer",
        model_profile="local_main",
    )
    return replace(base, **overrides)


def _message(
    label: str,
    role: MessageRole = MessageRole.USER,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> ConversationMessage:
    now = datetime.now(UTC)
    return ConversationMessage(
        message_id=f"msg-{label}",
        conversation_id="22222222-2222-2222-2222-222222222222",
        request_id=None,
        event_id=None,
        client_message_id=None,
        role=role,
        content=f"{label} message",
        content_hash=f"sha256:{label}",
        sensitivity=sensitivity,
        created_at=now,
        metadata={},
    )


def _memory(
    label: str,
    namespace: str = "project.personal_assistant",
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        id=f"mem-{label}",
        namespace=namespace,
        memory_type=MemoryType.FACT,
        content=f"{label} memory",
        summary=None,
        content_hash=f"sha256:{label}",
        sensitivity=sensitivity,
        confidence=0.9,
        importance=0.7,
        status=MemoryStatus.ACTIVE,
        indexing_status=IndexingStatus.INDEXED,
        source_event_ids=[],
        supersedes_memory_ids=[],
        superseded_by_memory_id=None,
        revision=1,
        created_at=now,
        updated_at=now,
        archived_at=None,
        archive_reason=None,
        valid_from=None,
        valid_until=None,
        metadata={},
    )


def _assembler(
    *,
    messages: list[ConversationMessage] | None = None,
    memories: list[MemoryHit] | None = None,
    memory_fails: bool = False,
    event_log: InMemoryEventLog | None = None,
    policy: FakePolicy | None = None,
) -> DeterministicContextAssembler:
    return DeterministicContextAssembler(
        conversation_store=FakeConversationStore(messages or []),
        memory_read=FakeMemoryRead(memories or [], fail=memory_fails),
        event_log=event_log or InMemoryEventLog(),
        policy=policy or FakePolicy(),
    )


def test_context_golden_fixed_section_order() -> None:
    context = asyncio.run(_assembler().assemble(_request()))

    assert [section.name for section in context.sections] == [
        "system_identity",
        "runtime_rules",
        "user_preferences",
        "working_style",
        "project_or_environment_memory",
        "recent_conversation",
        "current_user_message",
        "output_contract",
    ]


def test_includes_current_user_message() -> None:
    context = asyncio.run(_assembler().assemble(_request()))

    assert context.messages[-1].role == MessageRole.USER
    assert context.messages[-1].content[0].text == "current question"


def test_includes_recent_conversation_window() -> None:
    context = asyncio.run(
        _assembler(messages=[_message("first"), _message("second", MessageRole.ASSISTANT)])
        .assemble(_request()),
    )

    assert [message.content[0].text for message in context.messages] == [
        "first message",
        "second message",
        "current question",
    ]


def test_applies_max_messages() -> None:
    context = asyncio.run(
        _assembler(messages=[_message("one"), _message("two"), _message("three")]).assemble(
            _request(max_messages=2),
        ),
    )

    assert context.manifest.used_message_ids == ["msg-two", "msg-three"]
    assert context.manifest.dropped_refs[0].reason == "max_messages"


def test_applies_token_budget() -> None:
    messages = [
        _message("older very long long long long long"),
        _message("newer very long long long long long"),
    ]
    context = asyncio.run(
        _assembler(messages=messages).assemble(_request(max_input_tokens=18)),
    )

    assert "current question" in context.messages[-1].content[0].text
    assert any(ref.reason == "token_budget" for ref in context.manifest.dropped_refs)


def test_drops_oldest_messages_first() -> None:
    context = asyncio.run(
        _assembler(messages=[_message("old"), _message("middle"), _message("new")]).assemble(
            _request(max_messages=2),
        ),
    )

    assert context.manifest.used_message_ids == ["msg-middle", "msg-new"]
    assert context.manifest.dropped_refs[0].ref_id == "msg-old"


def test_retrieves_active_memories() -> None:
    hit = MemoryHit(memory=_memory("project"), score=0.9)
    context = asyncio.run(_assembler(memories=[hit]).assemble(_request()))

    assert context.manifest.used_memory_ids == ["mem-project"]
    assert "project memory" in _section(context, "project_or_environment_memory").content


def test_excludes_secret_memories_and_messages() -> None:
    context = asyncio.run(
        _assembler(
            messages=[_message("secret", sensitivity=Sensitivity.SECRET)],
            memories=[MemoryHit(memory=_memory("secret", sensitivity=Sensitivity.SECRET), score=1.0)],
        ).assemble(_request()),
    )

    assert "secret message" not in [message.content[0].text for message in context.messages]
    assert context.manifest.used_memory_ids == []
    assert {ref.reason for ref in context.manifest.dropped_refs} >= {"secret"}


def test_context_manifest_contains_used_refs() -> None:
    context = asyncio.run(
        _assembler(
            messages=[_message("recent")],
            memories=[MemoryHit(memory=_memory("project"), score=0.9)],
        ).assemble(_request()),
    )

    assert context.manifest.request_id == "11111111-1111-1111-1111-111111111111"
    assert context.manifest.used_message_ids == ["msg-recent"]
    assert context.manifest.used_memory_ids == ["mem-project"]


def test_context_manifest_is_event_recorded_without_raw_prompt() -> None:
    event_log = InMemoryEventLog()

    asyncio.run(_assembler(event_log=event_log).assemble(_request()))
    events = asyncio.run(event_log.query(EventFilter()))

    context_event = next(event for event in events if event.event_type.value == "context.assembled")
    assert "current question" not in str(context_event.payload)
    assert context_event.payload["full_prompt_stored"] is False
    assert "dropped_refs" in context_event.payload
    assert "active_namespaces" in context_event.payload


def test_no_full_prompt_logged_by_default() -> None:
    context = asyncio.run(_assembler().assemble(_request()))

    assert context.manifest.full_prompt_stored is False


def test_degraded_context_when_memory_retrieval_fails() -> None:
    context = asyncio.run(_assembler(memory_fails=True).assemble(_request()))

    assert context.manifest.degraded is True
    assert context.messages[-1].content[0].text == "current question"


def test_context_assembler_uses_policy_port_for_context_inclusion() -> None:
    policy = FakePolicy(deny_sensitivity={Sensitivity.INFRA})
    context = asyncio.run(
        _assembler(
            messages=[_message("infra-msg", sensitivity=Sensitivity.INFRA)],
            memories=[MemoryHit(memory=_memory("infra", sensitivity=Sensitivity.INFRA), score=1.0)],
            policy=policy,
        ).assemble(_request()),
    )

    assert [request.source_ref for request in policy.context_requests] == [
        "current_user_message",
        "memory:mem-infra",
        "message:msg-infra-msg",
    ]
    assert context.manifest.used_memory_ids == []
    assert {ref.reason for ref in context.manifest.dropped_refs} == {"sensitivity_denied"}


def test_context_assembler_records_allowed_policy_decisions() -> None:
    event_log = InMemoryEventLog()

    asyncio.run(
        _assembler(
            messages=[_message("allowed-msg")],
            memories=[MemoryHit(memory=_memory("allowed-memory"), score=1.0)],
            event_log=event_log,
        ).assemble(_request()),
    )
    events = asyncio.run(event_log.query(EventFilter()))

    policy_events = [
        event for event in events if event.event_type == EventType.POLICY_DECISION_RECORDED
    ]
    assert [event.payload["source_ref"] for event in policy_events] == [
        "current_user_message",
        "memory:mem-allowed-memory",
        "message:msg-allowed-msg",
    ]
    assert all(event.payload["allowed"] is True for event in policy_events)


def test_context_assembler_denies_current_secret_message_before_retrieval() -> None:
    memory_read = FakeMemoryRead(
        [MemoryHit(memory=_memory("secret-context", sensitivity=Sensitivity.PROJECT), score=1.0)],
    )
    assembler = DeterministicContextAssembler(
        conversation_store=FakeConversationStore([]),
        memory_read=memory_read,
        event_log=InMemoryEventLog(),
        policy=FakePolicy(),
    )

    with pytest.raises(ContextPolicyDenied):
        asyncio.run(
            assembler.assemble(
                _request(
                    current_user_message="secret payload",
                    current_message_sensitivity=Sensitivity.SECRET,
                ),
            ),
        )

    assert memory_read.queries == []


def test_memory_retrieved_event_is_recorded_with_causation() -> None:
    event_log = InMemoryEventLog()

    asyncio.run(
        _assembler(
            memories=[MemoryHit(memory=_memory("project"), score=0.9)],
            event_log=event_log,
        ).assemble(_request(causation_event_id="33333333-3333-3333-3333-333333333333")),
    )
    events = asyncio.run(event_log.query(EventFilter()))
    memory_event = next(event for event in events if event.event_type.value == "memory.retrieved")
    context_event = next(event for event in events if event.event_type.value == "context.assembled")

    assert memory_event.causation_id == "33333333-3333-3333-3333-333333333333"
    assert memory_event.payload["used_memory_ids"] == ["mem-project"]
    assert context_event.causation_id == memory_event.event_id


def test_serialized_context_fixture_matches_output() -> None:
    context = asyncio.run(
        _assembler(
            messages=[_message("recent")],
            memories=[MemoryHit(memory=_memory("project"), score=0.9)],
        ).assemble(_request()),
    )

    fixture = json.loads(
        Path("tests/golden/fixtures/context_assembler/basic_context.json").read_text(
            encoding="utf-8",
        ),
    )
    assert _serialized_context(context) == fixture


def _section(context, name: str):
    return next(section for section in context.sections if section.name == name)


def _serialized_context(context) -> dict:
    return {
        "messages": [
            {
                "role": message.role.value,
                "content": [part.text for part in message.content],
                "sensitivity": message.sensitivity.value,
            }
            for message in context.messages
        ],
        "sections": [
            {
                "name": section.name,
                "content": section.content,
                "token_estimate": section.token_estimate,
                "source_refs": section.source_refs,
            }
            for section in context.sections
        ],
        "manifest": {
            "context_manifest_id": context.manifest.context_manifest_id,
            "request_id": context.manifest.request_id,
            "conversation_id": context.manifest.conversation_id,
            "loop_strategy": context.manifest.loop_strategy,
            "model_profile": context.manifest.model_profile,
            "section_names": context.manifest.section_names,
            "used_message_ids": context.manifest.used_message_ids,
            "used_memory_ids": context.manifest.used_memory_ids,
            "dropped_refs": [
                {"kind": ref.kind, "ref_id": ref.ref_id, "reason": ref.reason}
                for ref in context.manifest.dropped_refs
            ],
            "token_estimate": context.manifest.token_estimate,
            "active_namespaces": context.manifest.active_namespaces,
            "retrieval_parameters": context.manifest.retrieval_parameters,
            "max_sensitivity": context.manifest.max_sensitivity.value,
            "sources_by_sensitivity": context.manifest.sources_by_sensitivity,
            "degraded": context.manifest.degraded,
            "full_prompt_stored": context.manifest.full_prompt_stored,
        },
        "token_estimate": context.token_estimate,
    }
