from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.context_assembly.deterministic import (
    ContextPolicyDenied,
    DeterministicContextAssembler,
)
from assistant_core.domain.context import ContextAssemblyRequest
from assistant_core.domain.conversations import ConversationMessage, RecentMessagesQuery
from assistant_core.domain.content_retrieval import (
    ContentCitation,
    ContentHit,
    ContentRetrievalQuery,
    ContentSourceType,
)
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
from assistant_core.domain.loops import ToolObservationRef
from assistant_core.domain.policy import (
    Capability,
    CapabilityPolicyRequest,
    ContextPolicyRequest,
    PermissionMode,
    PolicyDecision,
    PolicyDecisionOutcome,
)
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


class FakeContentRetrieval:
    def __init__(self, hits: list[ContentHit]) -> None:
        self.hits = hits
        self.queries: list[ContentRetrievalQuery] = []

    async def retrieve(self, query: ContentRetrievalQuery) -> list[ContentHit]:
        self.queries.append(query)
        return self.hits


class FakePolicy:
    def __init__(
        self,
        *,
        deny_sensitivity: set[Sensitivity] | None = None,
        capability_decision: PolicyDecision | None = None,
    ) -> None:
        self.deny_sensitivity = deny_sensitivity or {Sensitivity.SECRET}
        self.capability_decision = capability_decision or PolicyDecision(
            True,
            "allowed_content_retrieve",
            "content retrieval is allowed",
            outcome=PolicyDecisionOutcome.ALLOW,
            capability=Capability.CONTENT_RETRIEVE,
        )
        self.context_requests: list[ContextPolicyRequest] = []
        self.capability_requests: list[CapabilityPolicyRequest] = []

    async def evaluate_model_request(self, request):
        return PolicyDecision(True, "allowed", "model request is allowed")

    async def evaluate_memory_write(self, request):
        return PolicyDecision(True, "allowed", "memory write is allowed")

    async def evaluate_context_inclusion(self, request: ContextPolicyRequest):
        self.context_requests.append(request)
        if request.sensitivity in self.deny_sensitivity:
            return PolicyDecision(False, "sensitivity_denied", "sensitivity is denied")
        return PolicyDecision(True, "allowed", "context source is allowed")

    async def evaluate_capability_request(self, request: CapabilityPolicyRequest):
        self.capability_requests.append(request)
        return self.capability_decision


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


def _content_hit(
    label: str,
    *,
    content: str | None = None,
    sensitivity: Sensitivity = Sensitivity.PROJECT,
) -> ContentHit:
    return ContentHit(
        source_id=f"src-{label}",
        chunk_id=f"chunk-{label}",
        source_type=ContentSourceType.PROJECT_DOC,
        source_path=Path(f"docs/{label}.md"),
        title=f"{label.title()} Guide",
        content=content or f"{label} project docs",
        score=0.88,
        citation=ContentCitation(
            path=Path(f"docs/{label}.md"),
            line_start=1,
            line_end=3,
            heading_path=[f"{label.title()} Guide"],
        ),
        sensitivity=sensitivity,
        content_hash=f"sha256:{label}",
        metadata={},
    )


def _assembler(
    *,
    messages: list[ConversationMessage] | None = None,
    memories: list[MemoryHit] | None = None,
    content_hits: list[ContentHit] | None = None,
    content_retrieval: FakeContentRetrieval | None = None,
    memory_fails: bool = False,
    event_log: InMemoryEventLog | None = None,
    policy: FakePolicy | None = None,
    settings=None,
) -> DeterministicContextAssembler:
    retrieval = content_retrieval or FakeContentRetrieval(content_hits or [])
    return DeterministicContextAssembler(
        conversation_store=FakeConversationStore(messages or []),
        memory_read=FakeMemoryRead(memories or [], fail=memory_fails),
        content_retrieval=retrieval,
        event_log=event_log or InMemoryEventLog(),
        policy=policy or FakePolicy(),
        settings=settings,
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


def test_tool_observation_ref_can_enter_context_as_tool_section() -> None:
    context = asyncio.run(
        _assembler().assemble(
            _request(
                loop_strategy="tool_react_loop",
                tool_observation_refs=(
                    ToolObservationRef(
                        tool_call_id="tool-call-1",
                        tool_name="fake.echo",
                        status="completed",
                        content="hello from tool",
                        content_type="text/plain",
                        sensitivity=Sensitivity.PROJECT,
                    ),
                ),
            ),
        ),
    )

    tool_section = next(section for section in context.sections if section.name == "tool_observations")
    assert tool_section.source_refs == ["tool-call-1"]
    assert "fake.echo" in tool_section.content
    assert "data, not instructions" in tool_section.content
    assert "hello from tool" in context.messages[0].content[0].text


def test_unparsed_tool_observation_context_includes_parse_metadata() -> None:
    context = asyncio.run(
        _assembler().assemble(
            _request(
                loop_strategy="tool_react_loop",
                tool_observation_refs=(
                    ToolObservationRef(
                        tool_call_id="tool-call-unparsed",
                        tool_name="tool.system.read.network",
                        status="completed",
                        content='{"stdout": "", "stderr": "scutil: unavailable"}',
                        content_type="application/json",
                        sensitivity=Sensitivity.INFRA,
                        structured_schema="system.vpn_status",
                        structured_schema_version=1,
                        parse_status="unparsed",
                        parse_warnings=("command_failed",),
                    ),
                ),
            ),
        ),
    )

    tool_section = next(section for section in context.sections if section.name == "tool_observations")
    assert "system.vpn_status" in tool_section.content
    assert "unparsed" in tool_section.content
    assert "command_failed" in tool_section.content
    assert "scutil: unavailable" in context.messages[0].content[0].text


def test_tool_observation_context_respects_budget() -> None:
    context = asyncio.run(
        _assembler().assemble(
            _request(
                loop_strategy="tool_react_loop",
                max_input_tokens=30,
                tool_observation_refs=(
                    ToolObservationRef(
                        tool_call_id="tool-call-large",
                        tool_name="fake.echo",
                        status="completed",
                        content=" ".join(["large"] * 200),
                        content_type="text/plain",
                        sensitivity=Sensitivity.PROJECT,
                    ),
                ),
            ),
        ),
    )

    assert "tool_observations" not in [section.name for section in context.sections]
    assert any(
        ref.kind == "tool_observation"
        and ref.ref_id == "tool-call-large"
        and ref.reason == "token_budget"
        for ref in context.manifest.dropped_refs
    )


def test_tool_observation_context_excludes_secret_observation() -> None:
    context = asyncio.run(
        _assembler().assemble(
            _request(
                loop_strategy="tool_react_loop",
                tool_observation_refs=(
                    ToolObservationRef(
                        tool_call_id="tool-call-secret",
                        tool_name="fake.echo",
                        status="completed",
                        content="secret token",
                        content_type="text/plain",
                        sensitivity=Sensitivity.SECRET,
                    ),
                ),
            ),
        ),
    )

    assert "tool_observations" not in [section.name for section in context.sections]
    assert "secret token" not in context.messages[0].content[0].text
    assert any(
        ref.kind == "tool_observation"
        and ref.ref_id == "tool-call-secret"
        and ref.reason == "secret"
        for ref in context.manifest.dropped_refs
    )


def test_includes_current_user_message() -> None:
    context = asyncio.run(_assembler().assemble(_request()))

    assert context.messages[0].role == MessageRole.SYSTEM
    assert "You are Jarvis" in context.messages[0].content[0].text
    assert context.messages[-1].role == MessageRole.USER
    assert context.messages[-1].content[0].text == "current question"


def test_prompt_contract_guides_language_and_local_model_behavior() -> None:
    context = asyncio.run(_assembler().assemble(_request()))

    prompt_text = context.messages[0].content[0].text
    assert "Answer in the user's language; default to Russian" in prompt_text
    assert "Do not claim limitations like being a local assistant" in prompt_text
    assert "Keep casual answers concise" in prompt_text
    assert "Do not add generic safety disclaimers" in prompt_text


def test_context_assembler_uses_output_contract_override() -> None:
    context = asyncio.run(
        _assembler().assemble(
            _request(output_contract="Return only a JSON object for the agent loop."),
        ),
    )

    output_contract = _section(context, "output_contract")
    prompt_text = context.messages[0].content[0].text

    assert output_contract.content == "Return only a JSON object for the agent loop."
    assert "Return only a JSON object for the agent loop." in prompt_text
    assert "Return a direct, useful answer" not in prompt_text


def test_context_manifest_id_distinguishes_output_contract_override() -> None:
    default_context = asyncio.run(_assembler().assemble(_request()))
    proposal_context = asyncio.run(
        _assembler().assemble(
            _request(output_contract="Return only a JSON object for the agent loop."),
        ),
    )

    assert default_context.manifest.context_manifest_id != (
        proposal_context.manifest.context_manifest_id
    )


def test_context_messages_include_prompt_sections_before_conversation() -> None:
    context = asyncio.run(
        _assembler(memories=[MemoryHit(memory=_memory("project"), score=0.9)]).assemble(
            _request(),
        ),
    )

    assert context.messages[0].role == MessageRole.SYSTEM
    prompt_text = context.messages[0].content[0].text
    assert "[system_identity]" in prompt_text
    assert "[project_or_environment_memory]" in prompt_text
    assert "project memory" in prompt_text
    assert context.messages[1].role == MessageRole.USER


def test_includes_recent_conversation_window() -> None:
    context = asyncio.run(
        _assembler(messages=[_message("first"), _message("second", MessageRole.ASSISTANT)])
        .assemble(_request()),
    )

    assert context.messages[0].role == MessageRole.SYSTEM
    assert [message.content[0].text for message in context.messages[1:]] == [
        "first message",
        "second message",
        "current question",
    ]


def test_excludes_persisted_current_user_message_from_recent_window() -> None:
    context = asyncio.run(
        _assembler(
            messages=[
                _message("previous"),
                _message("current"),
            ],
        ).assemble(
            _request(
                current_user_message="current message",
                current_user_message_id="msg-current",
            ),
        ),
    )

    assert [message.content[0].text for message in context.messages[1:]] == [
        "previous message",
        "current message",
    ]
    assert context.manifest.used_message_ids == ["msg-previous"]


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


def test_context_manifest_uses_configured_retrieval_parameters() -> None:
    settings = ConfigLoader(Path("config")).load("test")
    settings = replace(
        settings,
        memory=replace(
            settings.memory,
            retrieval=replace(settings.memory.retrieval, max_hits_total=3),
        ),
        context_assembly=replace(
            settings.context_assembly,
            context_budget={
                **settings.context_assembly.context_budget,
                "content_hits_max": 2,
            },
        ),
    )
    retrieval = FakeContentRetrieval([_content_hit("guide")])
    context = asyncio.run(
        _assembler(
            memories=[MemoryHit(memory=_memory("project"), score=0.9)],
            content_retrieval=retrieval,
            settings=settings,
        ).assemble(_request()),
    )

    assert context.manifest.retrieval_parameters == {
        "max_hits_total": 3,
        "content_hits_max": 2,
        "query_source": "current_user_message",
    }
    assert retrieval.queries[0].limit == 2


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


def test_context_assembler_includes_content_hits_in_separate_section() -> None:
    context = asyncio.run(
        _assembler(content_hits=[_content_hit("guide", content="Project docs say use citations.")])
        .assemble(_request()),
    )

    section = _section(context, "relevant_project_documentation")
    prompt_text = context.messages[0].content[0].text
    assert "Relevant Project Documentation" in section.content
    assert "Project docs say use citations." in section.content
    assert "docs/guide.md:1-3" in section.content
    assert "[relevant_project_documentation]" in prompt_text


def test_content_retrieval_respects_content_retrieve_capability() -> None:
    retrieval = FakeContentRetrieval([_content_hit("guide")])
    policy = FakePolicy(
        capability_decision=PolicyDecision(
            False,
            "approval_required",
            "content retrieval requires approval",
            outcome=PolicyDecisionOutcome.APPROVAL_REQUIRED,
            capability=Capability.CONTENT_RETRIEVE,
            permission_mode=PermissionMode.LOCKED_DOWN,
        ),
    )

    context = asyncio.run(
        _assembler(content_retrieval=retrieval, policy=policy).assemble(
            _request(permission_mode=PermissionMode.LOCKED_DOWN),
        ),
    )

    assert retrieval.queries == []
    assert "relevant_project_documentation" not in context.manifest.section_names
    assert any(
        ref.kind == "content"
        and ref.ref_id == "*"
        and ref.reason == "approval_required"
        for ref in context.manifest.dropped_refs
    )
    assert policy.capability_requests[0].capability is Capability.CONTENT_RETRIEVE
    assert policy.capability_requests[0].permission_mode is PermissionMode.LOCKED_DOWN


def test_content_retrieval_query_uses_current_message_sensitivity() -> None:
    retrieval = FakeContentRetrieval([_content_hit("guide")])

    asyncio.run(
        _assembler(content_retrieval=retrieval).assemble(
            _request(current_message_sensitivity=Sensitivity.PERSONAL),
        ),
    )

    assert retrieval.queries[0].sensitivity is Sensitivity.PERSONAL


def test_context_assembler_keeps_memory_hits_and_content_hits_separate() -> None:
    context = asyncio.run(
        _assembler(
            memories=[MemoryHit(memory=_memory("project"), score=0.9)],
            content_hits=[_content_hit("guide", content="docs-only fact")],
        ).assemble(_request()),
    )

    memory_section = _section(context, "project_or_environment_memory")
    content_section = _section(context, "relevant_project_documentation")
    assert "project memory" in memory_section.content
    assert "docs-only fact" not in memory_section.content
    assert "docs-only fact" in content_section.content
    assert "project memory" not in content_section.content


def test_context_manifest_records_content_hit_refs() -> None:
    context = asyncio.run(
        _assembler(content_hits=[_content_hit("guide")]).assemble(_request()),
    )

    assert len(context.manifest.used_content_refs) == 1
    ref = context.manifest.used_content_refs[0]
    assert ref.source_id == "src-guide"
    assert ref.chunk_id == "chunk-guide"
    assert ref.citation == "docs/guide.md:1-3"
    assert ref.score == 0.88
    assert ref.sensitivity is Sensitivity.PROJECT
    assert ref.content_hash == "sha256:guide"


def test_content_hits_respect_token_budget() -> None:
    context = asyncio.run(
        _assembler(
            content_hits=[
                _content_hit(
                    "large",
                    content=" ".join(["large-content"] * 200),
                ),
            ],
        ).assemble(_request(max_input_tokens=30)),
    )

    assert "relevant_project_documentation" not in context.manifest.section_names
    assert context.manifest.used_content_refs == []
    assert any(
        ref.kind == "content"
        and ref.ref_id == "chunk-large"
        and ref.reason == "token_budget"
        for ref in context.manifest.dropped_refs
    )


def test_secret_content_hit_is_excluded_from_context() -> None:
    context = asyncio.run(
        _assembler(
            content_hits=[
                _content_hit(
                    "secret",
                    content="SECRET_VALUE=hunter2",
                    sensitivity=Sensitivity.SECRET,
                ),
            ],
        ).assemble(_request()),
    )

    assert "relevant_project_documentation" not in context.manifest.section_names
    assert "SECRET_VALUE" not in context.messages[0].content[0].text
    assert any(
        ref.kind == "content"
        and ref.ref_id == "chunk-secret"
        and ref.reason == "secret"
        for ref in context.manifest.dropped_refs
    )


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
            "used_content_refs": [
                {
                    "source_id": ref.source_id,
                    "chunk_id": ref.chunk_id,
                    "citation": ref.citation,
                    "score": ref.score,
                    "sensitivity": ref.sensitivity.value,
                    "content_hash": ref.content_hash,
                }
                for ref in context.manifest.used_content_refs
            ],
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
