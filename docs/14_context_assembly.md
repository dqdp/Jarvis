# 14 — Context Assembly

## 1. Purpose

Context Assembly is a first-class core subsystem in Phase 1.

It answers the question:

> What exactly should the model see for this specific model call?

It is separate from long-term memory, conversation history, runtime checkpoints and event log.

## 2. Core decision

Phase 1 introduces `ContextAssemblerPort` as the facade for prompt context construction.

Agent Runtime must not manually concatenate:

- recent messages;
- retrieved memories;
- system/runtime rules;
- policy constraints;
- prompt templates;
- output contracts.

Instead, Agent Runtime calls ContextAssembler.

## 3. Facade/Internal Separation

The facade is stable:

```python
class ContextAssemblerPort(Protocol):
    async def assemble(self, request: ContextAssemblyRequest) -> AssembledContext: ...
```

Internal implementation is replaceable.

Phase 1 implementation:

```text
deterministic assembler
  -> fixed prompt sections
  -> recent conversation window
  -> namespace-aware active memory retrieval
  -> static runtime rules
  -> token budget
```

Future implementations may add:

```text
context planning
learned reranking
compression
rolling summaries
multi-pass assembly
tool-aware context
multimodal context
```

Agent Runtime contract must not change when these internal mechanisms change.

## 4. What current context is not

Current prompt context is not:

- long-term memory;
- event log;
- conversation store;
- graph checkpoint;
- a permanent document;
- a source of historical truth.

It is an ephemeral artifact of one model call.

## 5. Inputs

ContextAssembler uses:

- current user message;
- conversation id;
- user id;
- active project/context namespace;
- loop strategy;
- model profile;
- token budget;
- ConversationStorePort;
- MemoryReadPort;
- PolicyPort;
- prompt templates / runtime rules.

## 6. Outputs

```python
@dataclass(frozen=True)
class AssembledContext:
    messages: list[ChatMessage]
    sections: list[ContextSection]
    manifest: ContextManifest
    token_estimate: int
```

`ContextManifest` is an explicit domain object, not an unstructured metadata
dictionary. Phase 1 `ContextManifest` includes at least:

```python
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
    max_sensitivity: Literal["public", "project", "personal", "infra", "secret"]
    sources_by_sensitivity: dict[str, list[str]]
    degraded: bool
    full_prompt_stored: bool = False
```

## 7. Canonical prompt sections

Phase 1 uses stable section order:

```text
1. System / runtime rules
2. User preferences and working style
3. Relevant project/environment memories
4. Recent conversation window
5. Current user message
6. Output contract
```

## 8. Phase 1 context policy

- Context is built fresh for each model call.
- Only `active` memories can be included by default.
- Retrieval is namespace-aware.
- Recent conversation window is budgeted.
- Prompt sections have deterministic order.
- Token budget is explicit.
- All used memories are logged via `memory.retrieved` event.
- Assembled context may be logged only with policy-aware redaction.


## 9. Phase 1 accepted decisions

The following decisions are accepted for Phase 1:

1. Raw full prompt logging is disabled by default.
   The system stores `ContextManifest` and model invocation metadata, not the complete prompt text, unless explicit debug mode is enabled.

2. Retrieval query generation does not require an extra LLM call.
   Phase 1 uses the current user message plus minimal recent-turn context as retrieval text. LLM-generated retrieval queries are deferred.

3. Reranking is not part of the Phase 1 MVP.
   Retrieval uses namespace-aware MemoryReadPort results and simple ranking by relevance/importance. A pgvector adapter may provide vector relevance later behind the same ContextAssembler facade.

4. Automatic rolling summaries are not part of the Phase 1 MVP.
   Recent context is managed by a bounded recent-message window. Rolling summaries and compression are deferred to later context-management or sleep/consolidation workflows.

## 10. Context sources

Phase 1 context is assembled from these sources:

```text
1. system identity / assistant role
2. hard runtime rules
3. active user preferences and working style memories
4. active project/environment memories
5. recent conversation window
6. current user message
7. output contract
```

The current user message, system identity and hard runtime rules are non-droppable. Other sections may be trimmed under token budget rules.

## 11. Context manifest

The ContextAssembler must return a manifest describing how context was assembled.

The manifest should include at least:

- request id;
- conversation id;
- loop strategy;
- model profile;
- section names;
- used message ids;
- used memory ids;
- dropped items and reasons;
- token estimate;
- active namespaces;
- retrieval parameters.

The manifest is safe to persist by default. Raw assembled prompt persistence is disabled by default and requires explicit debug configuration.

## 12. Retrieval baseline

Phase 1 uses a minimal deterministic retrieval baseline:

```text
retrieval_text = current_user_message + optional recent turn context
active_namespaces = selected by deterministic namespace policy
status_filter = active only
max_hits_total = small configurable limit
max_hits_per_namespace = small configurable limit
reranker = disabled
llm_query_rewrite = disabled
```

Retrieval score calibration is not assumed in Phase 1. Thresholds should be configurable and conservative.

## 13. Recent conversation window

Phase 1 uses recent-message tail selection:

- full conversation history is persisted in ConversationStore;
- only a bounded recent window is inserted into current context;
- selection is limited by message count and token budget;
- older messages are dropped before non-droppable sections;
- rolling summaries are deferred.

## 14. Token budget and trimming

ContextAssembler owns token budget enforcement.

Phase 1 trimming order:

```text
1. low-score or low-priority memories
2. older recent messages
3. optional working-style memories
4. optional output-contract details
```

Never drop:

```text
- system identity;
- hard runtime rules;
- current user message.
```

## 15. Relationship to Memory lifecycle

Memory lifecycle statuses (`active`, `archived`, `superseded`) apply only to long-term memory records.

Current context has no such lifecycle.

ContextAssembler consumes active memories, but does not own memory lifecycle.

## 16. Relationship to Agent Loop

Phase 1 loop:

```text
receive_message
  -> persist user.message
  -> select_loop_strategy
  -> assemble_context
  -> call_model_router
  -> stream_response
  -> persist assistant.message / events
```

`assemble_context` replaces ad-hoc `retrieve_memory + build_prompt` logic.

## 17. Contract tests

Minimum contract tests:

- assembler includes current user message;
- assembler includes recent messages according to budget;
- assembler retrieves only active memories;
- assembler respects active namespaces;
- assembler returns used_memory_ids and used_message_ids;
- assembler emits/returns enough metadata for audit;
- assembler does not expose adapter-specific objects;
- assembler stores/returns ContextManifest without raw prompt by default;
- assembler does not require LLM query rewriting in Phase 1;
- assembler respects trimming priorities under token pressure.


## 18. Post-MVP follow-up: advanced context management

The following capabilities are explicitly deferred beyond the Phase 1 MVP. They are not part of Phase 1 acceptance criteria, but the Phase 1 design must not block them.

### 18.1 Provider-neutral context representation

Phase 1 may send text-only chat messages to the local model backend, but the internal context representation must not be tied to a specific provider request format.

Required direction:

```text
ContextAssembler produces provider-neutral internal messages.
ModelRouter converts internal messages into provider-specific requests.
```

This prevents the assistant runtime from depending on OpenAI/vLLM/Ollama-specific message dictionaries.

Future internal content model should support typed content parts:

```text
text
image_ref
file_ref
audio_ref
tool_result_ref
structured_data_ref
```

Phase 1 may implement only `text`, but schemas and domain types should not make future typed parts impossible.

### 18.2 Advanced retrieval query generation

Phase 1 uses current user message plus minimal recent-turn context as retrieval text.

Post-MVP extensions may add:

- deterministic query normalization;
- LLM-generated retrieval query;
- multi-query retrieval;
- project-aware query expansion;
- intent-specific retrieval strategies.

These extensions must remain internal to ContextAssembler and must be reflected in ContextManifest.

### 18.3 Reranking

Phase 1 does not include learned reranking.

Post-MVP extensions may add:

- cross-encoder reranker;
- LLM-based relevance judgment;
- namespace-weighted reranking;
- recency/importance-aware reranking;
- diversity-aware selection.

Reranking must not change the `ContextAssemblerPort` contract.

### 18.4 Rolling summaries and compression

Phase 1 uses bounded recent-message windowing.

Post-MVP extensions may add:

- working conversation summaries;
- rolling summaries;
- extractive compression;
- model-based compression;
- memory-hit compression;
- tool-observation compression.

Working summaries used for current context must remain distinct from long-term memory summaries. Long-term summaries are memory records; working summaries are context-management artifacts.

ContextManifest must record compressed sources and compression strategy when compression is used.

### 18.5 Tool-aware context

When ToolGatewayPort is introduced, ContextAssembler may add sections such as:

```text
available_tools
tool_policy
recent_tool_observations
approval_state
risk_constraints
```

Tool-aware context must be selected according to loop strategy and PolicyPort decisions.

### 18.6 Planner-aware context

When planner-executor loops are introduced, ContextAssembler may add sections such as:

```text
goal
constraints
current_plan
previous_plan_attempts
open_blockers
available_capabilities
budget
approval_constraints
```

Planner context must remain a loop-strategy-specific context policy, not a special case inside AgentRuntime.

### 18.7 Multimodal context

Future versions may include screenshots, PDFs, images, audio transcripts, files and structured documents.

The Phase 1 text-only implementation must not prevent future multimodal context sources. Multimodal expansion should happen through typed content parts and provider-specific conversion in ModelRouter.

### 18.8 Context pipeline extensibility

The internal ContextAssembler implementation may evolve into a pipeline of strategies:

```text
NamespaceSelector
RetrievalQueryBuilder
MemoryRetriever
MemoryReranker
ConversationWindowBuilder
ConversationSummarizer
ContextCompressor
ToolContextBuilder
MultimodalContextBuilder
SectionBuilder
BudgetTrimmer
ManifestBuilder
```

Phase 1 does not need to implement all of these as separate classes, but the design should keep these boundaries visible enough to avoid future rewrites.

### 18.9 Non-goals for Phase 1

The following remain out of MVP scope:

- LLM-generated retrieval queries;
- learned rerankers;
- automatic rolling summaries;
- prompt compression;
- multimodal context;
- tool-aware context;
- planner-aware context;
- provider-specific prompt construction in AgentRuntime.

## 19. Sensitivity-aware context assembly

`ContextAssembler` must enforce the Phase 1 sensitivity policy.

Allowed in local model context:

```text
public
project
personal
infra
```

Forbidden in all prompt contexts:

```text
secret
```

Hard rule:

> `secret` is never included in prompt context, even for local models.

`ContextManifest` must record:

```text
max_sensitivity
sources_by_sensitivity
dropped_secret_sources, if any
```

Full raw prompt logging remains disabled by default.


## 20. PM-07 Content Retrieval integration

ContextAssembler uses:

```text
ConversationStorePort
MemoryReadPort
PolicyPort
ContentRetrievalPort
```

`ContentRetrievalPort` is optional at construction time for tests and narrow
runtime profiles, but the full runtime app wires it for PM-07 project-docs
retrieval.

Content retrieval is policy-gated through `content.retrieve` before query
embedding or storage retrieval.

`MemoryHit` and `ContentHit` must remain different domain objects.

Context sections should keep memory-derived context and source-content-derived context separate.


## 21. Conversation windowing policy

ContextAssembler owns windowing policy.

ConversationStore provides durable messages and recent-message access, but it does not own model-specific token budgeting or trimming strategy.

MVP default:

```yaml
conversation_window:
  max_messages: 12
  max_tokens: 3000
  include_roles: ["user", "assistant"]
  exclude_sensitivity: ["secret"]
  trimming_strategy: drop_oldest_first
```

These are configurable defaults, not hardcoded architectural constants.

Future ContextAssembler implementations may use rolling summaries, salience scoring, planner-aware windows or tool-aware windows behind the same `ContextAssemblerPort`.


## 22. Configuration relation

Context section order, conversation window limits, token budgets, trimming strategy and raw prompt logging flag are config-driven.

MVP defaults are not hardcoded architecture.
