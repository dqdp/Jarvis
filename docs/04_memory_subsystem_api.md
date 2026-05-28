# 04 — Memory Subsystem API

## 1. Purpose

Документ определяет заменяемую подсистему долговременной памяти.

Memory subsystem — это не PostgreSQL, не pgvector, не vector database и не prompt string. Это логическая подсистема, доступная через стабильные domain-level ports.

## 2. Граница подсистемы

Long-term memory отвечает на вопрос:

> Что система считает устойчивым знанием?

Она не отвечает за:

- сборку текущего prompt context;
- хранение полной истории сообщений;
- хранение runtime checkpoint state;
- хранение immutable event history;
- хранение secrets/credentials.

Связанные подсистемы:

```text
ConversationStorePort
  хранит messages и recent conversation window.

EventLogPort
  хранит immutable historical truth.

ContextAssemblerPort
  собирает ephemeral context для конкретного model call.

MemoryReadPort / MemoryWritePort
  управляют долговременной memory.
```

## 3. Event Log vs Memory

Event log является historical truth.

Memory является интерпретацией.

Пример:

```text
Event:
  user.message: "Хочу вести проектную документацию на русском."

Memory:
  preference: "Пользователь предпочитает русскоязычную проектную документацию."
```

Memory может быть ошибочной, устаревшей или superseded. Event log остается источником для реконструкции.

## 4. Namespaces

Phase 1 использует минимальный явный namespace set.

```text
user.preferences
user.working_style
project.personal_assistant
system.runtime_rules
environment.inference_node
```

Правила:

- namespace registry явный;
- LLM не может auto-create namespaces в Phase 1;
- namespace не используется как deep taxonomy;
- phase/component/ADR/tags хранятся в metadata;
- retrieval namespace-aware;
- новые namespaces добавляются только через документированное решение/ADR.

## 5. Core memory types

Phase 1 использует минимальный универсальный набор memory types:

```text
fact
preference
procedure
summary
```

Специализированные категории не являются core memory types:

```text
architecture_decision
runtime_rule
project_fact
environment_fact
hardware_fact
integration_fact
```

Они выражаются через metadata.

Пример:

```json
{
  "namespace": "project.personal_assistant",
  "memory_type": "fact",
  "content": "Phase 1 uses deterministic memory-augmented workflow, not ReAct.",
  "metadata": {
    "fact_kind": "architecture_decision",
    "adr": "ADR-012",
    "component": "agent_runtime"
  }
}
```

## 6. Namespace × type matrix

```text
user.preferences
  - preference

user.working_style
  - preference
  - procedure
  - summary

project.personal_assistant
  - fact
  - procedure
  - summary

system.runtime_rules
  - fact
  - procedure

environment.inference_node
  - fact
  - procedure
  - summary
```

`MemoryWritePort` validates namespace/type compatibility.

## 7. Memory lifecycle

Memory records use three statuses:

```text
active
archived
superseded
```

Default retrieval uses only `active` records.

### update

Use update when the same memory identity remains valid and only wording, confidence, importance, metadata or provenance changes.

### archive

Use archive when a memory should no longer participate in retrieval and there is no direct replacement.

### supersede

Use supersede when a new memory semantically replaces an older memory.

Hard delete is not part of normal Phase 1 lifecycle. It is reserved for explicit future privacy/purge operations.

## 8. Ports

### MemoryReadPort

Used by Agent Runtime indirectly through ContextAssembler and by admin/debug APIs.

```python
class MemoryReadPort(Protocol):
    async def retrieve(self, query: MemoryQuery) -> list[MemoryHit]: ...
    async def get_memory(self, memory_id: MemoryId) -> MemoryRecord | None: ...
    async def list_memories(self, filter: MemoryFilter, page: PageRequest) -> Page[MemoryRecord]: ...
```

### MemoryWritePort

Used by admin/manual API and controlled workflows. The autonomous conversation graph does not mutate memory directly in Phase 1.

```python
class MemoryWritePort(Protocol):
    async def create_memory(self, command: CreateMemoryCommand) -> MemoryRecord: ...
    async def update_memory(self, command: UpdateMemoryCommand) -> MemoryRecord: ...
    async def archive_memory(self, memory_id: MemoryId, reason: str) -> None: ...
    async def supersede_memory(self, command: SupersedeMemoryCommand) -> MemoryRecord: ...
```

### Future MemoryCandidatePort

```python
class MemoryCandidatePort(Protocol):
    async def propose_candidate(self, candidate: MemoryCandidate) -> MemoryCandidateId: ...
    async def approve_candidate(self, candidate_id: MemoryCandidateId) -> MemoryRecord: ...
    async def reject_candidate(self, candidate_id: MemoryCandidateId, reason: str) -> None: ...
```

### Future MemoryConsolidationPort

```python
class MemoryConsolidationPort(Protocol):
    async def consolidate_period(self, command: ConsolidatePeriodCommand) -> ConsolidationResult: ...
```

## 9. Domain Types

### MemoryRecord

```python
@dataclass(frozen=True)
class MemoryRecord:
    id: str
    namespace: str
    memory_type: Literal["fact", "preference", "procedure", "summary"]
    content: str
    summary: str | None
    content_hash: str
    sensitivity: Literal["public", "project", "personal", "infra", "secret"]
    confidence: float
    importance: float
    status: Literal["active", "archived", "superseded"]
    indexing_status: Literal["indexed", "embedding_pending", "embedding_failed"]
    source_event_ids: list[str]
    supersedes_memory_ids: list[str]
    superseded_by_memory_id: str | None
    revision: int
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    archive_reason: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    metadata: dict[str, Any]
```

`MemoryRecord` is a domain contract, not a direct copy of the PostgreSQL row.
However, `sensitivity`, `content_hash` and `indexing_status` are mandatory
domain-contract fields because policy enforcement, stale-embedding exclusion
and retrieval eligibility depend on them.

### MemoryQuery

```python
@dataclass(frozen=True)
class MemoryQuery:
    user_id: str
    text: str
    conversation_id: str | None
    namespaces: list[str]
    memory_types: list[str] | None
    include_statuses: list[str] = field(default_factory=lambda: ["active"])
    sensitivity_allowlist: list[str] = field(default_factory=list)
    limit: int = 8
    min_score: float | None = None
    metadata_filter: dict[str, Any] | None = None
```

### MemoryHit

```python
@dataclass(frozen=True)
class MemoryHit:
    memory: MemoryRecord
    score: float
    reason: str | None
    retrieval_strategy: str
```

## 10. Phase 1 Rules

- Conversation graph does not mutate memory directly.
- ContextAssembler retrieves active memories through MemoryReadPort.
- Manual memory creation is allowed through controlled API.
- memory_candidates exist in schema/domain model.
- Automatic memory extraction is not Phase 1 acceptance criterion.
- Every memory must have provenance.
- Every lifecycle change emits an event log record.

## 11. Future Retrieval Implementations

Possible future adapters:

- PostgreSQL + pgvector;
- PostgreSQL metadata + Qdrant vector index;
- hybrid BM25 + vector + reranker;
- external memory service;
- graph memory;
- multimodal memory store.

## Sensitivity rules for memory

Every `MemoryRecord` has a sensitivity label.

Phase 1 uses:

```text
public
project
personal
infra
secret
```

Default sensitivity is assigned by namespace registry:

```text
user.preferences              → personal
user.working_style             → personal
project.personal_assistant     → project
system.runtime_rules           → project
environment.inference_node     → infra
```

`MemoryWritePort` must reject `secret` memory writes.

Special rule:

> Secrets are not long-term memories. They must not be stored in `memories` or `memory_candidates` as raw content.

If a user asks to remember a secret, Phase 1 should reject the memory write and emit a redacted audit event.


## Embedding and retrieval baseline

Phase 1 memory retrieval indexes only explicit long-term memory records.

It does not index raw documents, events, raw conversation history, logs, files or tool observations.

Memory subsystem uses `EmbeddingPort` for embedding generation.

Default implementation delegates to `ModelRouter.embed(local_embedding)`.

Retrieval defaults:

```text
active memories only
namespace-aware
max_hits_total=8
max_hits_per_namespace=4
no hard min_score
no reranker
no hybrid search
```

A memory record may exist without valid embedding if embedding generation failed. Such records have `indexing_status=embedding_failed` and are excluded from retrieval until reindexed.

Full RAG is not part of Memory subsystem.
