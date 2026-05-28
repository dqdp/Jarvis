# 19 — Embeddings and Retrieval Baseline

## 1. Назначение

Этот документ фиксирует MVP-baseline для embeddings и retrieval в подсистеме долговременной памяти.

Цель Phase 1 — не построить полноценную RAG-платформу, а обеспечить простой, локальный, namespace-aware retrieval по явным long-term memory records.

---

## 2. Scope Phase 1

Phase 1 indexes only explicit long-term `MemoryRecord` content.

Indexed:

```text
memories.content
memories.summary, if present
```

Not indexed in MVP:

```text
raw conversation history
all events
raw documents
PDF chunks
tool observations
logs
files
web pages
codebase
```

Reason:

Memory retrieval and RAG/content retrieval are different subsystems.

---

## 3. Architectural boundary

Embeddings are accessed through `EmbeddingPort`.

Default implementation:

```text
Memory subsystem
  -> EmbeddingPort
      -> ModelRouter.embed(profile=local_embedding)
```

The Memory subsystem must not depend on concrete embedding provider, endpoint, model name or provider-specific API.

---

## 4. Storage baseline

Phase 1 stores memory embeddings in PostgreSQL + pgvector.

The architectural contract is not pgvector. The contract is `MemoryReadPort`.

Recommended table shape:

```sql
memory_embeddings (
  memory_id uuid not null references memories(id),
  embedding_profile text not null,
  embedding_model text not null,
  embedding_dimension int not null,
  content_hash text not null,
  embedding vector(...),
  created_at timestamptz not null default now(),

  primary key (memory_id, embedding_profile)
)
```

`embedding_profile` is included to allow future model/profile migration.

---

## 5. Memory indexing status

`memories` should include:

```text
content_hash
indexing_status
```

Allowed indexing statuses:

```text
indexed
embedding_pending
embedding_failed
```

MVP rule:

```text
Memory record may exist without valid embedding.
Such memory is visible in memory management APIs,
but is not returned by vector retrieval until indexed.
```

---

## 6. Embedding generation

Phase 1 uses synchronous embedding generation on memory create/update.

Create memory flow:

```text
validate memory
write memory record
generate embedding synchronously
write memory_embeddings
set indexing_status=indexed
```

If embedding generation fails:

```text
memory record is still created
indexing_status=embedding_failed
memory.embedding.failed event is emitted
memory is excluded from vector retrieval
```

Reason:

Do not require queue/job infrastructure in Phase 1, but do not lose valid memory records when embedding backend is temporarily unavailable.

---

## 7. Re-embedding

Re-embedding is required when:

```text
memory.content changes
memory.summary changes
embedding_profile changes
embedding model/dimension changes
content_hash mismatch is detected
```

Phase 1:

```text
update_memory recomputes embedding synchronously if content/summary changed
```

If re-embedding fails after content update:

```text
old embedding is considered stale
memory.indexing_status=embedding_failed
retrieval excludes stale embedding through content_hash mismatch
```

---

## 8. Retrieval query

Phase 1 does not use LLM-generated retrieval queries.

Retrieval text is built from:

```text
current user message
optional minimal recent-turn context
active project label / namespace hint
```

No additional model call is required for retrieval query rewriting in MVP.

---

## 9. Namespace-aware retrieval

Retrieval must be namespace-aware.

Input includes:

```text
namespaces
include_statuses=["active"]
exclude_sensitivity=["secret"]
limit
max_per_namespace
```

Default active namespaces are selected by ContextAssembler / NamespaceSelector.

Retrieval must not search globally across all memories unless explicitly requested by an admin/debug operation.

---

## 10. Retrieval limits

MVP baseline:

```yaml
memory_retrieval:
  max_hits_total: 8
  max_hits_per_namespace: 4
  min_score: null
```

No hard `min_score` is used in MVP because embedding score calibration depends on model/backend.

Scores should be logged for future tuning.

---

## 11. Ranking

MVP ranking:

```text
primary: vector score
secondary: importance desc
tertiary: updated_at desc
```

No complex relevance formula is defined in Phase 1.

---

## 12. Reranking and hybrid search

Not in MVP:

```text
reranker
BM25
hybrid lexical/vector search
multi-query retrieval
LLM query rewriting
score calibration
```

These can be added post-MVP behind `MemoryReadPort` and/or inside ContextAssembler strategies.

---

## 13. Retrieval failure behavior

Memory retrieval failure is non-fatal for normal chat.

If retrieval fails:

```text
emit memory.retrieval.failed
assemble degraded context without long-term memory
emit context.assembled with degraded=true
continue with recent conversation and system/runtime context
```

If the user explicitly asks for past memory, future behavior may choose to return a controlled error instead of degraded answer. This is deferred.

---

## 14. Events

Minimum events:

```text
memory.embedding.created
memory.embedding.failed
memory.retrieved
memory.retrieval.failed
```

`memory.retrieved` should include memory IDs, namespaces, scores, retrieval strategy and whether hits were used in context.

---

## 15. MVP vs deferred

MVP includes:

```text
local_embedding profile
EmbeddingPort
sync embedding on memory create/update
memory_embeddings table
content_hash
indexing_status
namespace-aware vector retrieval
active-only retrieval
max_hits_total / max_hits_per_namespace
score + importance + recency ordering
retrieval failure degraded mode
```

Deferred:

```text
reranker
hybrid search
BM25
LLM query rewriting
multi-query retrieval
bulk background reindex
embedding model migration workflow
advanced score calibration
document chunk retrieval
event log retrieval
conversation history vector indexing
```
