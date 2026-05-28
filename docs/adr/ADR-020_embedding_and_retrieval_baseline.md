# ADR-020 — Embedding and Retrieval Baseline

## Status

Accepted.

## Context

Phase 1 needs long-term memory retrieval, but must not become a full RAG platform.

Memory retrieval should work over explicit MemoryRecords and remain replaceable behind `MemoryReadPort`.

## Decision

Phase 1 indexes only explicit long-term `MemoryRecord` content/summary.

Phase 1 does not vector-index:

```text
documents
events
raw conversation history
logs
files
tool observations
web pages
codebase
```

Embeddings are accessed through `EmbeddingPort`.

Default implementation:

```text
EmbeddingPort -> ModelRouter.embed(local_embedding)
```

Embeddings are generated synchronously on memory create/update.

A valid memory record may be created even if embedding generation fails. In that case:

```text
indexing_status=embedding_failed
memory is excluded from retrieval
memory.embedding.failed event is emitted
```

Retrieval baseline:

```text
active memories only
namespace-aware
max_hits_total=8
max_hits_per_namespace=4
no hard min_score
no reranker
no hybrid search
ranking = adapter relevance score, then importance, then recency
```

The preferred PostgreSQL adapter can use pgvector similarity as the adapter
relevance score. The implemented MVP may use deterministic lexical ranking over
PostgreSQL-stored embeddings while keeping the same `MemoryReadPort` and
`MemoryHit` contract.

Retrieval failure is non-fatal for normal chat. ContextAssembler may proceed in degraded mode without long-term memory.

Stale embeddings are excluded through content hash mismatch.

## Rationale

This provides useful memory retrieval for MVP while avoiding RAG scope creep.

Synchronous embedding avoids queue infrastructure in Phase 1.

Allowing memory creation despite embedding failure preserves user intent and allows later reindexing.

## Consequences

Positive:

- small MVP;
- local memory retrieval works;
- embedding backend replaceable;
- no document/RAG complexity in Phase 1;
- retrieval failures do not break normal chat.

Trade-offs:

- no advanced retrieval quality;
- no reranker;
- no hybrid search;
- no background indexing;
- memory may exist temporarily without vector search availability.

## Deferred

- reranking;
- hybrid/BM25 search;
- LLM query rewriting;
- multi-query retrieval;
- bulk/background reindex;
- embedding model migration workflow;
- document/content retrieval;
- event log retrieval;
- conversation vector indexing.
