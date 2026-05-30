# ADR-034 — Content Retrieval Subsystem and Project Docs RAG

## Status

Accepted.

## Context

The MVP implements explicit long-term memory retrieval through `MemoryReadPort`.
It does not implement full RAG.

Post-MVP Alpha needs a narrow RAG capability so the assistant can answer from
project documentation and ADRs with citations. This must not turn Memory into a
document-chunk store and must not make `AgentRuntime` or loop strategies query
storage directly.

The first RAG target is project documentation:

```text
README.md
docs/*.md
docs/adr/*.md
```

Out of scope for the first RAG slice:

```text
source code corpus
logs
raw conversations
event log vector indexing
PDFs
web pages
Telegram history
MCP resources
arbitrary user files
secret-like files
```

## Decision

Introduce a separate Content Retrieval subsystem behind `ContentRetrievalPort`.

Memory and Content Retrieval remain separate domains:

```text
Memory subsystem:
  memories
  memory_embeddings
  MemoryReadPort

Content Retrieval subsystem:
  content_sources
  content_chunks
  content_embeddings
  ContentRetrievalPort
```

`ContentHit` is not `MemoryHit`.

Document chunks must not be stored in Memory tables.

The source file remains the source of truth. Chunks and embeddings are derived
artifacts and may be rebuilt.

## ContentRetrievalPort

Initial port shape:

```python
class ContentRetrievalPort(Protocol):
    async def retrieve(self, query: ContentQuery) -> list[ContentHit]: ...
```

Initial query fields:

```text
text
corpus
top_k
allowed_source_types
sensitivity_policy
request_id optional
```

Initial hit fields:

```text
source_id
chunk_id
source_type
title
content
score
citation
sensitivity
content_hash
metadata
```

## Source registry

Initial `content_sources` model:

```text
source_id
source_type = readme | project_doc | adr
uri/path
title
content_hash
last_seen_at
indexed_at
status = active | stale | deleted | failed
sensitivity
metadata
```

Only allowlisted project documentation paths are ingested:

```text
README.md
docs/*.md
docs/adr/*.md
```

Secret-like paths are denied even if they match a broad markdown pattern.

## Chunk model

Initial markdown chunking is deterministic:

```text
parse markdown headings
chunk by heading section first
split oversized sections by size limit
preserve heading path
preserve source path
preserve line_start and line_end when possible
store source content_hash on every chunk
```

Initial chunk fields:

```text
chunk_id
source_id
ordinal
heading_path
content
content_hash
line_start
line_end
sensitivity
status = active | stale | deleted
metadata
```

Chunks should be large enough to preserve meaning but small enough to fit the
ContextAssembler budget.

## Citation model

Every `ContentHit` must include a citation.

Initial citation format:

```text
path:line_start-line_end
```

When heading anchors are useful, the citation may also include:

```text
heading_path
```

The citation must point to the source document, not to generated memory.

## Embeddings and storage

PM-07 may reuse `EmbeddingPort` / `local_embedding`.

The initial storage adapter may use PostgreSQL arrays and deterministic
similarity behind `ContentRetrievalPort`. pgvector remains an adapter
optimization, not a required architectural contract.

Tests must use fake embedding providers. Real model or embedding calls must not
be required for CI.

## Ingestion lifecycle

Ingestion is explicit.

Initial workflow:

```text
scan allowlisted documentation files
compute source content_hash
create or update content_sources
chunk markdown deterministically
generate embeddings through EmbeddingPort
store active chunks and embeddings
mark old chunks stale when source content_hash changes
mark source/chunks deleted or stale when source disappears
```

Retrieval must exclude stale and deleted chunks.

If embedding generation fails, the source or chunk records may be kept with a
failed indexing status, but failed/stale chunks are excluded from retrieval.

## ContextAssembler integration

`ContextAssembler` may call `ContentRetrievalPort` after PM-07.

Content hits must be rendered in a separate section, not mixed into long-term
memory:

```text
Relevant Project Documentation
```

`ContextManifest` must record content hit references:

```text
source_id
chunk_id
citation
score
sensitivity
content_hash
```

`AgentRuntime` and loop strategies must not query content storage directly.

## Policy and sensitivity

Initial policy behavior:

```text
developer_local:
  project docs RAG -> allow

locked_down:
  project docs RAG -> approval_required or deny by configuration

automation:
  project docs RAG -> allow only for configured corpora
```

Secret content is never indexed.

The Content Retrieval subsystem must apply sensitivity filtering before
returning hits to ContextAssembler.

## Events

Planned event taxonomy:

```text
content.source.discovered
content.source.ingested
content.source.updated
content.source.deleted
content.chunk.created
content.chunk.stale
content.embedding.created
content.embedding.failed
content.retrieved
```

Events must not include full raw source content by default.

Current PM-07a/PM-07b implementation wires embedding, retrieval and retrieval
failure event emission through `EventLogPort`, including retrieval failure
events that store query hashes rather than raw query text.

Source/chunk lifecycle events such as `content.source.ingested` and
`content.chunk.created` remain a follow-up hardening slice unless they can be
emitted without raw content and with stable source identifiers.

## Testing requirements

PM-07 is implemented in two slices:

```text
PM-07a Project Docs ingestion and citation index
PM-07b Project Docs retrieval and ContextAssembler integration
```

PM-07a must include unit tests for:

```text
source allowlist matching
secret-like path denial
secret-like content denial
markdown heading chunking
oversized section splitting
citation formatting
stale/deleted status transitions
```

PM-07a must include integration tests for:

```text
source registry creates and updates sources
changed source marks old chunks stale
unchanged reingestion does not churn chunks
content revert/delete-restore reactivates existing chunks
failed source/chunk sync does not publish partial state
deleted source marks chunks deleted or stale
project docs deletion does not delete other content corpora
content tables remain separate from Memory tables
```

PM-07b must include contract and integration tests for:

```text
ContentRetrievalPort returns ContentHit not MemoryHit
retrieval excludes stale/deleted chunks
retrieval returns citations
fake embedding provider is used
embedding failure excludes failed chunks from retrieval
```

PM-07b must include golden context tests for:

```text
ContextAssembler includes content hits in a separate section
ContextManifest records content hit refs
MemoryHit and ContentHit sections remain separate
secret content is excluded
```

Architecture tests must ensure:

```text
Memory subsystem does not import Content Retrieval storage
Content Retrieval subsystem does not write Memory tables
AgentRuntime does not import content storage adapters
ContextAssembler does not import SQLAlchemy models
```

## Consequences

Benefits:

- the assistant can answer from project docs with citations;
- Memory remains stable interpreted knowledge, not a document index;
- RAG can evolve behind `ContentRetrievalPort`;
- pgvector can be introduced later without changing runtime contracts;
- tests can validate citation and stale-content behavior deterministically.

Costs:

- one more subsystem and storage model;
- ingestion and refresh logic must be maintained;
- context budgets become more complex because MemoryHit and ContentHit compete
  for prompt space;
- citation quality depends on deterministic line tracking.

## Non-goals

ADR-034 does not introduce:

```text
general file RAG
source code indexing
PDF ingestion
web ingestion
email or Telegram ingestion
MCP resource indexing
raw conversation vector indexing
event log vector indexing
cloud embeddings
mandatory pgvector dependency
provider-native file search
```
