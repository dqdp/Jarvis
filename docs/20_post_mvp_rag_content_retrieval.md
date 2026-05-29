# 20 — Post-MVP RAG and Content Retrieval

## 1. Decision

Full RAG is not part of Phase 1 Core Daemon.

Phase 1 implements only explicit long-term memory retrieval.

Full RAG will be introduced post-MVP as a separate Content Retrieval subsystem, not as an extension of Memory records.

---

## 2. Memory retrieval vs RAG

Memory retrieval answers:

```text
What has the assistant explicitly stored as stable knowledge?
```

RAG / Content Retrieval answers:

```text
What relevant source material can be found in external/project/document content?
```

These are different domains and must not be collapsed.

---

## 3. Why RAG is deferred

Full RAG would require:

```text
source registry
document ingestion
chunking
deduplication
indexing jobs
re-indexing
citations
chunk ranking
permissions
source refresh policy
connectors
```

This would expand Phase 1 beyond the Core Daemon MVP.

---

## 4. Future architecture

Post-MVP, ContextAssembler may use:

```text
ConversationStorePort
MemoryReadPort
ContentRetrievalPort
PolicyPort
```

Future shape:

```text
ContextAssembler
  -> MemoryReadPort        # stable interpreted memory
  -> ContentRetrievalPort  # retrieved source chunks/documents
```

`ContentRetrievalPort` is a separate future facade.

---

## 5. Future ContentRetrievalPort

Potential contract:

```python
class ContentRetrievalPort(Protocol):
    async def retrieve(self, query: ContentQuery) -> list[ContentHit]: ...
```

`ContentHit` is not `MemoryHit`.

Potential fields:

```text
source_id
chunk_id
source_type
title
content
score
citation
sensitivity
metadata
```

---

## 6. First post-MVP target

The first RAG target should be narrow:

```text
Project Documentation RAG
```

Initial corpus:

```text
docs/*.md
ADR documents
README
architecture docs
```

Reason:

- high value for this project;
- markdown is easy to ingest;
- permissions are simple;
- citations are straightforward;
- helps validate architecture before indexing arbitrary files.

---

## 7. Later RAG targets

Later targets:

```text
files
PDFs
codebase
web pages
email
Telegram history
logs
MCP resources
external manuals
```

Each target may require its own source adapter and policy rules.

---

## 8. Shared embedding capability

RAG may reuse `EmbeddingPort` / `local_embedding` capability.

However, storage and domain model must remain separate:

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

---

## 9. Event log relation

RAG ingestion/retrieval should produce events:

```text
content.source.ingested
content.source.updated
content.chunk.created
content.index.created
content.retrieved
```

But source documents remain the source of truth for document content.

Chunks are derived artifacts.

---

## 10. Boundary rules

Rules:

```text
Memory subsystem must not store document chunks.
Event log must not become default vector corpus.
Conversation history must not be vector-indexed in MVP.
RAG must have source registry, chunk model and citation model.
ContextAssembler combines MemoryHit and ContentHit through explicit sections.
```

---

## 11. Post-MVP follow-up

The accepted Content Retrieval ADR is:

```text
docs/adr/ADR-034_content_retrieval_subsystem_and_project_docs_rag.md
```

It defines:

```text
ContentRetrievalPort
source registry
chunk model
citation model
ingestion lifecycle
permission/sensitivity handling
retrieval ranking
context assembly integration
```
