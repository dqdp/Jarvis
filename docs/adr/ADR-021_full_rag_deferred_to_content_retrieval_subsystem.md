# ADR-021 — Full RAG Deferred to Content Retrieval Subsystem

## Status

Accepted.

## Context

The assistant will likely need RAG in the future for project documentation, files, PDFs, codebase, web pages, email, logs and MCP resources.

However, Phase 1 is focused on Core Daemon architecture: event log, memory, context assembly, model routing, local inference and basic API.

Full RAG would introduce ingestion, chunking, citations, source registry, permissions and indexing workflows.

## Decision

Full RAG is out of scope for Phase 1.

Phase 1 implements only explicit long-term memory retrieval.

Post-MVP RAG will be introduced as a separate Content Retrieval subsystem accessed through a future `ContentRetrievalPort`.

Memory subsystem must not store document chunks.

First post-MVP RAG target should be:

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

## Rationale

Memory and RAG represent different domains:

```text
Memory:
  interpreted stable knowledge

RAG / Content Retrieval:
  retrieved source material with citations
```

Keeping them separate prevents memory subsystem from becoming a generic search platform.

## Consequences

Positive:

- protects MVP scope;
- preserves clean memory model;
- creates clear future extension point;
- allows project docs RAG to be introduced incrementally.

Trade-offs:

- Phase 1 assistant cannot search arbitrary documents;
- project documentation lookup remains manual or memory-based until post-MVP;
- future RAG requires separate design/ADR.

## Deferred

- ContentRetrievalPort;
- source registry;
- content_sources table;
- content_chunks table;
- content_embeddings table;
- citation model;
- ingestion lifecycle;
- document permissions;
- codebase indexing;
- web/email/log connectors.
