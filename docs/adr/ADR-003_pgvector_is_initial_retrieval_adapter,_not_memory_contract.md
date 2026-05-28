# ADR-003: PostgreSQL-colocated embeddings and replaceable retrieval adapter

Status: Accepted

## Context

Phase 1 needs vector retrieval but should avoid split-brain between metadata and embeddings.

## Decision

Use PostgreSQL-colocated embeddings initially. Treat pgvector as the preferred
similarity-index adapter when available, not as an architectural contract.

For the implemented MVP, the storage adapter may use a portable PostgreSQL
numeric array representation and deterministic ranking while the `MemoryReadPort`
contract is being established. Replacing that adapter with pgvector must not
change AgentRuntime, ContextAssembler or domain contracts.

## Consequences

Easy MVP and consistent backups. Later replacement by pgvector,
Qdrant/Weaviate/Milvus/hybrid retrieval remains possible.
