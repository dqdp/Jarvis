# ADR-003: pgvector is initial retrieval adapter, not memory contract

Status: Accepted

## Context

Phase 1 needs vector retrieval but should avoid split-brain between metadata and embeddings.

## Decision

Use pgvector initially. Treat it as adapter implementation, not architectural contract.

## Consequences

Easy MVP and consistent backups. Later replacement by Qdrant/Weaviate/Milvus/hybrid retrieval remains possible.
