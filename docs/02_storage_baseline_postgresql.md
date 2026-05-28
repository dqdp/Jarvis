# 02 — Storage Baseline: PostgreSQL

## 1. Decision

PostgreSQL is the primary system of record for Phase 1.

pgvector is used as the initial vector retrieval adapter.

## 2. Why PostgreSQL

PostgreSQL is selected because Phase 1 needs durable, queryable, transactional system-of-record storage for:

- conversations;
- messages;
- append-only events;
- memories;
- memory_candidates;
- memory_embeddings;
- model_invocations;
- policy_decisions;
- runtime checkpoints;
- future tool_invocations;
- future scheduled tasks.

The key requirement is not best-in-class vector search. The key requirement is a reliable local store with:

- transactions;
- relational constraints;
- migrations;
- JSONB;
- indexes;
- backup/restore;
- local self-hosting;
- easy inspection;
- one audit trail.

## 3. Why not separate vector DB in Phase 1

A separate vector DB introduces split-brain risk:

```text
PostgreSQL:
  memory metadata, content, provenance, status

Vector DB:
  embeddings and similarity index
```

Failure cases:

- memory created but embedding failed;
- content updated but vector stale;
- archived memory still searchable;
- backup of metadata and vector index inconsistent;
- transaction boundary unclear.

Phase 1 avoids this by colocating memory records and embeddings in PostgreSQL.

## 4. Avoiding PostgreSQL Lock-in

PostgreSQL is physical baseline, not architectural contract.

Rules:

- Agent Runtime does not import PostgreSQL adapters.
- Runtime depends on ports, not DB models.
- Domain schemas are separate from DB schemas.
- Memory retrieval goes through MemoryReadPort.
- Event writing goes through EventLogPort.
- Conversation access goes through ConversationStorePort.
- Contract tests define expected behavior.

Future replacements:

```text
pgvector -> Qdrant
pgvector -> Weaviate
pgvector -> Milvus
pgvector -> OpenSearch/hybrid retrieval
PostgreSQL memory tables -> external memory service
```

## 5. Data Ownership

```text
ConversationStorePort:
  conversations, messages

EventLogPort:
  events

MemoryWritePort / MemoryReadPort:
  memories, memory_candidates, embeddings

ModelInvocationRepository:
  model_invocations

RuntimeCheckpointAdapter:
  LangGraph checkpoint tables
```

## 6. Retention Principles

Phase 1 default:

- events retained indefinitely;
- messages retained indefinitely;
- model request/response payloads retained unless sensitivity policy changes;
- runtime checkpoints may have retention/cleanup later;
- memories can be active/archived/superseded but not hard-deleted by default.

## 7. Later Re-evaluation Triggers

Revisit PostgreSQL + pgvector if:

- retrieval latency becomes unacceptable;
- memory volume grows significantly;
- multimodal retrieval is required;
- hybrid search becomes central;
- distributed memory service is needed;
- multiple assistant runtimes require shared retrieval service.
