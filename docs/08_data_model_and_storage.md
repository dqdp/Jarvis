# 08 — Data Model and Storage

## 1. Purpose

This document defines the Phase 1 data model baseline.

PostgreSQL is the primary system of record. Memory retrieval for explicit
long-term memory records is accessed through `MemoryReadPort`; pgvector is the
preferred similarity-index adapter when available, but not the data-model
contract.

Phase 1 uses domain tables plus an append-only event log. It does not implement full event sourcing.

---

## 2. Phase 1 tables

Baseline tables:

```text
conversations
messages
assistant_requests
events
memories
memory_candidates
memory_embeddings
model_invocations
policy_decisions
deferred graph checkpoint tables
```

---

## 3. conversations

Stores durable conversation metadata.

Recommended fields:

```text
conversation_id uuid primary key
user_id text not null
title text null
active_project_namespace text null
status text not null default 'active'
created_at timestamptz not null
updated_at timestamptz not null
metadata jsonb not null default '{}'
```

Allowed MVP statuses:

```text
active
archived
```

`active_project_namespace` is used by ContextAssembler to select project-specific memory namespaces.

---

## 4. messages

Stores operational conversation messages.

Recommended fields:

```text
message_id uuid primary key
conversation_id uuid not null references conversations(conversation_id)
request_id uuid null
event_id uuid null references events(event_id)
client_message_id text null
role text not null
content text not null
content_hash text not null
sensitivity text not null default 'personal'
created_at timestamptz not null
metadata jsonb not null default '{}'
```

Phase 1 persisted roles:

```text
user
assistant
```

Reserved future roles:

```text
system
tool
developer
```

Rules:

- messages are append-only in MVP;
- system/runtime prompt messages are not stored as ordinary conversation messages;
- `client_message_id` supports idempotency and should be unique per conversation when present;
- message content is text-only in MVP; typed content parts are post-MVP.

Recommended indexes:

```sql
create index messages_conversation_created_idx
on messages(conversation_id, created_at);

create index messages_request_idx
on messages(request_id);
```

---

## 5. assistant_requests

Operational lifecycle table for one user turn.

Recommended fields:

```text
request_id uuid primary key
conversation_id uuid not null
user_message_id uuid not null
assistant_message_id uuid null
status text not null
client_message_id text null
created_at timestamptz not null
started_at timestamptz null
completed_at timestamptz null
error_code text null
error_message text null
metadata jsonb not null default '{}'
```

Allowed MVP statuses:

```text
accepted
running
waiting_approval
completed
failed
```

Allowed MVP terminal status:

```text
cancelled
```

`request_id` links messages, events, model invocations and stream lifecycle.

---

## 6. events

Append-only historical log using stable `EventEnvelope`.

Recommended fields:

```text
event_id uuid primary key
event_seq bigserial not null
event_type text not null
event_version int not null
occurred_at timestamptz not null
recorded_at timestamptz not null
conversation_id uuid null
request_id uuid null
correlation_id uuid null
causation_id uuid null references events(event_id)
parent_event_id uuid null
actor_type text not null
actor_id text null
source_component text not null
source_node text null
sensitivity text not null
visibility text not null
idempotency_key text null
payload jsonb not null default '{}'
metadata jsonb not null default '{}'
```

Rules:

- `event_seq` provides global ordering;
- `request_id` links one user turn;
- `correlation_id` is reserved for long-running workflows;
- `causation_id` links direct causal chain;
- raw full prompts are not stored by default;
- raw message content is referenced by `message_id` / `content_hash` rather than duplicated by default;
- token-by-token streaming events are not persisted.

Canonical user turn flow:

```text
user.message.created
request.processing.started
context.assembly.started
memory.retrieved
context.assembled
model.request.created
model.response.received
assistant.message.created
request.processing.completed
```

Recommended indexes:

```sql
create index events_conversation_seq_idx on events(conversation_id, event_seq);
create index events_request_seq_idx on events(request_id, event_seq);
create index events_correlation_seq_idx on events(correlation_id, event_seq);
create index events_type_idx on events(event_type);
create index events_causation_idx on events(causation_id);
```

---

## 7. memories

Interpreted durable knowledge.

Recommended fields:

```text
memory_id uuid primary key
namespace text not null
memory_type text not null
content text not null
summary text null
content_hash text not null
sensitivity text not null
confidence real not null
importance real not null
status text not null default 'active'
indexing_status text not null default 'embedding_pending'
source_event_ids uuid[] not null default '{}'
supersedes_memory_ids uuid[] not null default '{}'
superseded_by_memory_id uuid null
revision int not null default 1
valid_from timestamptz null
valid_until timestamptz null
archived_at timestamptz null
archive_reason text null
created_at timestamptz not null
updated_at timestamptz not null
metadata jsonb not null default '{}'
```

Allowed Phase 1 namespaces:

```text
user.preferences
user.working_style
project.personal_assistant
system.runtime_rules
environment.inference_node
```

Allowed Phase 1 memory types:

```text
fact
preference
procedure
summary
```

Allowed memory statuses:

```text
active
archived
superseded
```

Allowed indexing statuses:

```text
indexed
embedding_pending
embedding_failed
```

Rules:

- `secret` cannot be stored as long-term memory;
- retrieval uses active memories only by default;
- archived/superseded memories are excluded from normal retrieval;
- stale embeddings are excluded through content hash mismatch.

Recommended indexes:

```sql
create index memories_namespace_idx on memories(namespace);
create index memories_type_idx on memories(memory_type);
create index memories_status_idx on memories(status);
create index memories_namespace_type_status_idx
  on memories(namespace, memory_type, status);
create index memories_retrieval_filter_idx
  on memories(namespace, memory_type, status, sensitivity, indexing_status);
```

---

## 8. memory_candidates

Future-safe intermediate records for proposed memories.

Recommended fields:

```text
candidate_id uuid primary key
proposed_namespace text not null
proposed_memory_type text not null
content text not null
sensitivity text not null
confidence real null
source_event_ids uuid[] not null default '{}'
status text not null
created_by text not null
created_at timestamptz not null
resolved_at timestamptz null
resolution_reason text null
metadata jsonb not null default '{}'
```

Allowed candidate statuses:

```text
pending
approved
rejected
merged
expired
```

Phase 1 includes table/domain model but does not require automatic extraction.

---

## 9. memory_embeddings

Embeddings for memories.

Recommended fields:

```text
memory_id uuid not null references memories(memory_id)
embedding_profile text not null
embedding_model text not null
embedding_dimension int not null
content_hash text not null
embedding vector(...) or double precision[]
created_at timestamptz not null
metadata jsonb not null default '{}'
primary key(memory_id, embedding_profile)
```

Rules:

- embeddings are generated through `EmbeddingPort`;
- default implementation delegates to `ModelRouter.embed(local_embedding)`;
- retrieval excludes embeddings whose `content_hash` does not match current memory content hash.

Optional pgvector index example:

```sql
create index memory_embeddings_hnsw_idx
on memory_embeddings
using hnsw (embedding vector_cosine_ops);
```

---

## 10. model_invocations

Audit of every model call, including embeddings.

Recommended fields:

```text
model_invocation_id uuid primary key
request_id uuid null
conversation_id uuid null
profile text not null
provider text not null
model text not null
purpose text not null
sensitivity text not null
status text not null
started_at timestamptz not null
finished_at timestamptz null
latency_ms int null
input_token_estimate int null
input_tokens_reported int null
output_tokens_reported int null
streaming boolean not null default false
error_type text null
error_message text null
context_manifest_id uuid null
metadata jsonb not null default '{}'
```

Rules:

- embeddings are model invocations with `purpose=embedding`;
- full raw prompt is not stored by default;
- chat calls should reference `context_manifest_id` where available.

---

## 11. policy_decisions

Audit of policy checks.

Recommended fields:

```text
policy_decision_id uuid primary key
request_id uuid null
conversation_id uuid null
decision_type text not null
decision text not null
reason text not null
policy_version text null
created_at timestamptz not null
metadata jsonb not null default '{}'
```

---

## 12. ContextManifest persistence

Assembled prompt context is not a primary domain record.

Phase 1 records the ContextManifest in the `context.assembled` event payload.
The manifest has a stable `context_manifest_id` that is copied into
`model_invocations.context_manifest_id` when a model call uses the
assembled context. A separate `context_manifests` table is not required in MVP.

The ContextManifest contains:

```text
used_message_ids
used_memory_ids
token_estimate
context_sections
dropped_refs
model_profile
max_sensitivity
sources_by_sensitivity
redaction_status
degraded flag
```

Full prompt body should not be persisted by default.

---

## 13. Runtime checkpoints

MVP does not require graph checkpoint tables. If LangGraph or another graph
runtime is introduced later, checkpoint tables may be stored in PostgreSQL but
remain logically runtime execution state.

Rules:

- not event log;
- not memory;
- not public API;
- can have cleanup policy later.

---

## 14. Transaction rules

For accepted user message submission, one transaction should create:

```text
assistant_request row
user message row
user.message.created event
request.processing.started event
```

If this transaction fails, the request is not accepted.

Domain mutation plus corresponding event append should share a transaction where practical.
