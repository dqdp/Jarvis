# 26 — Testing Strategy

## 1. Purpose

This document defines the Phase 1 testing strategy.

The project will be implemented by coding agents and must be developed TDD-first.

Tests are not only a correctness check. They are executable architecture policy and guardrails for agent-driven implementation.

---

## 2. Core principle

Phase 1 development is TDD-first.

For every implementation slice:

```text
write or update tests first
observe expected failures
implement minimal production code
keep scope within the slice
run architecture and contract tests
```

If a test is hard to write, the design boundary should be clarified before implementation.

---

## 3. Why tests are architectural guardrails

Coding agents may otherwise take shortcuts such as:

```text
AgentRuntime directly importing PostgreSQL adapters
AgentRuntime directly calling vLLM/Ollama/OpenAI clients
ContextAssembler directly querying pgvector
Memory subsystem storing document chunks
ModelRouter bypassing PolicyPort
raw prompts written to logs
event log skipped for convenience
```

Architecture tests and contract tests must make these shortcuts visible.

---

## 4. Required test layers

Phase 1 uses these test layers:

```text
unit
contract
integration
golden
architecture
e2e
```

Recommended layout:

```text
tests/
  unit/
  contract/
  integration/
  golden/
  architecture/
  e2e/
```

---

## 5. Unit tests

Unit tests cover pure logic without PostgreSQL or real LLM calls.

Examples:

```text
Config validation
Policy decisions
Context budget trimming
Namespace selection
Memory lifecycle transitions
Request status transitions
Error mapping
```

Unit tests must be fast and deterministic.

---

## 6. Contract tests

Contract tests are mandatory for replaceable ports.

Required ports:

```text
ConversationStorePort
EventLogPort
MemoryReadPort
MemoryWritePort
EmbeddingPort
ContextAssemblerPort
ModelRouterPort
PolicyPort
```

Example MemoryReadPort contract:

```text
returns only active memories
filters by namespace
excludes secret
excludes stale embeddings
respects max_hits_total
returns score and metadata
```

If an adapter is replaced, the new adapter must pass the same contract tests.

---

## 7. Integration tests

Integration tests verify real adapters with PostgreSQL and local infrastructure.
If a pgvector adapter is enabled, it must be covered by the same integration
and MemoryReadPort contract tests.

Examples:

```text
PostgresEventLogPort writes ordered events
PostgresConversationStore persists messages
PostgresMemoryStore creates memory and embedding rows
assistant_requests transaction creates request/message/events
```

Integration tests may use testcontainers or a docker-compose test database.

---

## 8. Golden tests

Golden tests are mandatory for `ContextAssembler`.

They verify assembled context and ContextManifest, not LLM answer quality.

Examples:

```text
fixed section order
current user message included
recent conversation window included
active memories included
archived/superseded memories excluded
secret memories/messages excluded
oldest messages dropped when over budget
ContextManifest includes used/dropped refs
degraded=true when memory retrieval fails
```

---

## 9. Architecture tests

Architecture tests are mandatory.

They enforce module boundaries and ports/adapters rules.

Examples:

```text
AgentRuntime must not import PostgreSQL adapters.
AgentRuntime must not import provider clients.
AgentRuntime must not import SQLAlchemy/pgvector.
AgentRuntime must depend on ports/domain schemas only.
ContextAssembler must not call provider-specific model clients.
ContextAssembler must not import SQLAlchemy models.
Memory domain must not import API layer or AgentRuntime.
ModelRouter must depend on PolicyPort.
Only storage adapters may import ORM models.
No raw prompt logging unless explicitly debug-enabled.
```

Architecture tests may be import-graph checks, AST checks, or simple static grep checks where appropriate.

---

## 10. E2E tests

At least one E2E user-turn lifecycle test is mandatory.

It must use fake model providers, not real LLM calls.

Required E2E scenario:

```text
POST message
  -> request_id returned
  -> SSE/runtime events emitted
  -> assistant message persisted
  -> event chain correlated by request_id
  -> model_invocation created
  -> ContextManifest references selected context
```

Canonical event chain:

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

---

## 11. Fake providers

Fake providers are mandatory for deterministic TDD.

Required fakes:

```text
FakeModelProvider
FakeEmbeddingProvider
FakeMemoryRetriever failure mode
FakePolicyEngine
```

FakeModelProvider should support:

```text
fixed response
stream fixed tokens
raise timeout
raise provider unavailable
return invalid JSON once then valid JSON
```

Real LLM calls are not required for CI.

---

## 12. Required MVP test groups

### Config validation

```text
default config validates
test config validates
JARVIS_ env overrides apply through double-underscore nested keys
cloud_reasoning disabled by default
raw_prompt_logging=false by default
local_main/local_structured/local_embedding exist
memory types are fact/preference/procedure/summary
accepted namespaces exist
memory_augmented_answer max_model_calls=1
memory_augmented_answer max_tool_calls=0
```

### Policy

```text
local model + project sensitivity -> allow
local model + secret sensitivity -> deny
cloud model -> deny by default
secret memory write -> deny
secret context inclusion -> deny
```

### EventLog

```text
append event assigns event_seq
query by request_id returns ordered events
causation_id links events
envelope required fields validated
sanitized payload does not contain raw secret marker
```

### ConversationStore

```text
create conversation
append user message
append assistant message
load ordered messages
client_message_id idempotency returns existing request
same client_message_id different content -> conflict
messages are append-only
```

### Request lifecycle

```text
POST message creates assistant_request
successful runtime sets completed
failed model sets failed
no assistant message on system failure
request_id links user and assistant messages
```

### ContextAssembler

```text
fixed section order
recent window
token budget
active memories only
secret exclusion
ContextManifest not full prompt
AssembledContext exposes explicit ContextManifest
degraded mode on memory retrieval failure
```

### Memory

```text
create allowed memory
MemoryRecord has sensitivity/content_hash/indexing_status
memory_candidates schema/domain exists without auto-extraction
reject unknown namespace
reject invalid memory type
reject namespace/type mismatch
reject secret memory write
archive excludes from retrieval
superseded excludes from retrieval
embedding failure keeps memory with indexing_status=embedding_failed
stale embedding excluded by content_hash mismatch
```

### ModelRouter

```text
PolicyPort invoked before call
cloud_reasoning denied
secret sensitivity denied
model_invocation created
streaming emits normalized events
chat retry = 0
structured invalid JSON retries once
embedding retries once
no automatic fallback
```

### API

```text
POST /conversations creates conversation
POST /messages returns request_id
GET /requests/{id} returns status
GET /requests/{id}/stream emits runtime events
POST /memories creates manual memory
GET /memories lists memories
GET /health returns healthy status
same client_message_id returns existing request
same client_message_id different content -> 409
standard error format
```

---

## 13. CI expectations

Recommended commands:

```text
make test
make test-unit
make test-contract
make test-integration
make test-golden
make test-architecture
make test-e2e
```

Pytest markers:

```text
unit
contract
integration
golden
architecture
e2e
```

---

## 14. Out of MVP tests

Do not require in MVP:

```text
real LLM quality tests
benchmark tests
load tests
multi-user auth tests
WebSocket tests
tool execution tests
MCP tests
RAG/document ingestion tests
rolling summary tests
planner/ReAct tests
cloud provider live tests
```

---

## 15. TDD implementation slices

The testing strategy must be mapped to implementation slices.

However, the detailed high-level TDD implementation plan by slices is intentionally separated into a follow-up document and discussion:

```text
27_tdd_implementation_slices_plan.md
```

This prevents mixing test policy with delivery planning.

---

## 16. MVP acceptance for testing

MVP is not complete unless:

```text
unit tests pass
contract tests pass
integration tests pass
golden tests pass
architecture tests pass
main e2e user-turn lifecycle test passes
config validation tests pass
```


## Slice plan status

The detailed TDD implementation slice plan is accepted as provisional baseline:

```text
27_tdd_implementation_slices_plan.md
```

It may be revised after coding-agent repository/dependency analysis.

Architecture-changing changes require ADR updates.
