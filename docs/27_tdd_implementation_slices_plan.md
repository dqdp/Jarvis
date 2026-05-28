# 27 — TDD Implementation Slices Plan

## Status

Accepted as provisional baseline.

This plan defines the initial TDD implementation order for Phase 1.

It may be revised after coding-agent repository analysis, dependency analysis, prototype spikes or implementation risk discovery.

Architecture-changing revisions require ADR updates.

Implementation-order-only revisions may update this document without ADR.

---

## Goal execution milestones

These milestones are intended for running the full Phase 1 MVP through a
long-lived `/goal`. They do not replace the slice-level TDD workflow. A
milestone is complete only when every included slice satisfies its own
definition of done.

### Milestone 1 — Foundation and guardrails

Scope:

```text
Slice 00 — Repository skeleton and tooling
Slice 01 — Config and Settings
Slice 02 — Domain schemas and enums
Slice 02a — Architecture guardrails baseline
```

Exit criteria:

```text
package imports
pytest markers and task runner work
default/test config validate
canonical domain enums and event names exist
baseline architecture tests are green
```

### Milestone 2 — Durable core contracts

Scope:

```text
Slice 03 — EventLogPort with in-memory adapter
Slice 04 — PostgreSQL foundation and migrations
Slice 05 — PostgresEventLogPort
Slice 06 — ConversationStorePort and assistant_requests
Slice 07 — PolicyPort / ConfigPolicyEngine
```

Exit criteria:

```text
EventLog contract passes for in-memory and Postgres adapters
migrations apply cleanly
conversation/message/request lifecycle is durable
client_message_id idempotency is enforced and event-linked
minimal policy decisions are enforced and audited
```

### Milestone 3 — Runtime, API and MVP acceptance

Scope:

```text
Slice 08 — ModelRouterPort with FakeModelProvider
Slice 09 — Local OpenAI-compatible provider adapter
Slice 10 — MemoryWritePort baseline
Slice 11 — EmbeddingPort and memory embeddings
Slice 12 — MemoryReadPort retrieval
Slice 13 — ContextAssembler golden baseline
Slice 14 — AgentRuntime memory_augmented_answer
Slice 15 — Assistant API request lifecycle
Slice 16 — SSE streaming
Slice 17 — E2E user-turn lifecycle
Slice 18 — Architecture guardrails hardening
Slice 19 — Documentation / acceptance review
```

Exit criteria:

```text
model routing works with fake providers in CI
manual memory write/retrieval works with fake embeddings
ContextAssembler golden tests pass
runtime emits canonical event chain
API and SSE lifecycle pass with fake model provider
main e2e user-turn lifecycle test passes
architecture hardening and MVP acceptance checklist are green
```

---

## Remaining implementation decisions for startup

These are implementation choices fixed for the initial run. They may be
revised only through the normal slice-plan change process if repository or
dependency evidence shows a concrete problem.

```text
Python packaging/test baseline:
  pyproject.toml, pytest, pytest-asyncio, pytest markers, Makefile targets.

Config implementation:
  typed Settings object, YAML defaults/test config, environment overrides,
  strict startup validation.

Database implementation:
  SQLAlchemy async + asyncpg + Alembic, with PostgreSQL/pgvector as the
  integration-test target.

Integration database:
  prefer a docker-compose/test PostgreSQL service; allow DATABASE_URL override
  for environments that provide a ready test database.

Provider tests:
  fake model/embedding providers are mandatory for CI; local OpenAI-compatible
  provider is tested through HTTP mocks, not live inference.

Runtime restart semantics:
  durable messages, events and request status are mandatory; automatic
  continuation of an in-flight streaming request after daemon restart is not
  required in MVP.
```

---

## 1. Purpose

This document turns the Testing Strategy into an initial implementation plan for coding agents.

The plan is intentionally slice-based and TDD-first.

Each slice must be:

```text
small
test-first
independently reviewable
architecture-preserving
bounded by explicit acceptance criteria
```

---

## 2. Slice format

Each implementation slice should be executed using this structure:

```text
Goal:
  What becomes available after the slice.

Inputs:
  Relevant docs/ADR.

Tests first:
  Tests to write before production code.

Implementation:
  Minimal implementation required.

Acceptance criteria:
  What must be green.

Out of scope:
  What must not be added in this slice.

Architecture guardrails:
  Boundaries that must not be violated.
```

---

## 3. Slice 00 — Repository skeleton and tooling

### Goal

Create minimal repository skeleton without business logic.

### Tests first

```text
test_import_assistant_core
test_pytest_runs
test_config_test_environment_loads_minimally
```

### Implementation

```text
pyproject.toml
src/assistant_core/
tests/
config/
Makefile or task runner
pytest markers
```

### Acceptance criteria

```text
make test-unit works
pytest sees markers
package imports successfully
```

### Out of scope

```text
FastAPI
PostgreSQL
LangGraph
ModelRouter
```

---

## 4. Slice 01 — Config and Settings

### Goal

Implement typed config loader and startup validation.

### Tests first

```text
test_default_config_validates
test_test_config_validates
test_cloud_reasoning_disabled_by_default
test_raw_prompt_logging_disabled_by_default
test_required_model_profiles_exist
test_memory_namespace_registry_valid
test_runtime_budget_memory_augmented_answer_limits
test_secret_not_allowed_in_memory_write_policy_config
```

### Implementation

```text
ConfigLoader
Settings schema
config/default.yaml
config/test.yaml
env override mechanism
validation errors
```

### Acceptance criteria

```text
config validation tests green
invalid config fails fast
no secrets in YAML
```

### Out of scope

```text
hot reload
admin config UI
policy DSL
```

---

## 5. Slice 02 — Domain schemas and enums

### Goal

Create canonical domain types before adapters.

### Tests first

```text
test_memory_type_enum_values
test_sensitivity_enum_values
test_request_status_transitions
test_event_envelope_required_fields
test_event_type_enum_includes_canonical_user_turn_chain
test_event_type_enum_includes_error_and_degraded_events
test_chat_message_provider_neutral_shape
```

### Implementation

```text
domain/events.py
domain/messages.py
domain/memory.py
domain/models.py
domain/policy.py
domain/context.py
```

Canonical Phase 1 event names are defined in `domain/events.py`.

User-turn chain:

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

Required error/degraded events:

```text
request.processing.failed
context.assembly.failed
context.assembly.truncated
memory.retrieval.failed
memory.embedding.created
memory.embedding.failed
model.request.failed
model.request.denied
policy.decision.recorded
runtime.error
```

### Acceptance criteria

```text
domain tests green
domain types do not import adapters
```

### Out of scope

```text
database
API
provider clients
```

---

## 6. Slice 02a — Architecture guardrails baseline

### Goal

Introduce architecture tests early so boundary violations are visible before
runtime and adapter implementation expands.

### Tests first

```text
test_domain_does_not_import_adapters
test_domain_does_not_import_api
test_domain_does_not_import_runtime
test_ports_do_not_expose_adapter_types
```

### Implementation

```text
tests/architecture/
import graph or AST helper
baseline module boundary rules
```

### Acceptance criteria

```text
architecture baseline tests green
architecture test target exists
```

### Out of scope

```text
full runtime/provider/storage guardrails
adapter-specific checks before adapters exist
```

Later slices extend these tests as modules are added. Slice 18 remains the
final hardening pass.

---

## 7. Slice 03 — EventLogPort with in-memory adapter

### Goal

Implement EventLog contract against in-memory adapter for fast TDD.

### Tests first

```text
test_event_log_contract_append_assigns_sequence
test_event_log_contract_query_by_request_id_ordered
test_event_log_contract_causation_chain
test_event_envelope_validation
```

### Implementation

```text
EventLogPort
InMemoryEventLog
EventEnvelope validation
```

### Acceptance criteria

```text
contract tests pass against InMemoryEventLog
```

### Out of scope

```text
PostgreSQL
API
```

---

## 8. Slice 04 — PostgreSQL foundation and migrations

### Goal

Set up storage foundation: DB connection, migrations, test DB.

### Tests first

```text
test_database_connects
test_migrations_apply_cleanly
test_migrations_are_idempotent
```

### Implementation

```text
SQLAlchemy/SQLModel or chosen DB layer
Alembic
PostgreSQL test fixture
events table base migration
```

### Acceptance criteria

```text
integration DB tests pass
migrations create schema
```

### Out of scope

```text
business adapters except minimal migration smoke
```

---

## 9. Slice 05 — PostgresEventLogPort

### Goal

Implement EventLogPort on PostgreSQL.

### Tests first

Same contract tests against Postgres adapter:

```text
test_event_log_contract_append_assigns_sequence[postgres]
test_event_log_contract_query_by_request_id_ordered[postgres]
test_event_log_contract_causation_chain[postgres]
test_event_log_contract_preserves_idempotency_key
```

### Implementation

```text
PostgresEventLog
events table
transaction support
```

### Acceptance criteria

```text
same contract tests pass for in-memory and postgres adapters
```

### Architecture guardrails

```text
runtime must not import PostgresEventLog directly
```

---

## 10. Slice 06 — ConversationStorePort and assistant_requests

### Goal

Implement durable conversations/messages/request lifecycle storage.

### Tests first

```text
test_create_conversation
test_append_user_message
test_append_assistant_message
test_load_messages_ordered
test_client_message_id_idempotency_same_content
test_client_message_id_conflict_different_content
test_client_message_id_is_copied_to_event_idempotency_key
test_create_assistant_request
test_request_status_transitions
```

### Implementation

```text
ConversationStorePort
PostgresConversationStore
conversations/messages/assistant_requests tables
idempotency constraint
```

External API and storage use `client_message_id`. EventEnvelope stores the
same value in `idempotency_key`.

### Acceptance criteria

```text
conversation contract tests pass
integration tests green
```

### Out of scope

```text
FastAPI
runtime execution
```

---

## 11. Slice 07 — PolicyPort / ConfigPolicyEngine

### Goal

Implement minimal policy boundary.

### Tests first

```text
test_local_model_project_allowed
test_local_model_secret_denied
test_cloud_model_denied_by_default
test_secret_memory_write_denied
test_secret_context_inclusion_denied
```

### Implementation

```text
PolicyPort
ConfigPolicyEngine
PolicyDecision
```

MVP `PolicyPort` methods:

```text
evaluate_model_request
evaluate_memory_write
evaluate_context_inclusion
```

`evaluate_context_inclusion` is intentionally minimal in Phase 1: deny
`secret`, allow non-secret local context. It is not a policy DSL.

### Acceptance criteria

```text
policy tests green
```

### Out of scope

```text
policy DSL
tool policies
```

---

## 12. Slice 08 — ModelRouterPort with FakeModelProvider

### Goal

Implement ModelRouter without real LLM.

### Tests first

```text
test_model_router_calls_policy_before_provider
test_cloud_reasoning_denied
test_secret_sensitivity_denied
test_chat_creates_model_invocation
test_stream_chat_emits_normalized_tokens
test_chat_retry_zero
test_structured_invalid_json_retries_once
test_embedding_retries_once
test_no_automatic_fallback
```

### Implementation

```text
ModelRouterPort
ModelProfileRegistry
FakeModelProvider
FakeEmbeddingProvider
ModelInvocationRepository
model_invocations table
```

### Acceptance criteria

```text
ModelRouter tests green
no real LLM calls
```

### Out of scope

```text
vLLM/Ollama provider
live model tests
```

---

## 13. Slice 09 — Local OpenAI-compatible provider adapter

### Goal

Add real provider adapter, tested through HTTP mocks.

### Tests first

```text
test_local_openai_provider_builds_chat_request
test_local_openai_provider_parses_chat_response
test_local_openai_provider_stream_normalization
test_local_openai_provider_timeout_maps_to_model_error
```

### Implementation

```text
LocalOpenAICompatibleProviderAdapter
HTTP client
provider-specific conversion
timeout/error mapping
```

### Acceptance criteria

```text
adapter unit tests with mocked HTTP pass
```

### Out of scope

```text
live vLLM required in CI
```

---

## 14. Slice 10 — MemoryWritePort baseline

### Goal

Create memory records without retrieval.

### Tests first

```text
test_create_memory_allowed_namespace_type
test_reject_unknown_namespace
test_reject_invalid_memory_type
test_reject_namespace_type_mismatch
test_reject_secret_memory_write
test_archive_memory
test_supersede_memory
```

### Implementation

```text
MemoryWritePort
PostgresMemoryStore write side
memories table
memory lifecycle events
```

### Acceptance criteria

```text
memory write contract tests pass
```

### Out of scope

```text
vector retrieval
RAG
```

---

## 15. Slice 11 — EmbeddingPort and memory embeddings

### Goal

Add embeddings for memories.

### Tests first

```text
test_embedding_port_delegates_to_model_router
test_create_memory_generates_embedding
test_embedding_failure_keeps_memory_with_embedding_failed
test_update_memory_content_recomputes_embedding
test_stale_embedding_excluded_by_content_hash
```

### Implementation

```text
EmbeddingPort
memory_embeddings table
content_hash
indexing_status
sync embedding on create/update
```

### Acceptance criteria

```text
embedding tests green with fake embedding provider
```

### Out of scope

```text
bulk reindex
background jobs
```

---

## 16. Slice 12 — MemoryReadPort retrieval

### Goal

Implement namespace-aware active memory retrieval.

### Tests first

```text
test_retrieve_active_memories_only
test_exclude_archived
test_exclude_superseded
test_filter_by_namespace
test_exclude_secret
test_respects_max_hits_total
test_respects_max_hits_per_namespace
test_ranking_score_importance_recency
test_retrieval_failure_can_be_reported
```

### Implementation

```text
MemoryReadPort
pgvector query
MemoryHit
retrieval config
```

### Acceptance criteria

```text
MemoryReadPort contract tests pass
```

### Out of scope

```text
reranker
hybrid search
document chunks
```

---

## 17. Slice 13 — ContextAssembler golden baseline

### Goal

Implement deterministic ContextAssembler.

### Tests first

```text
test_context_golden_fixed_section_order
test_includes_current_user_message
test_includes_recent_conversation_window
test_applies_max_messages
test_applies_token_budget
test_drops_oldest_messages_first
test_retrieves_active_memories
test_excludes_secret_memories_and_messages
test_context_manifest_contains_used_refs
test_context_manifest_is_event_recorded_without_raw_prompt
test_no_full_prompt_logged_by_default
test_degraded_context_when_memory_retrieval_fails
```

### Implementation

```text
ContextAssemblerPort
DeterministicContextAssembler
NamespaceSelector
ConversationWindow policy
ContextManifest
```

Phase 1 records `ContextManifest` in the `context.assembled` event payload.
It also produces a stable `context_manifest_id` used by model invocation audit.
No separate `context_manifests` table is required in MVP.

### Acceptance criteria

```text
golden tests green
```

### Out of scope

```text
reranker
compression
rolling summary
LLM query rewriting
```

---

## 18. Slice 14 — AgentRuntime memory_augmented_answer

### Goal

Connect ContextAssembler, ModelRouter and EventLog in deterministic workflow.

### Tests first

```text
test_runtime_persists_event_chain_success
test_runtime_uses_context_assembler
test_runtime_calls_model_router_once
test_runtime_max_model_calls_one
test_runtime_memory_retrieval_failure_degraded
test_runtime_model_failure_marks_request_failed
test_runtime_uses_canonical_event_type_enum
test_runtime_context_manifest_id_links_model_invocation_to_context_event
test_no_assistant_message_on_system_failure
```

### Implementation

```text
AgentRuntime
memory_augmented_answer workflow
LangGraph integration or minimal graph wrapper
runtime events
request status updates
```

LangGraph checkpoints are PostgreSQL-backed runtime state, but Phase 1 does
not require automatic continuation of an in-flight streaming request after
daemon restart. Durable messages, events and request status remain mandatory.

### Acceptance criteria

```text
runtime tests green
max_model_calls=1 enforced
```

### Out of scope

```text
ReAct
tools
scheduler
```

---

## 19. Slice 15 — Assistant API request lifecycle

### Goal

Add FastAPI endpoints without full SSE or with minimal stream placeholder.

### Tests first

```text
test_post_conversation
test_post_message_returns_request_id
test_get_request_status
test_get_conversation_messages
test_idempotent_message_submit
test_conflicting_client_message_id_returns_409
test_standard_error_format
```

### Implementation

```text
FastAPI app
conversation endpoints
message submit
request status
memory create/list minimal
health
```

### Acceptance criteria

```text
API tests green
```

### Out of scope

```text
auth
WebSocket
cancellation
```

---

## 20. Slice 16 — SSE streaming

### Goal

Add `/requests/{request_id}/stream`.

### Tests first

```text
test_sse_stream_emits_request_started
test_sse_stream_emits_token_events
test_sse_stream_emits_assistant_message_created
test_sse_stream_emits_request_completed
test_token_events_not_persisted
test_failed_request_emits_failure_event
```

### Implementation

```text
RuntimeStreamEvent
SSE endpoint
in-process stream broadcaster
fake model stream integration
```

### Acceptance criteria

```text
SSE tests green
```

### Out of scope

```text
token replay guarantee
WebSocket
cancellation
```

---

## 21. Slice 17 — E2E user-turn lifecycle

### Goal

Implement main MVP smoke test.

### Tests first

```text
test_e2e_user_turn_lifecycle_with_memory_and_fake_model
```

Expected:

```text
POST message returns request_id
stream completes
assistant message persisted
model_invocation created
events correlated by request_id
ContextManifest references selected memory
request status completed
```

### Implementation

Minimal integration fixes only.

### Acceptance criteria

```text
main E2E test green
```

---

## 22. Slice 18 — Architecture guardrails hardening

### Goal

Complete boundary tests after all MVP modules exist.

### Tests first

```text
test_runtime_does_not_import_postgres_adapters
test_runtime_does_not_import_provider_clients
test_context_assembler_does_not_import_provider_clients
test_only_storage_adapters_import_orm_models
test_no_pgvector_import_outside_storage_or_memory_adapter
test_model_router_uses_policy_port
test_no_raw_prompt_logging_by_default
```

### Implementation

```text
import graph checks
AST checks
grep checks where appropriate
CI target
```

### Acceptance criteria

```text
architecture tests green
```

Baseline guardrails are introduced in Slice 02a. This slice extends them to
runtime, storage, provider adapters, pgvector usage and raw prompt logging.

---

## 23. Slice 19 — Documentation / acceptance review

### Goal

Synchronize docs, config, tests and implementation.

### Tests/checks

```text
test_docs_referenced_adrs_exist
test_config_matches_documented_defaults
test_all_required_ports_have_contract_tests
```

### Acceptance criteria

```text
README current
ADR index current
MVP acceptance checklist complete
archive generated
```

---

## 24. How coding agents may propose changes

A coding agent may propose slice-plan changes after repository/dependency analysis.

Required proposal format:

```text
Proposed slice plan change
Reason
Affected docs/ADR
Risk
What tests change
Whether architecture decision changes
```

Rules:

```text
Implementation-order-only changes update this document.
Architecture-changing changes require ADR update.
```

---

## 25. Non-negotiable constraints

Coding agents must not:

```text
change accepted architecture through slice plan
bypass ports/adapters
add tools/RAG/ReAct to MVP
enable cloud fallback
remove TDD-first workflow
remove contract tests for replaceable ports
remove architecture tests
```
