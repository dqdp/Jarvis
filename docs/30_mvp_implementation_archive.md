# 30 — MVP Implementation Archive

## Status

Implemented MVP baseline, Slice 00 through Slice 19.

Date: 2026-05-28

## Scope Completed

The implementation covers the Phase 1 Core Daemon MVP:

- repository/test tooling;
- typed configuration and strict startup validation;
- domain schemas and sensitivity model;
- architecture boundary tests;
- EventLogPort, ConversationStorePort, PolicyPort, ModelRouterPort;
- PostgreSQL migrations and storage adapters;
- manual memory write/list/retrieve with embedding status handling;
- deterministic ContextAssembler and ContextManifest;
- `memory_augmented_answer` AgentRuntime workflow;
- FastAPI assistant API;
- SSE runtime stream;
- E2E user-turn lifecycle with fake model and embedding providers;
- final documentation and acceptance checks in Slice 19.

## TDD Evidence

Slice implementation followed the red/green workflow:

- tests were added or updated before production behavior;
- red runs failed on missing behavior or missing documentation state;
- production changes were kept to the active slice;
- contract, architecture, golden, integration and e2e tests were run as each
  layer became available.

Slice 19 red phase added documentation acceptance checks:

- `test_docs_referenced_adrs_exist`;
- `test_config_matches_documented_defaults`;
- `test_all_required_ports_have_contract_tests`;
- `test_mvp_acceptance_checklist_complete`;
- `test_mvp_implementation_archive_exists`.

## Verification Commands

Final acceptance uses:

```text
make test
git diff --check
```

Final local verification result on 2026-05-28:

```text
make test
  unit: 34 passed
  contract: 79 passed
  integration: 3 passed
  golden: 12 passed
  architecture: 16 passed
  e2e: 1 passed
```

Post-review verification result after replacing the API adapter with FastAPI and
fixing the highest-risk review findings:

```text
make test
  unit: 37 passed
  contract: 89 passed
  integration: 3 passed
  golden: 15 passed
  architecture: 16 passed
  e2e: 1 passed
```

Final hardening verification result after the consistency closure:

```text
make test
  unit: 41 passed
  contract: 120 passed
  integration: 8 passed
  golden: 17 passed
  architecture: 22 passed
  e2e: 1 passed
```

The latest fixes add regression coverage for cancellation side effects,
atomic assistant completion, namespace default sensitivity, model policy audit
in API/E2E wiring, policy allow/deny decisions and storage referential
integrity.

Final review closure verification result on 2026-05-29:

```text
make test
  unit: 41 passed
  contract: 120 passed
  integration: 8 passed
  golden: 17 passed
  architecture: 22 passed
  e2e: 1 passed
```

The final closure adds regression coverage for null-safe migration preflights,
public SSE replay DTOs, configured memory retrieval exclusions and score
thresholds, and no-scope-creep architecture packages.

Layer-specific targets:

```text
make test-unit
make test-contract
make test-integration
make test-golden
make test-architecture
make test-e2e
```

## Implementation Adjustments

One implementation-level storage adjustment was accepted for the MVP:

- The memory retrieval adapter stores embeddings as PostgreSQL numeric arrays
  and ranks deterministically behind `MemoryReadPort`; pgvector remains a
  storage adapter follow-up.

This adjustment preserves ports/adapters boundaries and avoids expanding MVP
scope.

## FastAPI Adapter Follow-up

The initial minimal ASGI adapter has been replaced with FastAPI. API, SSE and
E2E tests now exercise the FastAPI app through `httpx.ASGITransport`.

## Post-review Fixes

The multi-agent review found several MVP-quality blockers. The following were
fixed immediately:

- `secret` user-message sensitivity now reaches runtime and blocks model calls;
- FastAPI stream reconnect after completion no longer re-runs the provider;
- API stream passes the conversation `active_project_namespace` to runtime;
- memory retrieval excludes non-indexed and stale-embedding records;
- superseded replacement memories are embedded/indexed before retrieval;
- recent conversation loading returns the latest limited window in chronological
  order;
- model invocation failure audit no longer persists raw provider exception text;
- default API bind is local-only (`127.0.0.1`).

## Hardening Closure

The remaining post-review consistency fixes are now implemented:

- request execution is decoupled from SSE subscription;
- completed/failed/cancelled stream reconnect does not re-run providers;
- running requests can be cancelled explicitly;
- cancellation side effects are blocked after terminal request status is
  observed;
- SSE emits heartbeat events while waiting;
- terminal SSE failure payloads use the public error object shape;
- concurrent `client_message_id` replay returns one durable request;
- context assembly emits `memory.retrieved` and preserves causation through
  `context.assembled` and `model.request.created`;
- context inclusion goes through `PolicyPort`;
- policy allow/deny decisions are audited for model, memory-write and context
  inclusion checks;
- memory writes without an explicit sensitivity use namespace default
  sensitivity;
- storage rejects cross-conversation request/message links;
- memory readiness checks include embedding constraints and retrieval indexes;
- API request bodies are strict Pydantic schemas;
- validation errors omit raw rejected input;
- destructive test DB helpers reject non-local/non-test database URLs;
- runtime context/model timeout failures are mapped to `runtime_timeout`;
- `/v1/health` reports liveness/readiness checks;
- golden tests compare serialized ContextAssembler output to fixtures.

## Final Review Closure

The final review follow-ups are now implemented:

- migration 0006 rejects `assistant_message_id` links where the assistant
  message has no matching `messages.request_id`;
- SSE live and replay output use an explicit public field allowlist;
- memory retrieval applies configured `exclude_sensitivity` and `min_score`;
- architecture tests fail if MVP-forbidden packages such as tools, MCP, RAG,
  ReAct, planner or voice appear under `assistant_core`;
- cancellation documentation states that terminal request state wins races.

## Deferred

The following remain out of MVP:

- tools/MCP;
- RAG/content retrieval;
- ReAct/tool loop;
- planner-executor;
- voice;
- WebSocket/control channel;
- hot config reload;
- cloud fallback.
