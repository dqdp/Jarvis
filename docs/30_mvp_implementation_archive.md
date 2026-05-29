# 30 — MVP Implementation Archive

## Status

Implemented MVP baseline, Slice 00 through Slice 19.

Date: 2026-05-29

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

Dogfood runtime verification result on 2026-05-29:

```text
make test
  unit: 42 passed
  contract: 122 passed
  integration: 8 passed
  golden: 17 passed
  architecture: 22 passed
  e2e: 1 passed
```

This verification adds a production composition root, profile-specific provider
selection, OpenAI-compatible embedding calls and Makefile runtime targets.

Local Ollama and CLI verification result on 2026-05-29:

```text
make test
  unit: 70 passed
  contract: 122 passed
  integration: 8 passed
  golden: 19 passed
  architecture: 25 passed
  e2e: 1 passed
```

This verification adds native Ollama provider support, the `ollama` runtime
profile, local model Makefile targets and a thin CLI for health, manual memory
and chat smoke checks. The current local chat model is `qwen3.5:4b`; the local
embedding model is `embeddinggemma:latest`.

Interactive CLI shell verification was added after the first CLI pass. Running
`make cli` without `ARGS`, or `make cli ARGS='chat'`, opens a terminal chat
session with `/help`, `/new`, `/memory add`, `/memory list` and `/exit`.
The CLI also cancels the server request on stream interruption, and the local
Ollama dogfood profile now uses `qwen3.5:4b` with `max_output_tokens` capped at
1024 to bound runaway generations.
The interactive shell uses Unix `readline` on TTY for Up/Down in-session input
history without persisting raw prompts to disk. `/memory add` payloads and
`secret` sensitivity sessions are excluded from readline history.

The final consistency pass also verifies that `DeterministicContextAssembler`
includes assembled context sections in provider-neutral model messages,
provider adapters surface timeouts and embedding cardinality mismatches
explicitly, native Ollama stream error chunks fail the request instead of
completing partial output, CLI daemon errors are reported as `error>`, and
migration entrypoints require a local database host unless explicitly
overridden. The CLI history regression checks also cover TTY auto-history
removal before filtering `/memory add` payloads and `secret` sensitivity
sessions.

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

## Dogfood Runtime Verification

The first operational verification pass closed the main gap between test wiring
and a runnable daemon:

- `assistant_core.app_factory:create_asgi_app` now assembles the FastAPI app
  from config, PostgreSQL adapters, policy, context assembler, model router and
  local providers;
- `ModelRouter` supports profile-specific providers before falling back to
  provider-name lookup, allowing `local_main`, `local_structured` and
  `local_embedding` to use separate profile adapters;
- the local OpenAI-compatible adapter supports `/embeddings`;
- `make migrate` and `make run` provide standard local runtime commands;
- `make cli`, `make models-list` and `make models-pull` provide local operation
  commands;
- a contract test exercises the app factory with fake providers through
  health, memory write, message submission and SSE completion.

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
