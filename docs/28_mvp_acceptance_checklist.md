# 28 — MVP Acceptance Checklist

## 1. Purpose

This checklist defines what must be true before Phase 1 Core Daemon can be considered MVP-complete.

It is intentionally implementation-facing and should be used by coding agents during final acceptance review.

Status: completed for the local TDD MVP baseline on 2026-05-29. The API adapter
uses FastAPI. Switching the current PostgreSQL array retrieval adapter to
pgvector remains tracked as an adapter-level follow-up, not an MVP blocker.

---

## 2. Documentation baseline

- [x] README is current and lists all docs/ADR.
- [x] All accepted ADRs exist.
- [x] No accepted ADR is missing from README index.
- [x] All MVP/post-MVP boundaries are explicit.
- [x] TDD implementation slice plan is current or explicitly revised after agent analysis.

---

## 3. Architecture boundaries

- [x] AgentRuntime depends on ports/domain schemas, not adapters.
- [x] AgentRuntime does not import PostgreSQL adapters, SQLAlchemy/pgvector or provider clients.
- [x] AgentRuntime calls `ContextAssemblerPort`, not prompt-building helpers directly.
- [x] ContextAssembler does not call provider-specific model clients.
- [x] Memory subsystem does not store document chunks.
- [x] ModelRouter consults PolicyPort before model calls.
- [x] Cloud fallback is disabled.

---

## 4. Core runtime

- [x] Conversation can be created.
- [x] User message submission returns `request_id`.
- [x] `assistant_requests` lifecycle works.
- [x] Deterministic `memory_augmented_answer` workflow runs.
- [x] `max_model_calls=1` and `max_tool_calls=0` are enforced.
- [x] Assistant message is persisted on success.
- [x] No assistant message is created on system failure by default.

---

## 5. Event log

- [x] EventEnvelope required fields are enforced.
- [x] Events are ordered by `event_seq`.
- [x] Events can be queried by `request_id`.
- [x] Canonical user-turn event chain is produced.
- [x] Causation links are recorded where applicable.
- [x] Token-by-token events are not persisted.
- [x] Raw full prompt is not persisted by default.

---

## 6. Memory and retrieval

- [x] Memory namespaces are registered and validated.
- [x] Memory types are limited to `fact`, `preference`, `procedure`, `summary`.
- [x] `memory_candidates` schema/domain exists without automatic extraction.
- [x] `MemoryRecord` domain contract includes `sensitivity`, `content_hash` and `indexing_status`.
- [x] Secret memory writes are denied.
- [x] Active memory retrieval works.
- [x] Archived/superseded memories are excluded.
- [x] Retrieval is namespace-aware.
- [x] Manual memory records can be created and listed through MVP API.
- [x] Embedding failure creates memory with `indexing_status=embedding_failed`.
- [x] Stale embeddings are excluded by content hash mismatch.

---

## 7. Context assembly

- [x] ContextAssembler uses fixed MVP section order.
- [x] Current user message is included.
- [x] Recent conversation window is included within budget.
- [x] Active retrieved memories are included.
- [x] Secret messages/memories are excluded.
- [x] ContextManifest is produced.
- [x] `AssembledContext` exposes `ContextManifest` as an explicit domain object.
- [x] Full raw prompt logging is off by default.
- [x] Memory retrieval failure produces degraded context.

---

## 8. Model router

- [x] `local_main`, `local_structured`, `local_embedding` profiles exist.
- [x] `cloud_reasoning` is disabled by default.
- [x] FakeModelProvider is available for tests.
- [x] Model invocation audit records are created.
- [x] Structured output validation retry is implemented.
- [x] Embedding retry is implemented.
- [x] No automatic fallback is implemented.

---

## 9. API and streaming

- [x] `POST /v1/conversations` works.
- [x] `POST /v1/conversations/{conversation_id}/messages` works.
- [x] `GET /v1/requests/{request_id}` works.
- [x] `GET /v1/requests/{request_id}/stream` works.
- [x] `POST /v1/requests/{request_id}/cancel` works.
- [x] `POST /v1/memories` works.
- [x] `GET /v1/memories` works.
- [x] `GET /v1/health` works.
- [x] SSE emits runtime events.
- [x] `client_message_id` idempotency works.
- [x] Same `client_message_id` with different content returns conflict.
- [x] Standard error format is used.

---

## 10. Configuration

- [x] Default config validates.
- [x] Test config validates.
- [x] `JARVIS_` environment overrides apply through double-underscore nested keys.
- [x] Invalid config fails startup validation.
- [x] No secrets are stored in YAML.
- [x] Cloud is disabled by default.
- [x] Raw prompt logging is disabled by default.
- [x] Runtime budgets and windowing are config-driven.

---

## 11. Tests

- [x] Unit tests pass.
- [x] Contract tests pass.
- [x] Integration tests pass.
- [x] Golden tests pass.
- [x] Architecture tests pass.
- [x] E2E user-turn lifecycle test passes.
- [x] Standard `make test-*` targets exist for every required test layer.
- [x] Real LLM calls are not required for CI.

---

## 12. Out of MVP confirmation

- [x] No tools/MCP implementation.
- [x] No RAG/content retrieval implementation.
- [x] No ReAct/tool loop implementation.
- [x] No planner-executor implementation.
- [x] No voice pipeline.
- [x] No WebSocket/control channel.
- [x] No message edit/delete lifecycle.
- [x] No hot config reload.
