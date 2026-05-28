# 28 — MVP Acceptance Checklist

## 1. Purpose

This checklist defines what must be true before Phase 1 Core Daemon can be considered MVP-complete.

It is intentionally implementation-facing and should be used by coding agents during final acceptance review.

---

## 2. Documentation baseline

- [ ] README is current and lists all docs/ADR.
- [ ] All accepted ADRs exist.
- [ ] No accepted ADR is missing from README index.
- [ ] All MVP/post-MVP boundaries are explicit.
- [ ] TDD implementation slice plan is current or explicitly revised after agent analysis.

---

## 3. Architecture boundaries

- [ ] AgentRuntime depends on ports/domain schemas, not adapters.
- [ ] AgentRuntime does not import PostgreSQL adapters, SQLAlchemy/pgvector or provider clients.
- [ ] AgentRuntime calls `ContextAssemblerPort`, not prompt-building helpers directly.
- [ ] ContextAssembler does not call provider-specific model clients.
- [ ] Memory subsystem does not store document chunks.
- [ ] ModelRouter consults PolicyPort before model calls.
- [ ] Cloud fallback is disabled.

---

## 4. Core runtime

- [ ] Conversation can be created.
- [ ] User message submission returns `request_id`.
- [ ] `assistant_requests` lifecycle works.
- [ ] Deterministic `memory_augmented_answer` workflow runs.
- [ ] `max_model_calls=1` and `max_tool_calls=0` are enforced.
- [ ] Assistant message is persisted on success.
- [ ] No assistant message is created on system failure by default.

---

## 5. Event log

- [ ] EventEnvelope required fields are enforced.
- [ ] Events are ordered by `event_seq`.
- [ ] Events can be queried by `request_id`.
- [ ] Canonical user-turn event chain is produced.
- [ ] Causation links are recorded where applicable.
- [ ] Token-by-token events are not persisted.
- [ ] Raw full prompt is not persisted by default.

---

## 6. Memory and retrieval

- [ ] Memory namespaces are registered and validated.
- [ ] Memory types are limited to `fact`, `preference`, `procedure`, `summary`.
- [ ] Secret memory writes are denied.
- [ ] Active memory retrieval works.
- [ ] Archived/superseded memories are excluded.
- [ ] Retrieval is namespace-aware.
- [ ] Embedding failure creates memory with `indexing_status=embedding_failed`.
- [ ] Stale embeddings are excluded by content hash mismatch.

---

## 7. Context assembly

- [ ] ContextAssembler uses fixed MVP section order.
- [ ] Current user message is included.
- [ ] Recent conversation window is included within budget.
- [ ] Active retrieved memories are included.
- [ ] Secret messages/memories are excluded.
- [ ] ContextManifest is produced.
- [ ] Full raw prompt logging is off by default.
- [ ] Memory retrieval failure produces degraded context.

---

## 8. Model router

- [ ] `local_main`, `local_structured`, `local_embedding` profiles exist.
- [ ] `cloud_reasoning` is disabled by default.
- [ ] FakeModelProvider is available for tests.
- [ ] Model invocation audit records are created.
- [ ] Structured output validation retry is implemented.
- [ ] Embedding retry is implemented.
- [ ] No automatic fallback is implemented.

---

## 9. API and streaming

- [ ] `POST /v1/conversations` works.
- [ ] `POST /v1/conversations/{conversation_id}/messages` works.
- [ ] `GET /v1/requests/{request_id}` works.
- [ ] `GET /v1/requests/{request_id}/stream` works.
- [ ] SSE emits runtime events.
- [ ] `client_message_id` idempotency works.
- [ ] Same `client_message_id` with different content returns conflict.
- [ ] Standard error format is used.

---

## 10. Configuration

- [ ] Default config validates.
- [ ] Test config validates.
- [ ] Invalid config fails startup validation.
- [ ] No secrets are stored in YAML.
- [ ] Cloud is disabled by default.
- [ ] Raw prompt logging is disabled by default.
- [ ] Runtime budgets and windowing are config-driven.

---

## 11. Tests

- [ ] Unit tests pass.
- [ ] Contract tests pass.
- [ ] Integration tests pass.
- [ ] Golden tests pass.
- [ ] Architecture tests pass.
- [ ] E2E user-turn lifecycle test passes.
- [ ] Real LLM calls are not required for CI.

---

## 12. Out of MVP confirmation

- [ ] No tools/MCP implementation.
- [ ] No RAG/content retrieval implementation.
- [ ] No ReAct/tool loop implementation.
- [ ] No planner-executor implementation.
- [ ] No voice pipeline.
- [ ] No WebSocket/control channel.
- [ ] No message edit/delete lifecycle.
- [ ] No hot config reload.
