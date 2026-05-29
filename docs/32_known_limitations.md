# 32 — Known Limitations

## MVP Limitations

- CLI is the primary usable interface; there is no polished web UI.
- Conversation history uses recent-window context only; there are no rolling
  summaries yet.
- Memory writes are manual. There is no autonomous memory extraction.
- Memory retrieval uses the current PostgreSQL adapter and deterministic
  ranking; pgvector remains an adapter-level optimization follow-up.
- Memory lifecycle is soft-state oriented: archive/supersede are normal
  operations, hard purge is deferred to a future privacy slice.
- Local model answer quality depends on the installed Ollama model and profile
  parameters.
- There is no hot config reload; daemon restart is required after runtime
  config changes.
- Request execution is in-process. There is no durable queue or background
  worker system yet.
- Tools, MCP, RAG/content ingestion, ReAct/planner-executor, voice and external
  integrations are intentionally out of MVP.
- Cloud model fallback is disabled and must remain disabled until a future ADR
  changes the policy.

## Alpha Priorities

1. CLI control surface and conversation navigation.
2. Memory search/delete lifecycle.
3. Model behavior tuning and smoke/eval scenarios.
4. Later, content retrieval/RAG and tool-capable loop strategies.

The first three Alpha priorities are implemented as the first post-MVP slice.
Manual model smoke coverage is tracked in `docs/33_alpha_model_behavior_smoke.md`.
