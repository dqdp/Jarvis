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
- Tools, RAG/content ingestion and a bounded tool loop are now implemented as
  post-MVP Alpha slices, not as original MVP scope.
- Automatic loop selection and CLI readiness now cover PM-08a through PM-08i,
  including the Codex-like interactive shell surface. Canonical Jarvis runtime
  startup is tracked as PM-08j and request routing architecture review/classifier
  calibration is tracked as PM-08k before PM-09. PM-09 voice, durable graph
  checkpoints and full-screen TUI behavior remain future work.
- MCP, planner-executor, voice, remembered approvals, durable background
  workers and external integrations remain future work.
- Approval pauses are durable as approval records and request status, but there
  is no general graph checkpoint/resume engine yet.
- Cloud model fallback is disabled and must remain disabled until a future ADR
  changes the policy.

## Alpha Priorities

Implemented first Alpha wave:

- CLI control surface and conversation navigation.
- Memory search/delete lifecycle.
- Capability policy, permission modes and one-shot approvals.
- ToolGateway, safe tools, read-only shell and system diagnostics.
- Project docs content ingestion/retrieval and ContextAssembler integration.
- Model behavior tuning and smoke/eval scenarios.

Completed PM-08 Alpha sequence so far:

- PM-08a loop selection domain and selector contract;
- PM-08b API/request lifecycle auto mode;
- PM-08c CLI auto mode and mode controls;
- PM-08d CLI tool/RAG/approval readiness surface;
- PM-08e Model-backed intent classifier adapter;
- PM-08f Typed tool observations and direct-answer hardening;
- PM-08g Direct planner and capability routing registry cleanup;
- PM-08h Tool-intent corpus hardening and pre-voice routing gate;
- PM-08i interactive CLI shell UX hardening.

Next priority:

- PM-08j canonical Jarvis runtime startup;
- PM-08k request routing architecture review and classifier calibration;
- PM-09 voice gateway foundation;

Later priority candidates:

- durable task queue / resumable workflow checkpoints;
- broader context management and summaries;
- MCP gateway;
- external integrations;
- richer voice interaction beyond the PM-09 foundation.

Manual model smoke coverage is tracked in `docs/33_alpha_model_behavior_smoke.md`.
