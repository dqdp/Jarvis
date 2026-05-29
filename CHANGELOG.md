# Changelog

## mvp-0.1 — 2026-05-29

Phase 1 Core Daemon MVP is complete.

Included:

- durable PostgreSQL-backed conversations, messages, requests, events,
  memories and model invocation audit records;
- deterministic `memory_augmented_answer` runtime through explicit ports;
- FastAPI assistant API and SSE runtime stream;
- local-first policy with cloud fallback disabled;
- native Ollama dogfood profile using `qwen3.5:9b`;
- local CLI with interactive chat, slash commands, in-session history, manual
  memory commands and request cancellation on stream interruption;
- anti-repeat protection for local Ollama generation loops;
- unit, contract, integration, golden, architecture and e2e test layers.

Known limitations are tracked in `docs/32_known_limitations.md`.
