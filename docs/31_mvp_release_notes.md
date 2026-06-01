# 31 — MVP Release Notes

## Release

`mvp-0.1`

Date: 2026-05-29

Status: complete.

## Summary

Phase 1 Core Daemon MVP is complete and accepted as the baseline for post-MVP
Alpha work.

The release provides a local-first assistant runtime with durable PostgreSQL
state, explicit ports/adapters boundaries, deterministic memory-augmented
runtime, local model routing, SSE streaming, manual long-term memory and a
usable local CLI.

## Acceptance Evidence

The MVP has green coverage across the required layers:

```text
unit
contract
integration
golden
architecture
e2e
```

Real LLM calls are not required for CI; fake model and embedding providers are
used for deterministic acceptance.

## Runtime Dogfood

The local dogfood path uses:

```text
CONFIG_PROFILE=ollama make run
make cli
```

The Ollama profile uses `qwen3.5:9b` for chat, `qwen3.5:2b` for structured
classification, and `embeddinggemma:latest` for embeddings.
PM-08k supersedes classifier-first routing for the production request path:
normal typed input and future voice transcripts should enter the bounded agent
loop, with tool execution controlled by PolicyPort and ToolGatewayPort.
Structured classifier model notes remain historical/evaluation context.

## Scope Boundary

The release intentionally does not include tools, MCP, RAG/content ingestion,
ReAct/planner-executor, autonomous background workers, cloud fallback, voice or
Telegram integrations.

Those are post-MVP capabilities and require separate design slices.
