# ADR-009: Streaming events as first-class runtime API

Status: Accepted

## Context

Assistant needs streaming not only for tokens but for runtime progress, memory, model and future tool events.

## Decision

Define transport-agnostic RuntimeStreamEvent. Use SSE in Phase 1.

## Consequences

Simple MVP; WebSocket can be added later without changing event schema.
