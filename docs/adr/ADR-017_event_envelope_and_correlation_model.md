# ADR-017 — Event Envelope and Correlation Model

## Status

Accepted.

## Context

Phase 1 needs a reliable audit/reconstruction substrate. The assistant must be able to explain one response in terms of user input, selected context, retrieved memories, model invocation, policy decisions and final assistant message.

At the same time, Phase 1 should not become a fully event-sourced system.

## Decision

Phase 1 uses an append-only event log as immutable historical truth about system actions, not as a full event-sourcing implementation.

All events use a stable `EventEnvelope` with:

- `event_id`;
- `event_seq`;
- `event_type`;
- `event_version`;
- `occurred_at`;
- `recorded_at`;
- `conversation_id`;
- `request_id`;
- `correlation_id`;
- `causation_id`;
- `actor_type` / `actor_id`;
- `source_component` / `source_node`;
- `sensitivity`;
- `visibility`;
- `idempotency_key`;
- `payload`;
- `metadata`.

`request_id` identifies one user turn. All events produced while answering a single user message share the same `request_id`.

`correlation_id` is reserved for larger workflows that may span multiple requests. In Phase 1 it may equal `request_id`.

`causation_id` links the direct causal chain between events.

## Accepted event flow for one user turn

```text
user.message.created
  → request.processing.started
    → context.assembly.started
      → memory.retrieved
      → context.assembled
        → model.request.created
          → model.response.received
            → assistant.message.created
              → request.processing.completed
```

## Raw content policy

Raw full prompts are not stored by default. `context.assembled` stores `ContextManifest`.

Raw message content is stored in `messages`. Events store `message_id`, `content_hash`, content refs and optional redacted snapshots.

Token-by-token streaming events are not persisted in event log.

## Consequences

Positive:

- one user turn can be traced through request, context, memory, model and response;
- later sleep/reflection and memory reconstruction can use event history;
- event log remains privacy-aware by avoiding raw prompt storage by default;
- domain tables can stay simple operational read/write models.

Trade-offs:

- event log is not fully self-contained by default because raw message content is referenced, not duplicated;
- debug workflows may require joining events with domain tables;
- future privacy/purge semantics must account for both messages and event payload references.

## Non-goals

- full event sourcing;
- token-by-token persisted model traces;
- storing raw full prompts by default;
- using graph checkpoints as historical truth.
