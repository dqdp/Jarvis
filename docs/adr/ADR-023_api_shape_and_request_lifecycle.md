# ADR-023 — API Shape and Request Lifecycle

## Status

Accepted.

## Context

Phase 1 requires a small API that supports durable conversation turns, streaming responses, idempotent retries and future long-running workflows.

A one-call `POST /messages?stream=true` style would be simpler for demos but weaker for recovery and request tracing.

## Decision

Message submission and streaming are separate.

```text
POST /v1/conversations/{conversation_id}/messages
  -> returns request_id

GET /v1/requests/{request_id}/stream
  -> SSE runtime events
```

`request_id` is the lifecycle anchor for one user turn.

`client_message_id` provides idempotency per conversation.

Same `client_message_id` with same content returns existing `request_id`.

Same `client_message_id` with different content returns `409 Conflict`.

Token-by-token events are not persisted and are not guaranteed to replay after SSE reconnect.

Final assistant message is recoverable through conversation messages.

Request statuses:

```text
accepted
running
completed
failed
```

`cancelled` is reserved.

Cancel endpoint is reserved but not required in MVP.

No blocking completion endpoint is required in MVP.

A standardized error format is used.

An `assistant_requests` table is introduced.

On system failure, no assistant message is created by default; request status becomes `failed`.

## Rationale

This API shape supports durability, idempotency, event correlation and reconnect behavior without introducing queues or background infrastructure in Phase 1.

It is future-compatible with long-running workflows and queued execution.

## Consequences

Positive:

- robust retry behavior;
- clear request tracing;
- stream reconnect can recover final state;
- aligns with EventEnvelope and request_id;
- does not require WebSocket in MVP.

Trade-offs:

- clients perform two steps instead of one;
- no token replay guarantee;
- cancellation is deferred.

## Deferred

- request cancellation;
- blocking complete endpoint;
- WebSocket transport;
- event trace UI;
- advanced pagination;
- auth/multi-user;
- rate limiting.
