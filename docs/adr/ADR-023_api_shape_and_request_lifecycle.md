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
waiting_approval
completed
failed
cancelled
```

Cancel endpoint is implemented for MVP hardening:

```http
POST /v1/requests/{request_id}/cancel
```

Cancellation changes only a still non-terminal `accepted`, `running` or
`waiting_approval` request. If `completed`, `failed` or `cancelled` has already
won the status race, the terminal state is returned unchanged.

SSE stream opening subscribes to public runtime events. It does not execute the
request itself and does not expose raw persisted EventEnvelope payloads. Runtime
execution starts after message submission, and reconnect must not re-run the
model provider.

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
- SSE disconnect does not strand a request in `running`;
- aligns with EventEnvelope and request_id;
- does not require WebSocket in MVP.

Trade-offs:

- clients perform two steps instead of one;
- no token replay guarantee;
- cancellation requires an explicit second API call.

## Deferred

- blocking complete endpoint;
- WebSocket transport;
- event trace UI;
- advanced pagination;
- auth/multi-user;
- rate limiting.
