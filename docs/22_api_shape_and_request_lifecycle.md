# 22 — API Shape and Request Lifecycle

## 1. Purpose

This document defines the Phase 1 API shape and the lifecycle of one user turn.

The API must support a durable request lifecycle:

```text
create conversation
send user message
receive request_id
stream runtime events
persist assistant message
recover after retry/reconnect
```

The API must remain small enough for MVP but structured enough for future long-running workflows.

---

## 2. Core decision

Message submission and streaming are separate.

```text
POST /v1/conversations/{conversation_id}/messages
  -> creates durable request
  -> returns request_id

GET /v1/requests/{request_id}/stream
  -> streams runtime events through SSE
```

Rationale:

- better idempotency;
- easier recovery after disconnect;
- explicit request lifecycle;
- aligns with EventEnvelope and request_id;
- future-compatible with long-running/background workflows.

---

## 3. One-turn lifecycle

A single user turn:

```text
Client
  -> POST /v1/conversations/{conversation_id}/messages
Assistant API
  -> create assistant_request
  -> create user message
  -> append user.message.created event
  -> append request.processing.started event
  -> start runtime processing
Client
  -> GET /v1/requests/{request_id}/stream
Runtime
  -> emit context/model/token/assistant events
  -> create assistant message
  -> mark request completed
```

Canonical event flow:

```text
user.message.created
request.processing.started
context.assembly.started
memory.retrieved
context.assembled
model.request.created
model.response.received
assistant.message.created
request.processing.completed
```

---

## 4. API surface

Required for MVP:

```http
POST /v1/conversations
POST /v1/conversations/{conversation_id}/messages
GET  /v1/requests/{request_id}
GET  /v1/requests/{request_id}/stream
POST /v1/requests/{request_id}/cancel
GET  /v1/conversations/{conversation_id}/messages
POST /v1/memories
GET  /v1/memories
GET  /v1/health
```

Implemented Alpha control surface:

```http
GET  /v1/conversations?limit=20
GET  /v1/conversations/{conversation_id}
GET  /v1/memories?limit=100&query=optional
DELETE /v1/memories/{memory_id}
POST /v1/memories/{memory_id}/archive
GET  /v1/runtime/status
GET  /v1/approvals/pending
POST /v1/approvals/{approval_id}/grant
POST /v1/approvals/{approval_id}/deny
POST /v1/content/project-docs/ingest
POST /v1/content/project-docs/reindex
GET  /v1/content/sources
GET  /v1/content/status
```

Designed but not implemented in the current baseline:

```http
GET  /v1/conversations/{conversation_id}/events
GET  /v1/requests/{request_id}/events
POST /v1/conversations/{conversation_id}/archive
GET  /v1/model-profiles
```

---

## 5. Create conversation

Endpoint:

```http
POST /v1/conversations
```

Request:

```json
{
  "title": "Phase 1 architecture discussion",
  "active_project_namespace": "project.personal_assistant",
  "metadata": {}
}
```

Response:

```json
{
  "conversation_id": "uuid",
  "title": "Phase 1 architecture discussion",
  "active_project_namespace": "project.personal_assistant",
  "status": "active",
  "created_at": "...",
  "updated_at": "..."
}
```

If title is missing, MVP may leave it null or later set it to first user message truncated. No LLM title generation in MVP.

---

## 6. List and get conversations

Endpoints:

```http
GET /v1/conversations?limit=20
GET /v1/conversations/{conversation_id}
```

`limit` must be between 1 and 100.

`GET /v1/conversations` returns most recently active conversations first.
Conversation `updated_at` moves when messages are appended.

---

## 7. Send message

Endpoint:

```http
POST /v1/conversations/{conversation_id}/messages
```

Request:

```json
{
  "client_message_id": "client-generated-uuid",
  "content": "Давай обсудим API shape.",
  "sensitivity": "project",
  "metadata": {}
}
```

Response:

```json
{
  "request_id": "uuid",
  "conversation_id": "uuid",
  "user_message_id": "uuid",
  "status": "accepted",
  "stream_url": "/v1/requests/{request_id}/stream",
  "created_at": "..."
}
```

The runtime may start processing immediately after request creation.

---

## 8. Request status

Endpoint:

```http
GET /v1/requests/{request_id}
```

Response:

```json
{
  "request_id": "uuid",
  "conversation_id": "uuid",
  "user_message_id": "uuid",
  "assistant_message_id": "uuid|null",
  "status": "accepted|running|waiting_approval|completed|failed|cancelled",
  "created_at": "...",
  "started_at": "...",
  "completed_at": "...",
  "error": null
}
```

`queued` may be added when a real queue is introduced.

Cancellation is explicit:

```http
POST /v1/requests/{request_id}/cancel
```

It transitions an `accepted` or `running` request to `cancelled` only while
the request is still non-terminal. If completion, failure or a prior
cancellation has already won the status race, the endpoint returns the
unchanged terminal state. Client disconnect from SSE does not cancel runtime
execution.

---

## 9. SSE stream

Endpoint:

```http
GET /v1/requests/{request_id}/stream
Accept: text/event-stream
```

The stream emits normalized public runtime events, not raw provider events or
raw persisted EventEnvelope payloads. Replay uses the same public DTO
projection as the live stream, so internal fields such as manifest internals,
policy audit reasons, raw prompts or metadata are not exposed through SSE.

Examples:

```text
event: request.processing.started
data: {"request_id":"...","event_id":"..."}

event: context.assembly.started
data: {"request_id":"...","event_id":"..."}

event: memory.retrieved
data: {"request_id":"...","event_id":"..."}

event: token
data: {"request_id":"...","delta":"Да"}

event: assistant.message.created
data: {"request_id":"...","event_id":"...","message_id":"...","content_hash":"..."}

event: request.processing.completed
data: {"request_id":"...","event_id":"...","assistant_message_id":"..."}
```

Persisted events should use event names matching EventEnvelope event types.

Transient stream-only events may use names such as:

```text
token
heartbeat
```

Token events are not persisted in event log.

The implemented FastAPI adapter starts runtime execution after message
submission. Opening the SSE stream subscribes to the in-process event buffer for
active same-process streams. After terminal state and subscriber drain, the live
buffer is cleaned up and reconnect uses durable replay through
`assistant_requests`, persisted events and conversation messages.

---

## 10. Stream reconnect and recovery

No token-by-token replay guarantee in MVP.

If SSE connection drops:

```text
GET /v1/requests/{request_id}
```

can be used to check status.

If request completed:

```text
GET /v1/conversations/{conversation_id}/messages
```

returns final assistant message.

Persisted lifecycle events may be retrieved separately through:

```http
GET /v1/requests/{request_id}/events
```

if implemented.

---

## 11. Persisted events endpoint

Designed endpoint:

```http
GET /v1/requests/{request_id}/events
```

Returns persisted EventEnvelope records for this request as JSON.

This is separate from SSE stream to avoid Accept-header ambiguity.

MVP implementation may defer this endpoint if event data is otherwise inspectable.

---

## 12. Idempotency

Clients should provide:

```text
client_message_id
```

Rule:

```text
unique(conversation_id, client_message_id)
```

EventEnvelope stores the same value as `idempotency_key`; the public API
does not expose a second idempotency field in Phase 1.

If the same `client_message_id` and same content are submitted again:

```text
return existing request_id
do not create duplicate message
do not create duplicate request
```

Response may include:

```json
{
  "idempotent_replay": true
}
```

If the same `client_message_id` is reused with different content:

```text
409 Conflict
```

---

## 13. Error format

Standard error response:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "client_message_id was already used with different content",
    "request_id": "uuid|null",
    "details": {}
  }
}
```

MVP error codes:

```text
invalid_request
not_found
conflict
policy_denied
model_unavailable
request_failed
runtime_timeout
cancelled
internal_error
```

---

## 14. Cancellation

Cancel endpoint is implemented:

```http
POST /v1/requests/{request_id}/cancel
```

Behavior:

```text
accepted -> cancelled
running -> cancelled
completed/failed/cancelled -> unchanged terminal state
```

Cancellation emits `request.processing.cancelled` only when the request was
actually moved from `accepted` or `running` to `cancelled`. It is explicit; SSE
subscriber disconnect does not cancel the request.

---

## 15. Blocking mode

No separate blocking completion endpoint is required in MVP.

Primary path:

```text
POST message -> request_id
SSE stream -> final result
```

CLI clients may consume SSE and print final answer.

---

## 16. Memory API

MVP required:

```http
POST /v1/memories
GET  /v1/memories
```

Implemented Alpha control surface:

```http
GET  /v1/memories?limit=100&query=optional
DELETE /v1/memories/{memory_id}
POST /v1/memories/{memory_id}/archive
```

Designed endpoints:

```http
GET   /v1/memories/{memory_id}
PATCH /v1/memories/{memory_id}
POST  /v1/memories/{memory_id}/supersede
POST  /v1/memories/search
```

`GET /v1/memories` lists non-secret memories. `query` is a literal text
filter over memory content/summary and does not treat `%` or `_` as wildcards.
`limit` must be between 1 and 500.

`POST /v1/memories/{memory_id}/archive` is the explicit lifecycle endpoint.
`DELETE /v1/memories/{memory_id}` is a compatibility soft-delete alias that
archives the memory with reason `deleted_by_user`; it does not hard-purge data.
Both endpoints return lifecycle metadata only and do not return memory
`content` or `summary`.

`POST /v1/memories/search` remains a designed future endpoint for debugging
MemoryReadPort retrieval behavior.

---

## 17. Runtime status

Endpoint:

```http
GET /v1/runtime/status
```

Response exposes configured local model profiles and runtime budgets without
secrets. It is intended for CLI diagnostics such as `/model`.

---

## 18. Request processing transaction boundary

On message submit, one database transaction should create:

```text
assistant_request row
user message row
user.message.created event
request.processing.started event
```

Then runtime processing starts.

Assistant message is created only after final model response.

On system failure:

```text
request.status = failed
model/request error events emitted
no assistant message by default
```

---

## 19. assistant_requests table

Recommended shape:

```sql
assistant_requests (
  request_id uuid primary key,
  conversation_id uuid not null,
  user_message_id uuid not null,
  assistant_message_id uuid null,

  status text not null,
  client_message_id text null,

  created_at timestamptz not null,
  started_at timestamptz null,
  completed_at timestamptz null,

  error_code text null,
  error_message text null,

  metadata jsonb not null default '{}'
)
```

Allowed MVP statuses:

```text
accepted
running
waiting_approval
completed
failed
cancelled
```

---

## 20. MVP vs deferred

MVP includes:

- request lifecycle with `request_id`;
- explicit cancellation endpoint for `accepted` and `running` requests;
- separate SSE stream;
- client idempotency;
- request status endpoint;
- basic messages endpoint;
- basic memory create/list;
- health endpoint;
- standardized error format.

Deferred:

- blocking complete endpoint;
- conversation archive endpoint;
- full memory lifecycle endpoints;
- event trace UI;
- sophisticated pagination;
- auth/multi-user;
- rate limiting;
- WebSocket;
- OpenAPI polish.
