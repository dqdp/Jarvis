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

## 4. Minimal MVP endpoints

Required for MVP:

```http
POST /v1/conversations
POST /v1/conversations/{conversation_id}/messages
GET  /v1/requests/{request_id}
GET  /v1/requests/{request_id}/stream
GET  /v1/conversations/{conversation_id}/messages
POST /v1/memories
GET  /v1/memories
GET  /v1/health
```

Designed but not necessarily required in first implementation:

```http
GET  /v1/conversations/{conversation_id}
GET  /v1/conversations/{conversation_id}/events
GET  /v1/requests/{request_id}/events
POST /v1/conversations/{conversation_id}/archive
POST /v1/requests/{request_id}/cancel
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

## 6. Send message

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

## 7. Request status

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
  "status": "accepted|running|completed|failed",
  "created_at": "...",
  "started_at": "...",
  "completed_at": "...",
  "error": null
}
```

Reserved future status:

```text
cancelled
```

`queued` may be added when a real queue is introduced.

---

## 8. SSE stream

Endpoint:

```http
GET /v1/requests/{request_id}/stream
Accept: text/event-stream
```

The stream emits normalized runtime events, not raw provider events.

Examples:

```text
event: request.processing.started
data: {"request_id":"...","event_id":"..."}

event: context.assembly.started
data: {"request_id":"..."}

event: memory.retrieved
data: {"hit_count":4,"used_memory_ids":["..."]}

event: token
data: {"delta":"Да"}

event: assistant.message.created
data: {"message_id":"..."}

event: request.processing.completed
data: {"assistant_message_id":"...","status":"completed"}
```

Persisted events should use event names matching EventEnvelope event types.

Transient stream-only events may use names such as:

```text
token
heartbeat
```

Token events are not persisted in event log.

---

## 9. Stream reconnect and recovery

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

## 10. Persisted events endpoint

Designed endpoint:

```http
GET /v1/requests/{request_id}/events
```

Returns persisted EventEnvelope records for this request as JSON.

This is separate from SSE stream to avoid Accept-header ambiguity.

MVP implementation may defer this endpoint if event data is otherwise inspectable.

---

## 11. Idempotency

Clients should provide:

```text
client_message_id
```

Rule:

```text
unique(conversation_id, client_message_id)
```

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

## 12. Error format

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
internal_error
```

---

## 13. Cancellation

Cancel endpoint is reserved but not required in MVP:

```http
POST /v1/requests/{request_id}/cancel
```

Rationale for deferral:

- provider stream cancellation needs careful implementation;
- request state synchronization becomes more complex;
- Phase 1 can operate without cancellation.

---

## 14. Blocking mode

No separate blocking completion endpoint is required in MVP.

Primary path:

```text
POST message -> request_id
SSE stream -> final result
```

CLI clients may consume SSE and print final answer.

---

## 15. Memory API

MVP required:

```http
POST /v1/memories
GET  /v1/memories
```

Designed endpoints:

```http
GET   /v1/memories/{memory_id}
PATCH /v1/memories/{memory_id}
POST  /v1/memories/{memory_id}/archive
POST  /v1/memories/{memory_id}/supersede
POST  /v1/memories/search
```

`POST /v1/memories/search` is useful for debugging MemoryReadPort and retrieval behavior.

---

## 16. Request processing transaction boundary

On message submit, one database transaction should create:

```text
assistant_request row
user message row
user.message.created event
request.processing.started or accepted event
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

## 17. assistant_requests table

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
completed
failed
```

Reserved:

```text
cancelled
```

---

## 18. MVP vs deferred

MVP includes:

- request lifecycle with `request_id`;
- separate SSE stream;
- client idempotency;
- request status endpoint;
- basic messages endpoint;
- basic memory create/list;
- health endpoint;
- standardized error format.

Deferred:

- request cancellation;
- blocking complete endpoint;
- conversation archive endpoint;
- full memory lifecycle endpoints;
- event trace UI;
- sophisticated pagination;
- auth/multi-user;
- rate limiting;
- WebSocket;
- OpenAPI polish.
