# 10 — API and Streaming

## 1. Decision

Phase 1 API uses separate message submission and streaming.

```text
POST /v1/conversations/{conversation_id}/messages
  -> creates durable request
  -> returns request_id

GET /v1/requests/{request_id}/stream
  -> streams runtime events through SSE
```

Detailed request lifecycle is defined in:

```text
22_api_shape_and_request_lifecycle.md
ADR-023_api_shape_and_request_lifecycle.md
```

---

## 2. Required MVP endpoints

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

---

## 3. Streaming transport

Phase 1 streaming transport:

```text
SSE
```

WebSocket is deferred.

SSE stream emits `RuntimeStreamEvent` objects.

Provider token stream is normalized by ModelRouter/AgentRuntime and exposed as runtime token events.

Token-by-token events are not persisted in Event Log.

---

## 4. Request lifecycle

`request_id` links:

```text
user.message.created
context.assembly.started
memory.retrieved
context.assembled
model.request.created
model.response.received
assistant.message.created
request.processing.completed
```

`client_message_id` provides idempotency.

---

## 5. Reconnect

No token replay guarantee after reconnect.

Clients recover by reading:

```http
GET /v1/requests/{request_id}
GET /v1/conversations/{conversation_id}/messages
```

Persisted events may be read through:

```http
GET /v1/requests/{request_id}/events
```

if implemented.

---

## 6. Error format

Errors use a standardized response:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "...",
    "request_id": "uuid|null",
    "details": {}
  }
}
```
