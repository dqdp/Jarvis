# 23 — Error Handling and Runtime Budgets

## 1. Purpose

This document defines Phase 1 error handling, degraded-mode behavior, runtime limits and retry policy.

The goal is to keep the Core Daemon predictable and safe without turning the MVP into a workflow engine.

---

## 2. Core principle

Phase 1 has one primary loop strategy:

```text
memory_augmented_answer
```

The budgets below apply to this loop strategy only.

They are configurable defaults, not global architectural limits.

Future loop strategies must define their own budgets, allowed capabilities, stopping conditions, policy hooks, failure semantics and emitted events.

---

## 3. Phase 1 runtime budget

Default budget:

```yaml
runtime_budgets:
  memory_augmented_answer:
    max_model_calls: 1
    max_tool_calls: 0
    max_wall_time_seconds: 180
    max_context_assembly_seconds: 10
    max_memory_retrieval_seconds: 5
    max_model_call_seconds: 120
    max_output_tokens: 2048
    allow_cloud: false
    allow_tools: false
    allow_autonomous_memory_write: false
```

Key constraints:

```text
max_model_calls = 1
max_tool_calls = 0
```

This prevents Phase 1 from accidentally becoming a ReAct/tool loop.

---

## 4. Error categories

Phase 1 recognizes these error categories:

```text
client_error
policy_error
dependency_error
runtime_error
model_error
```

Examples:

```text
client_error:
  invalid payload, unknown conversation, idempotency conflict

policy_error:
  cloud call denied, secret attempted to enter context, secret memory write

dependency_error:
  local inference unavailable, embedding backend unavailable, PostgreSQL unavailable

runtime_error:
  invalid runtime state, ContextAssembler exception, event append failure

model_error:
  timeout, stream interrupted, invalid structured output, context length exceeded
```

---

## 5. Fatal vs degraded mode

Not every failure is fatal.

### Degraded mode

Allowed degraded behavior:

```text
memory retrieval failure
retrieval query embedding failure
optional context section unavailable
embedding failure during memory write
```

For normal chat, memory retrieval failure is non-fatal:

```text
emit memory.retrieval.failed
assemble context with degraded=true
answer using recent conversation and runtime context
```

### Fatal errors

Fatal for request processing:

```text
EventLog append failure after request is accepted
ConversationStore/PostgreSQL unavailable for accepted request lifecycle
Policy denies model request
secret would enter model context
local_main model call fails
minimal context cannot be assembled
current user message too large
```

Rule:

```text
No accepted request without event log.
```

---

## 6. Minimal required context

Long-term memory is optional.

Current user message and mandatory system/runtime constraints are required.

Minimum viable context:

```text
system_identity
runtime_rules
current_user_message
```

If this cannot be built, request fails.

---

## 7. Component behavior

### ConversationStore

ConversationStore failure is generally fatal for request lifecycle.

### EventLog

Event append failure for accepted request is fatal.

### MemoryReadPort

Retrieval failure is non-fatal for normal chat and causes degraded context.

### MemoryWritePort / EmbeddingPort

Memory may be created even if embedding fails:

```text
indexing_status=embedding_failed
memory.embedding.failed event
```

### ContextAssembler

Optional memory section failure may degrade.

Failure to build minimal context is fatal.

### ModelRouter

`local_main` failure is fatal for the request.

No automatic fallback in Phase 1.

---

## 8. Timeout policy

Default timeouts:

```text
context assembly: 10s
memory retrieval: 5s
local_main model call: 120s
local_structured model call: 60s
local_embedding call: 30s
```

Memory retrieval timeout:

```text
degraded context
```

Chat model timeout:

```text
request failed
```

Embedding timeout:

```text
retry once, then embedding_failed
```

Structured timeout:

```text
no retry by default
```

---

## 9. Retry policy

Accepted retry baseline:

```text
chat retry: 0
structured validation retry: 1
embedding retry: 1
```

Details:

```text
chat provider timeout: no retry
chat stream interrupted: no retry
structured invalid JSON/schema: one repair/validation retry
structured timeout/provider unavailable: no retry by default
embedding timeout/provider unavailable: one retry
```

---

## 10. Partial streaming failure

If provider stream fails after partial tokens have been sent:

```text
request.status = failed
model.request.failed event
request.processing.failed event
partial_output_persisted = false
```

Partial streamed tokens are not persisted as assistant message in MVP.

The SSE client may have seen them, but they are not historical assistant output.

---

## 11. Assistant message on failure

On system failure, no assistant message is created by default.

The failure is visible through:

```text
GET /v1/requests/{request_id}
SSE request.processing.failed
event log
```

Post-MVP may introduce configurable user-visible assistant error messages.

---

## 12. Error events

Minimum error-related events:

```text
request.processing.failed
runtime.error
context.assembly.failed
memory.retrieval.failed
model.request.failed
policy.decision.recorded
memory.embedding.failed
```

Error payloads must be sanitized.

They must not contain:

```text
raw prompt
raw secret
raw credential
unredacted sensitive payload
```

---

## 13. API error mapping

Recommended mapping:

```text
invalid_request -> 400
payload_too_large -> 413
not_found -> 404
conflict -> 409
policy_denied -> 403
model_unavailable -> 503
request_failed -> 500
internal_error -> 500
```

For accepted async requests, failures are surfaced through request status, SSE and events.

---

## 14. Request status transitions

Allowed MVP transitions:

```text
accepted -> running
accepted -> failed
running -> completed
running -> failed
```

Reserved:

```text
running -> cancelled
```

Not allowed:

```text
completed -> running
failed -> running
completed -> failed
```

No retry/resume of same request in MVP.

A repeated `client_message_id` returns the existing request status.

---

## 15. Budget exceeded behavior

If context is too large:

```text
trim droppable sections
drop low-ranked memories
drop oldest conversation messages
```

If current user message alone is too large:

```text
fail request
error_code=message_too_large
```

If model finishes due to output token limit:

```text
finish_reason=length
assistant.message.created still occurs
not a system failure
```

---

## 16. MVP vs deferred

MVP includes:

```text
runtime budgets for memory_augmented_answer
timeouts for context/memory/model
retry policy
degraded mode for memory retrieval failure
hard fail for model failure
hard fail for policy denied
hard fail if event log cannot be written
request status transitions
standard error codes
sanitized error events
no assistant message on system failure
no partial response persistence
```

Deferred:

```text
request cancellation implementation
request retry/resume
partial assistant message persistence
stream resume
workflow compensation
dead letter queue
background retry workers
circuit breakers
rate limiting
provider failover
user-visible assistant error messages
advanced incident reporting
```


## Configuration relation

Runtime budgets and timeout/retry defaults are config-driven.

The Phase 1 values for `memory_augmented_answer` are defaults, not hardcoded global limits.
