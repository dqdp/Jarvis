# 11 — Observability and Audit

## 1. Goals

Phase 1 must be debuggable and auditable.

Observability is part of the architecture, not an afterthought.

## 2. Required Logs

Structured logs include:

- request_id;
- conversation_id;
- user_id where applicable;
- selected_loop;
- model_profile;
- provider/model;
- latency;
- error status.

## 3. Required Audit Events

Event log should include:

- user.message;
- assistant.message;
- memory.retrieved;
- model.request;
- model.response;
- policy.decision;
- runtime.error.

## 4. Metrics

Basic metrics:

- request latency;
- model latency;
- memory retrieval latency;
- token counts;
- error rate;
- local inference availability;
- stream duration.

## 5. Model Invocation Audit

Every model call creates model_invocations row.

Cloud calls, when enabled in future, must be especially visible.

## 6. Policy Decision Audit

PolicyPort decisions should be recorded for:

- cloud model attempt;
- memory write;
- future tool call;
- future autonomous task.


## 7. Event Envelope and Correlation

All audit events use the stable `EventEnvelope` defined in `16_event_log_schema_and_correlation.md`.

Required correlation fields:

- `request_id` — one user turn;
- `correlation_id` — larger workflow/cross-request trace;
- `causation_id` — direct causal predecessor;
- `event_seq` — global ordering.

Phase 1 stores append-only audit/reconstruction events, but does not implement full event sourcing.

## 8. Raw Content and Prompt Logging

Raw full prompts are not stored by default.

Context assembly emits and stores `ContextManifest`, not the full prompt.

Raw message content lives in `messages`; event payloads store `message_id`, content refs/hashes and optional redacted snapshots.

Token-by-token streaming events are not persisted in event log.

## 9. Canonical Trace for One User Turn

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

## Sensitivity-aware logging

Operational logs must not contain raw private content by default.

Allowed in logs:

```text
request_id
conversation_id
event_id
model_profile
latency
token counts
memory ids
context section names
error codes
```

Forbidden by default:

```text
raw user messages
raw full prompts
raw secrets
raw credentials
large raw model outputs
```

Every event must have a sensitivity label. Raw secrets must be redacted and represented only through hashes/refs/redaction metadata.


## ModelRouter audit

Every model operation must be auditable:

- chat;
- streaming chat;
- structured output;
- embedding.

ModelRouter emits or causes:

```text
policy.decision.recorded
model.request.created
model.response.received
model.request.failed
model.request.denied
```

Token-by-token streaming events are not persisted in event log.

Operational logs should include IDs, latency, profile, provider and status, but not raw prompts or raw secrets.
