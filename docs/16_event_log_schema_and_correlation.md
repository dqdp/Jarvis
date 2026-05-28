# 16 — Event Log Schema and Correlation Model

## 1. Назначение

Event log в Phase 1 является immutable historical truth о действиях системы.

Он отвечает на вопрос:

> Что реально произошло в системе, в каком порядке, по какой причине и с какими последствиями?

Event log не является утверждением об истинности мира. Если модель дала неверный ответ, event log фиксирует не то, что ответ был верным, а то, что `model.response.received` был получен с конкретным результатом и метаданными.

---

## 2. Event log vs domain tables

Phase 1 не строит полноценную event-sourced систему.

Решение:

> Используем append-only event log как audit/reconstruction substrate. Domain tables остаются operational read/write state.

Примеры:

```text
messages
  operational read model для UI и conversation history

events
  append-only historical record: user.message.created, assistant.message.created
```

```text
memories
  текущее состояние интерпретированной долгосрочной памяти

events
  memory.created / memory.updated / memory.superseded / memory.archived
```

```text
model_invocations
  audit/details of model call

events
  model.request.created / model.response.received / model.request.failed
```

---

## 3. Historical truth definition

Historical truth включает:

- пользователь отправил сообщение;
- система начала обработку request;
- ContextAssembler собрал контекст;
- были извлечены такие-то memories;
- ModelRouter вызвал такую-то модель через такой-то profile;
- модель вернула ответ или ошибку;
- ассистент создал сообщение;
- memory была создана, обновлена, заменена или архивирована;
- PolicyPort принял решение.

Historical truth не включает:

- фактическую правильность ответа модели;
- истинность memory content как знания о мире;
- LangGraph checkpoint как доменную истину;
- embedding vector как знание;
- ContextManifest как первичный факт мира.

Но event log фиксирует факт создания/использования этих артефактов.

---

## 4. EventEnvelope

Все события используют стабильный envelope.

```json
{
  "event_id": "uuid",
  "event_seq": 12345,
  "event_type": "model.request.created",
  "event_version": 1,

  "occurred_at": "2026-05-28T15:20:31.123Z",
  "recorded_at": "2026-05-28T15:20:31.130Z",

  "conversation_id": "uuid|null",
  "request_id": "uuid|null",
  "correlation_id": "uuid|null",
  "causation_id": "uuid|null",
  "parent_event_id": "uuid|null",

  "actor_type": "user|assistant|system|model|tool|scheduler",
  "actor_id": "string|null",

  "source_component": "assistant_api|agent_runtime|context_assembler|model_router|memory_service|policy_engine",
  "source_node": "string|null",

  "sensitivity": "public|personal|project|infra|secret",
  "visibility": "internal|user_visible|debug",

  "idempotency_key": "string|null",

  "payload": {},
  "metadata": {}
}
```

Envelope должен быть стабильнее, чем payload. Payload может версионироваться через `event_version`.

---

## 5. Identity and correlation fields

### 5.1 `event_id`

Уникальный идентификатор события.

### 5.2 `event_seq`

Глобальная монотонная последовательность записи события в PostgreSQL.

Не полагаться только на timestamp для ordering.

### 5.3 `conversation_id`

Связывает событие с conversation. Для большинства Phase 1 событий обязателен.

Некоторые future events могут быть вне conversation:

- system.startup;
- scheduled_task.created;
- sleep.workflow.started.

### 5.4 `request_id`

Один user turn / одна обработка user input.

Все события цепочки ответа на одно пользовательское сообщение имеют один `request_id`:

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

### 5.5 `correlation_id`

Более широкий trace id для long-running workflows и cross-request chains.

Phase 1:

```text
correlation_id may equal request_id
```

Future:

```text
correlation_id может связывать multi-step plan, sleep workflow, tool workflow или proactive task.
```

### 5.6 `causation_id`

ID события, которое непосредственно вызвало текущее событие.

Пример:

```text
context.assembly.started
  causation_id = user.message.created

model.request.created
  causation_id = context.assembled

assistant.message.created
  causation_id = model.response.received
```

### 5.7 `parent_event_id`

Зарезервировано для будущей иерархии событий. В Phase 1 может быть `null`.

---

## 6. Minimal Phase 1 event types

### 6.1 Request lifecycle

```text
request.processing.started
request.processing.completed
request.processing.failed
```

### 6.2 Messages

```text
user.message.created
assistant.message.created
```

### 6.3 Context

```text
context.assembly.started
context.assembled
context.assembly.failed
context.assembly.truncated
```

`context.assembly.truncated` emits only when items were dropped due to budget.

### 6.4 Memory

```text
memory.retrieved
memory.retrieval.failed
memory.embedding.created
memory.embedding.failed
memory.created
memory.updated
memory.archived
memory.superseded
```

### 6.5 Model

```text
model.request.created
model.response.received
model.request.failed
model.request.denied
```

### 6.6 Policy

```text
policy.decision.recorded
```

---

## 7. Canonical user turn event flow

```text
1. user.message.created
2. request.processing.started
3. context.assembly.started
4. memory.retrieved
5. context.assembled
6. model.request.created
7. model.response.received
8. assistant.message.created
9. request.processing.completed
```

Causation chain:

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

All events share the same `request_id`.

---

## 8. Raw content policy

### 8.1 Messages

`messages` table stores raw user-visible content.

Events store by default:

- `message_id`;
- `content_hash`;
- `content_ref`;
- optional redacted snapshot.

Events do not store full raw message content by default.

### 8.2 Prompt / assembled context

Full raw prompt is not stored by default.

`context.assembled` stores `ContextManifest`, including:

- context_manifest_id;
- section names;
- source refs;
- used memory IDs;
- used message IDs;
- token estimates;
- dropped items;
- `full_prompt_stored = false`.

Debug mode may allow full prompt capture, but it must be explicitly enabled.

### 8.3 Model streaming tokens

Token-by-token model stream events are not persisted in event log.

SSE may emit token events to client, but event log stores only aggregate model response metadata and final assistant message.

---

## 9. Example event payloads

### 9.1 `user.message.created`

```json
{
  "message_id": "msg-uuid",
  "role": "user",
  "content_hash": "sha256:...",
  "content_ref": "messages:msg-uuid",
  "content_snapshot_policy": "not_stored"
}
```

### 9.2 `memory.retrieved`

```json
{
  "query_text_hash": "sha256:...",
  "namespaces": ["user.preferences", "project.personal_assistant"],
  "include_statuses": ["active"],
  "memory_hits": [
    {
      "memory_id": "mem-uuid",
      "namespace": "project.personal_assistant",
      "memory_type": "fact",
      "score": 0.82,
      "used_in_context": true
    }
  ]
}
```

### 9.3 `context.assembled`

```json
{
  "context_manifest_id": "ctx-uuid",
  "model_profile": "local_main",
  "token_estimate": 7600,
  "sections": [
    {
      "name": "system_identity",
      "token_estimate": 200,
      "source_refs": []
    },
    {
      "name": "project_memory",
      "token_estimate": 1200,
      "source_refs": ["memory:mem-1", "memory:mem-2"]
    },
    {
      "name": "recent_conversation",
      "token_estimate": 3200,
      "source_refs": ["message:msg-10", "message:msg-11"]
    }
  ],
  "dropped": [
    {
      "kind": "memory",
      "id": "mem-9",
      "reason": "token_budget"
    }
  ],
  "full_prompt_stored": false
}
```

### 9.4 `model.request.created`

```json
{
  "model_invocation_id": "mi-uuid",
  "provider": "local_vllm",
  "model": "qwen-or-mistral-local",
  "profile": "local_main",
  "streaming": true,
  "context_manifest_id": "ctx-uuid",
  "input_token_estimate": 7600
}
```

### 9.5 `model.response.received`

```json
{
  "model_invocation_id": "mi-uuid",
  "provider": "local_vllm",
  "model": "qwen-or-mistral-local",
  "profile": "local_main",
  "status": "success",
  "latency_ms": 4300,
  "input_tokens": 7600,
  "output_tokens": 950
}
```

### 9.6 `policy.decision.recorded`

```json
{
  "decision_type": "model_provider_access",
  "subject": "cloud_reasoning",
  "decision": "deny",
  "reason": "cloud_models_disabled_by_default",
  "policy_version": "phase1-config-policy"
}
```

---

## 10. PostgreSQL table baseline

```sql
create table events (
    event_id uuid primary key,
    event_seq bigserial not null,

    event_type text not null,
    event_version int not null default 1,

    occurred_at timestamptz not null,
    recorded_at timestamptz not null default now(),

    conversation_id uuid null,
    request_id uuid null,
    correlation_id uuid null,
    causation_id uuid null references events(event_id),
    parent_event_id uuid null references events(event_id),

    actor_type text not null,
    actor_id text null,

    source_component text not null,
    source_node text null,

    sensitivity text not null default 'personal',
    visibility text not null default 'internal',

    idempotency_key text null,

    payload jsonb not null default '{}',
    metadata jsonb not null default '{}'
);

create index events_conversation_seq_idx on events(conversation_id, event_seq);
create index events_request_seq_idx on events(request_id, event_seq);
create index events_correlation_seq_idx on events(correlation_id, event_seq);
create index events_type_idx on events(event_type);
create index events_causation_idx on events(causation_id);
```

---

## 11. Idempotency

API-level message submission should support idempotency.

Phase 1 baseline:

- public API clients provide `client_message_id`;
- repeated submission with the same `client_message_id` should not create duplicate user messages;
- EventEnvelope stores this value in `idempotency_key`;
- related events include the same `idempotency_key`.

Recommended table support:

```text
messages.client_message_id nullable unique per conversation
```

---

## 12. Error events

Errors are first-class events.

Examples:

```text
context.assembly.failed
model.request.failed
request.processing.failed
```

Error payload should include:

- error_type;
- error_message or redacted message;
- retryable;
- related IDs;
- component.

---

## 13. Phase 1 decisions

Accepted decisions:

1. Event log is immutable historical truth about system actions.
2. Phase 1 uses append-only audit/reconstruction log, not full event sourcing.
3. All events use stable EventEnvelope.
4. `request_id` links one user turn.
5. `correlation_id` is reserved for cross-request/long-running workflows; in Phase 1 it may equal `request_id`.
6. `causation_id` links direct causal chain.
7. Full raw prompts are not stored by default.
8. Raw message content lives in `messages`; events store message refs/hashes/redacted snapshots by default.
9. Token-by-token streaming events are not persisted in event log.
10. Event log records memory retrieval, context assembly, model invocation, policy decisions and assistant message creation.

## Sensitivity in EventEnvelope

Every event carries a `sensitivity` label.

Allowed values:

```text
public
project
personal
infra
secret
```

Event sensitivity should be derived from the maximum sensitivity of the event payload and source artifacts.

Rules:

- raw secrets must not be stored in event payloads;
- raw full prompts must not be stored by default;
- secret-bearing events must use redacted payloads;
- refs/hashes/manifests are preferred over raw snapshots;
- cloud-related events must record policy decisions.

Example redacted event payload:

```json
{
  "message_id": "...",
  "content_hash": "sha256:...",
  "redacted": true,
  "redaction_reason": "contains_secret"
}
```


## Conversation/message relation

Messages are operational read models.

Events are historical truth.

When a user message is created, the system creates:

```text
messages row
event: user.message.created
```

When an assistant response is created, the system creates:

```text
messages row
event: assistant.message.created
```

Both user and assistant messages for one turn share the same `request_id`.

Events should reference `message_id` and content hash rather than duplicate raw content by default.


## API request lifecycle relation

`request_id` is created when `POST /v1/conversations/{conversation_id}/messages` accepts a user message.

All events for that user turn carry the same `request_id`.

SSE stream exposes runtime events for a request through:

```text
GET /v1/requests/{request_id}/stream
```

Persisted events may be exposed through:

```text
GET /v1/requests/{request_id}/events
```

Token stream deltas are not persisted in Event Log.


## Post-MVP agent step events

Phase 1 does not require step-level agent events.

Future agent loops should introduce:

```text
agent.step.started
agent.step.completed
agent.step.failed
```

Future correlation model:

```text
request_id:
  one user interaction

correlation_id:
  long-running workflow or task

step_id:
  one agent iteration
```

Tool observations should be recorded as events/runtime state, not as conversation messages by default.
