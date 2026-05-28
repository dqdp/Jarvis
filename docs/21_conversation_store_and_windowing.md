# 21 — Conversation Store and Windowing

## 1. Purpose

This document defines the Phase 1 baseline for storing conversations/messages and selecting recent conversation context for model calls.

The goal is to keep the MVP simple while avoiding hardcoded policies that would block future context strategies.

---

## 2. Core distinction

The system separates:

```text
Conversation Store
  durable operational read/write model for conversations and messages

Event Log
  immutable historical truth about system actions

Context Window
  selected recent subset of conversation history for one model call

Long-Term Memory
  explicit interpreted memory records
```

Conversation Store is not Memory subsystem.

Conversation history is not vector-indexed in Phase 1.

---

## 3. Conversation Store responsibilities

`ConversationStorePort` is responsible for:

- creating conversations;
- storing user/assistant messages;
- loading messages for UI/API;
- loading recent messages for ContextAssembler;
- linking messages to `request_id` and event IDs;
- preserving conversation history across daemon restarts.

It is not responsible for:

- vector retrieval;
- long-term memory;
- event causality;
- prompt assembly;
- model invocation audit;
- rolling summaries.

---

## 4. Conversation model

Minimal table shape:

```sql
conversations (
  conversation_id uuid primary key,
  user_id text not null,
  title text null,
  active_project_namespace text null,
  status text not null default 'active',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  metadata jsonb not null default '{}'
)
```

Allowed Phase 1 statuses:

```text
active
archived
```

`active_project_namespace` is used by ContextAssembler to select project-specific memory namespaces.

---

## 5. Message model

Minimal table shape:

```sql
messages (
  message_id uuid primary key,
  conversation_id uuid not null references conversations(conversation_id),
  request_id uuid null,
  event_id uuid null references events(event_id),
  client_message_id text null,
  role text not null,
  content text not null,
  content_hash text not null,
  sensitivity text not null default 'personal',
  created_at timestamptz not null default now(),
  metadata jsonb not null default '{}'
)
```

Recommended indexes:

```sql
create index messages_conversation_created_idx
on messages(conversation_id, created_at);

create index messages_request_idx
on messages(request_id);
```

Recommended idempotency constraint:

```sql
unique(conversation_id, client_message_id)
where client_message_id is not null
```

---

## 6. Role taxonomy

Phase 1 persisted roles:

```text
user
assistant
```

Reserved future roles:

```text
system
tool
developer
```

System/runtime prompt messages are not stored as conversation messages in MVP.

Rationale:

- system/runtime rules belong to ContextAssembler/PromptTemplateRegistry;
- they may vary by model profile and loop strategy;
- they are recorded through ContextManifest and events, not as ordinary user-facing dialogue.

---

## 7. Message content

Phase 1 message content is text-only:

```text
content: text
```

Post-MVP may introduce provider-neutral typed content parts:

```text
text
image_ref
file_ref
audio_ref
tool_result_ref
structured_data_ref
```

Text-only content is an MVP simplification, not a permanent architectural constraint.

---

## 8. Event linkage

For one user turn:

```text
request_id = req1

user message:
  role=user
  request_id=req1

assistant message:
  role=assistant
  request_id=req1
```

The corresponding event flow:

```text
user.message.created
...
assistant.message.created
```

Messages are operational read models; events are historical truth.

---

## 9. Windowing responsibility

Windowing policy belongs to `ContextAssembler`, not `ConversationStore`.

`ConversationStorePort` should provide recent messages. `ContextAssembler` applies:

- role filtering;
- token budget;
- trimming strategy;
- sensitivity filtering;
- loop-strategy-specific rules.

Reason:

Different loop strategies may need different window policies.

---

## 10. MVP windowing defaults

Default MVP policy:

```yaml
conversation_window:
  max_messages: 12
  max_tokens: 3000
  include_roles:
    - user
    - assistant
  trimming_strategy: drop_oldest_first
  exclude_sensitivity:
    - secret
```

These are **configurable policy defaults**, not hardcoded architecture.

`max_messages=12` and `max_tokens=3000` are starting defaults and may be changed through configuration.

---

## 11. Trimming strategy

MVP trimming:

```text
Load latest messages.
Preserve chronological order.
Estimate tokens.
If over budget, drop oldest messages first.
Never include secret messages.
```

This is the default Phase 1 strategy, not a permanent rule.

Post-MVP strategies may include:

- summary + recent tail;
- salience-based trimming;
- role-aware trimming;
- task-aware trimming;
- tool-aware trimming;
- planner-specific context windows.

---

## 12. Rolling summaries

Automatic rolling conversation summaries are not part of MVP.

Reason:

They require:

- summary generation workflow;
- validity and update policy;
- provenance;
- additional model calls;
- distinction from long-term memory summaries.

Post-MVP may introduce:

```text
conversation_summary + recent_tail
```

This must remain behind ContextAssembler policy.

---

## 13. Long-term memory summary vs conversation summary

`memory_type=summary` is a long-term memory record.

A future rolling conversation summary is a working context artifact.

They must not be conflated.

---

## 14. No conversation vector indexing in MVP

Phase 1 does not vector-index raw conversation history.

Conversation history is available through ConversationStore and recent-window selection.

Future search over old conversations must be designed as a separate feature or content retrieval extension.

---

## 15. Message mutation policy

Phase 1 messages are append-only.

Out of MVP:

```text
message edit
message delete
message redaction lifecycle
conversation branching/forking
```

Rationale:

Message mutation complicates event causality and historical audit.

Privacy delete/purge policy is deferred to a future privacy/retention design.

---

## 16. Conversation title

MVP title policy:

```text
default title = first user message truncated
no LLM-generated title
```

LLM title generation is deferred.

---

## 17. Sensitivity

Context window excludes messages with:

```text
sensitivity=secret
```

This is policy-controlled.

Automatic secret detection is out of MVP, but manual/explicit sensitivity labels must be respected.

---

## 18. Configurability requirement

Windowing and message selection policies must be configurable.

MVP defaults must not be hardcoded into AgentRuntime.

Configuration should support at least:

```yaml
conversation_window:
  max_messages: 12
  max_tokens: 3000
  include_roles: ["user", "assistant"]
  exclude_sensitivity: ["secret"]
  trimming_strategy: drop_oldest_first
```

Future policy variants may be selected by:

```text
loop_strategy
model_profile
conversation_type
active_project_namespace
latency_mode
```

---

## 19. MVP vs deferred

MVP includes:

- durable conversation/message storage;
- `user` and `assistant` messages;
- request/message linkage;
- client idempotency support;
- recent-tail windowing;
- configurable max messages/tokens;
- secret exclusion;
- append-only messages.

Deferred:

- rolling summaries;
- vector indexing conversation history;
- message edit/delete/redaction lifecycle;
- conversation branching;
- LLM-generated titles;
- multimodal messages;
- tool messages in persisted conversation history;
- conversation search;
- export/import.
