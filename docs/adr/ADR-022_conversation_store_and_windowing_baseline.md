# ADR-022 — Conversation Store and Windowing Baseline

## Status

Accepted.

## Context

Phase 1 needs durable conversation history and a way to provide recent dialogue context to the model.

This must not be confused with long-term memory, event log, or full conversation search.

The system must also avoid cementing early windowing heuristics as permanent architecture.

## Decision

Phase 1 uses `ConversationStorePort` for durable conversations and messages.

Conversation Store stores full message history as an operational read/write model.

Event Log remains the immutable historical truth.

Phase 1 persisted message roles:

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

System/runtime prompt messages are not stored as conversation messages.

Phase 1 message content is text-only; typed content parts are deferred.

Context windowing belongs to `ContextAssembler`, not `ConversationStore`.

Default Phase 1 windowing policy:

```yaml
max_messages: 12
max_tokens: 3000
include_roles: ["user", "assistant"]
exclude_sensitivity: ["secret"]
trimming_strategy: drop_oldest_first
```

These values are configurable policy defaults, not hardcoded architecture.

Phase 1 does not include:

```text
automatic rolling summaries
conversation vector indexing
message edit/delete/redaction lifecycle
conversation branching
LLM-generated titles
```

## Rationale

This keeps the MVP small and predictable.

ConversationStore remains a simple durable operational model.

ContextAssembler owns context policy, making future advanced windowing strategies possible without changing AgentRuntime or storage contracts.

## Consequences

Positive:

- simple implementation;
- durable conversation history;
- clear separation from EventLog and Memory;
- configurable windowing;
- future advanced context strategies remain possible.

Trade-offs:

- long conversations are handled only by recent tail in MVP;
- no conversation search;
- no automatic summaries;
- no edit/delete lifecycle.

## Deferred

- rolling summaries;
- summary + recent tail;
- salience-based trimming;
- tool-aware windows;
- planner-specific windows;
- conversation search;
- multimodal message model;
- message mutation/redaction lifecycle.
