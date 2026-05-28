# 07 — Core Daemon Design

## 1. Process Model

Phase 1 runs primarily as one assistant-api process:

```text
assistant-api:
  FastAPI
  AgentRuntime
  ModelRouter
  Memory adapters
  Storage adapters
  Policy module
```

External processes:

```text
postgres
local-inference-node
```

## 2. Package Layout

```text
assistant_core/
  api/
  runtime/
  conversations/
  events/
  memory/
  models/
  policy/
  storage/
  streaming/
  observability/
```

## 3. Request Flow

```text
POST /v1/conversations/{id}/messages
  → validate request
  → append user.message
  → emit conversation.started
  → AgentRuntime executes selected LoopStrategy
  → retrieve memory
  → call ModelRouter
  → stream tokens/events
  → append assistant.message
  → append audit events
```

## 4. Failure Principles

- Persist user message before model call.
- Log model failures as events.
- Return structured error events in stream.
- Never silently fall back to cloud.
- Never write autonomous memory on failure recovery.

## 5. Restart Semantics

After restart:

- conversations/messages available from PostgreSQL;
- events available from event log;
- memory available from memory tables;
- checkpoints may allow continuation/debugging;
- in-flight request may need retry by user/API client.


## Testing and TDD

Core daemon implementation must be TDD-first.

No production component should be implemented without corresponding unit, contract, integration, golden, architecture or e2e tests as appropriate.

Coding agents must not bypass ports/adapters boundaries to make tests pass.
