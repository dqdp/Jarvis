# ADR-024 — Error Handling and Runtime Budgets

## Status

Accepted.

## Context

Phase 1 uses a deterministic `memory_augmented_answer` loop. It must remain bounded, reliable and simple.

The system must define failure behavior and runtime limits without prematurely implementing a full workflow engine.

## Decision

Phase 1 defines runtime budgets for `memory_augmented_answer` only.

Default budget:

```text
max_model_calls=1
max_tool_calls=0
allow_cloud=false
allow_tools=false
allow_autonomous_memory_write=false
```

Budgets are loop-strategy-specific configurable defaults, not global architecture limits.

EventLog write failure for an accepted request is fatal.

Memory retrieval failure is non-fatal for normal chat and produces degraded context.

Embedding failure during memory write creates the memory with `indexing_status=embedding_failed`.

`local_main` model failure is fatal for the request.

Chat retry is 0.

Structured invalid output gets one validation retry.

Embedding gets one retry.

Partial streamed output is not persisted in MVP.

No assistant message is created on system failure by default.

Token-limit finish reason is not system failure.

Error payloads must be sanitized and must not contain raw prompts or secrets.

## Rationale

The system needs predictable MVP behavior and clear safety boundaries.

Failing hard when historical truth or safety policy is at risk preserves auditability.

Degrading when long-term memory retrieval fails improves availability for normal chat.

Avoiding chat retries prevents duplicate partial output and complex stream repair.

## Consequences

Positive:

- predictable runtime behavior;
- bounded MVP loop;
- clear degraded vs fatal semantics;
- no hidden agentic retries;
- audit remains trustworthy.

Trade-offs:

- no automatic recovery from model outages;
- no partial response persistence;
- no request resume/retry;
- user sees failed request rather than assistant error message by default.

## Deferred

- request cancellation implementation;
- partial assistant message persistence;
- stream resume;
- workflow compensation;
- dead letter queue;
- background retry workers;
- circuit breakers;
- provider failover;
- user-visible assistant error messages.
