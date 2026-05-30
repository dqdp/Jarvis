# ADR-006: Custom deterministic workflow for MVP runtime

Status: Accepted

## Context

Phase 1 needs a deterministic, testable user-turn runtime with durable request
status, event causation, SSE streaming and strict ports/adapters boundaries.
The current dependency set and MVP scope do not require graph checkpoint
replay, tool loops or autonomous multi-step orchestration.

## Decision

Use a custom deterministic workflow for the MVP runtime:

```text
request.processing.started
context.assembly.started
memory.retrieved
context.assembled
model.request.created
model.response.received
assistant.message.created
request.processing.completed
```

LangGraph is deferred until the runtime needs graph-native branching,
checkpoint replay or post-MVP tool/planner workflows. Runtime code must still
stay behind ports so a later LangGraph adapter can replace the in-process
workflow without changing storage, API or context contracts.

## Consequences

The MVP has fewer moving parts and simpler TDD coverage. PostgreSQL remains the
system of record for conversations, events, memory and model invocations, while
LangGraph checkpoint tables are not required for MVP readiness.

The cost is that post-MVP graph migration needs a focused adapter slice and
updated contract tests before enabling checkpoint-backed continuation.

## Post-MVP revisit

As of PM-08 planning, the original deferral condition is now approaching:

```text
tool-capable loops exist;
approvals exist;
Project Docs RAG exists;
automatic loop selection is planned;
planner-executor, code sandbox, sleep/reflection and durable workflows are next.
```

This does not change the MVP decision. It means LangGraph should be evaluated
through a dedicated adapter/gate slice before implementing planner-executor,
durable code sandbox workflows or long-running background workflows.

Rules for the revisit:

- LangGraph must be introduced behind existing Jarvis ports, not as a direct
  dependency of API, CLI, storage adapters or tool adapters.
- LangGraph checkpoint state must not silently replace Jarvis conversation,
  request, event, approval or memory tables.
- LangGraph interrupts/checkpoints must be mapped explicitly to Jarvis approval,
  request status, SSE and event-log semantics.
- Adoption requires contract and architecture tests proving that existing custom
  loops still work and that graph-backed loops do not bypass policy or
  ToolGateway.
