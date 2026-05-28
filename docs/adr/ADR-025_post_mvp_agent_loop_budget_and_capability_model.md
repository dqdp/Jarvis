# ADR-025 — Post-MVP Agent Loop Budget and Capability Model

## Status

Accepted as post-MVP direction.

## Context

Phase 1 defines a single deterministic loop strategy. Future versions will need real agent loops such as bounded ReAct, planner-executor, approval-gated actions and sleep/reflection workflows.

The Phase 1 budget must not be mistaken for a global architecture limitation.

## Decision

Runtime budgets are loop-strategy-specific.

Every future loop strategy must declare:

```text
max_steps
max_model_calls
max_tool_calls
max_wall_time_seconds
allowed capabilities
policy hooks
stopping conditions
failure semantics
emitted events
context assembly policy
```

Future agent loops must remain bounded and policy-gated.

Tool observations are not conversation messages by default.

Long-running autonomous tasks require separate future domain entities such as:

```text
tasks
task_runs
agent_steps
tool_invocations
approvals
```

SSE is Phase 1 transport. Future interactive agent loops may add WebSocket/control channels without changing the event model.

## Rationale

This preserves a small MVP while keeping a clear path to real agentic behavior.

Explicit per-loop budgets prevent free-running agents and uncontrolled tool use.

Separate task/step entities prevent `assistant_requests` from becoming an overloaded generic workflow table.

## Consequences

Positive:

- real agent loops remain possible;
- Phase 1 remains bounded;
- future ReAct/planner/sleep workflows have clear design obligations;
- policy/event/context boundaries remain valid.

Trade-offs:

- true agentic workflows are deferred;
- future phases require additional task/tool/approval models;
- WebSocket/control channels may be needed later.

## Deferred

- ToolGatewayPort implementation;
- bounded ReAct loop;
- planner-executor task model;
- approval system;
- scheduler/event bus;
- sleep/reflection implementation;
- step-level event model;
- WebSocket/control channel.
