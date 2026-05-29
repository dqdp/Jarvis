# 24 — Post-MVP Agent Loop Follow-ups

## 1. Purpose

This document records post-MVP follow-ups needed to support real agent loops beyond the Phase 1 deterministic `memory_augmented_answer` workflow.

Phase 1 budgets and error rules are not global architecture limits. They are the defaults for one loop strategy.

---

## 2. Future loop strategies

Post-MVP loop strategies may include:

```text
tool_react_loop
planner_executor_loop
approval_gated_loop
sleep_consolidation_loop
maintenance_workflow
```

Each loop strategy must explicitly declare:

```text
max_steps
max_model_calls
max_tool_calls
max_wall_time_seconds
max_consecutive_failures
allowed capabilities
policy hooks
stopping conditions
failure semantics
emitted events
context assembly policy
```

---

## 3. Bounded ReAct loop

Future `tool_react_loop` should be bounded and policy-gated.

`tool_react_loop` is a loop strategy, not a tool. It may use tools through
`ToolGatewayPort`, but the reasoning/control algorithm must remain separate
from tool execution.

Example budget:

```yaml
tool_react_loop:
  max_steps: 8
  max_model_calls: 8
  max_tool_calls: 5
  max_consecutive_tool_failures: 2
  max_wall_time_seconds: 300
  allow_memory_write: false
  allow_memory_candidates: true
```

Flow:

```text
assemble_context
model proposes action
parse action
policy check
approval if needed
ToolGateway call
record observation
continue or final answer
```

ReAct must not be free-running.

---

## 4. Planner-executor loop

Future `planner_executor_loop` should support:

```text
plan creation
plan validation
policy check
approval if required
step execution
observation
plan update
finalization
```

Likely future tables:

```text
tasks
task_runs
agent_steps
plans
plan_steps
tool_invocations
approvals
```

`assistant_requests` must not become the universal table for all long-running tasks.

---

## 5. Sleep/reflection loop

Future sleep/reflection must be a bounded workflow, not free-running self-dialogue.

Example budget:

```yaml
sleep_consolidation_loop:
  max_model_calls: 10
  max_tool_calls: 0
  max_wall_time_seconds: 900
  allow_direct_memory_write: false
  allow_memory_candidates: true
```

Expected flow:

```text
select events for period
summarize episodes
propose memory_candidates
detect stale memories
propose archive/supersede actions
write sleep report
```

Direct memory mutation should be avoided unless explicitly approved by policy.

---

## 6. ToolGatewayPort

Real agent loops require `ToolGatewayPort`.

Model output may propose tool calls, but runtime must own execution.

Flow:

```text
model proposes tool call
PolicyPort evaluates
approval if required
ToolGateway executes
EventLog records
observation returned to loop
```

Future tool events:

```text
tool.call.requested
tool.call.approved
tool.call.denied
tool.call.started
tool.call.completed
tool.call.failed
tool.observation.recorded
```

---

## 7. Step-level events

Future agent loops require step-level events:

```text
agent.step.started
agent.step.completed
agent.step.failed
```

Correlation model:

```text
request_id:
  one user interaction

correlation_id:
  long-running workflow / task

step_id:
  one agent iteration
```

---

## 8. Observations

Tool observations are not conversation messages by default.

They should be represented as:

```text
event log records
runtime state for current loop
future content/artifact references for large outputs
```

Only user-relevant summaries should become assistant messages.

---

## 9. Approval model

Future agent loops need approval events:

```text
approval.required
approval.granted
approval.denied
approval.expired
```

Approvals may require WebSocket or separate HTTP endpoints plus SSE.

---

## 10. Transport follow-ups

SSE is sufficient for Phase 1.

Future interactive agent loops may need:

```text
WebSocket
approval endpoints
pause/resume endpoints
cancel task
inject user clarification
change goal
```

This must be added without changing the core runtime/event model.

---

## 11. Scheduler / event bus

Phase 1 has no mandatory Redis/NATS.

Long-running workflows may require:

```text
SchedulerPort
EventPublisherPort
NATS JetStream or equivalent
durable background workers
```

This is post-MVP.

---

## 12. Follow-up ADRs

Future ADRs should define:

```text
ToolGateway boundary and security model
agent loop strategy architecture
bounded ReAct loop
planner-executor task model
approval model
scheduler/event bus
sleep/reflection workflow
step-level event model
WebSocket/control channel
```
