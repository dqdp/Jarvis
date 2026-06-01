# ADR-031 — Agent Loop Strategy Architecture

## Status

Accepted, amended by ADR-037 and PM-08l.

PM-08k/ADR-037 supersedes the selector-first production default for
natural-language user input. For normal typed input and future voice transcripts,
`auto`, `chat` and `tools` are policy modes of one bounded agent loop, not a
semantic route classifier that chooses between separate natural-language
strategies before the loop. This ADR still defines the loop-strategy boundary,
the requirement that strategies use ports, and the rule that tool execution goes
through `ToolGatewayPort`.

## Context

The MVP runtime has one deterministic loop:

```text
memory_augmented_answer
```

Its behavior is intentionally simple:

```text
user message
assemble context
one model call
stream assistant answer
persist assistant message
```

Its budget is also intentionally strict:

```text
max_model_calls = 1
max_tool_calls = 0
```

Post-MVP tools cannot be added by inserting tool calls directly into this MVP
loop. A tool-capable assistant needs a different execution algorithm:

```text
assemble context
model proposes action
parse action
policy check
approval if needed
ToolGateway executes
observation returns to loop context
model continues or produces final answer
```

That algorithm is not a tool. It is an agent loop strategy.

ADR-030 defines `ToolGatewayPort` as the boundary for executing tools. This ADR
defines how agent loops should evolve so ToolGateway can be used without
turning `AgentRuntime` into an ad hoc ReAct implementation.

## Decision

Introduce explicit loop strategies behind `AgentRuntime`.

`AgentRuntime` becomes the orchestrator that selects and invokes a loop
strategy. It does not itself contain every loop algorithm.

Initial strategy set:

```text
memory_augmented_answer
```

Planned post-MVP strategies:

```text
tool_react_loop
planner_executor_loop
approval_gated_loop
sleep_consolidation_loop
maintenance_workflow
```

Each loop strategy must declare:

```text
strategy_name
max_steps
max_model_calls
max_tool_calls
max_wall_time_seconds
max_consecutive_failures
allowed_capabilities
policy_hooks
context_assembly_policy
stopping_conditions
failure_semantics
emitted_events
```

The existing `memory_augmented_answer` strategy remains the reliable baseline
and keeps `max_tool_calls=0`.

ADR-035 later defined user-facing `auto` selection. ADR-037 and PM-08l amend
that production default: for natural-language input, `auto` is now a policy mode
of the bounded agent loop. Historical selector/classifier behavior may remain
only for migration, explicit non-default overrides, evaluation fixtures or
non-natural-language workflows that have a separate ADR.

## Architecture shape

Target shape:

```text
AgentRuntime
  -> request plan / policy mode after ADR-037
  -> LoopStrategyRegistry
      -> MemoryAugmentedAnswerLoop
      -> ToolReactLoop / AgentLoop
      -> PlannerExecutorLoop later

LoopStrategy
  -> ContextAssemblerPort
  -> ModelRouterPort
  -> ToolGatewayPort optional
  -> PolicyPort
  -> ConversationStorePort
  -> EventLogPort
```

Rules:

- strategies use ports, not concrete adapters;
- production natural-language `auto/chat/tools` behavior is represented as
  request-plan policy for the bounded agent loop;
- only tool-capable strategies depend on `ToolGatewayPort`;
- `memory_augmented_answer` must not gain hidden tool behavior;
- planner-executor and sleep/reflection remain separate future strategies;
- provider-native tool calling still converts into domain tool proposals and
  executes through ToolGateway.

## Domain concepts

Potential domain objects:

```text
LoopStrategyName
LoopExecutionRequest
LoopExecutionResult
LoopBudget
LoopStep
LoopStepStatus
ToolProposal
ToolObservationRef
LoopFailure
```

`LoopExecutionRequest` should include:

```text
request_id
conversation_id
user_id
user_message
sensitivity
active_project_namespace
strategy_name
budget
permission_mode
correlation_id optional
metadata
```

`LoopExecutionResult` should include:

```text
status
assistant_message optional
final_response optional
steps
used_model_calls
used_tool_calls
context_manifest_refs
error optional
```

## Event model

Introduce loop-level and step-level events.

Initial loop events:

```text
agent.loop.started
agent.loop.completed
agent.loop.failed
agent.loop.cancelled
```

Initial step events:

```text
agent.step.started
agent.step.completed
agent.step.failed
```

Tool-capable strategies also emit ToolGateway events from ADR-030.

The first extraction slice should add only loop-level events. Step-level events
may be added with the first tool-capable loop, where step boundaries are
behaviorally meaningful.

Event linkage:

```text
request_id      one user turn
correlation_id  long-running workflow when applicable
step_id         one loop iteration or plan step
causation_id    prior event that caused this event
```

The current canonical MVP event chain remains valid. Loop/step events may be
added without removing existing request, context, model and message events.

## Tool-capable loop boundary

`tool_react_loop` is the implemented post-MVP Alpha tool-capable loop strategy,
not a tool.

It should:

- call ModelRouter for tool proposals or final answer;
- parse proposals into domain `ToolProposal` objects;
- call ToolGateway for execution;
- feed `ToolObservation` refs back into context;
- stop on budget exhaustion, malformed actions, repeated failures or final
  answer.

It must not:

- execute tools directly;
- bypass ToolGateway policy checks;
- run without max steps/model calls/tool calls;
- persist tool observations as conversation messages by default.

## Context integration

Loop strategies do not manually assemble prompts.

They call `ContextAssemblerPort` with strategy-specific context requirements.

Future ContextAssembler inputs may include:

```text
strategy_name
loop_step
tool_observation_refs
plan_state
approval_state
budget_state
```

This keeps prompt/context evolution behind the existing context boundary.

## Rationale

ToolGateway solves execution. It does not solve decision-making.

The system needs a separate place for the algorithm that decides whether to call
a tool, continue reasoning, ask for approval, stop or answer.

Keeping loop strategies explicit prevents three architectural mistakes:

- turning the MVP loop into a hidden ReAct loop;
- letting AgentRuntime accumulate every future workflow;
- confusing tools with the agent algorithm that uses tools.

## Consequences

Positive:

- MVP loop remains simple and reliable;
- tool-capable behavior can be added without mutating the base loop;
- planner/sleep/maintenance workflows get a consistent extension path;
- budgets and stopping conditions become testable per strategy;
- event model can represent step-level behavior.

Trade-offs:

- adds another internal abstraction;
- requires careful event-chain compatibility;
- strategy selection must be explicit and test-covered;
- ContextAssembler needs strategy-aware inputs over time.

## Alternatives considered

### Add tools directly to memory_augmented_answer

Rejected. This would change MVP semantics and create hidden multi-step behavior
inside a loop whose contract says `max_tool_calls=0`.

### Put ReAct behavior inside ToolGateway

Rejected. ToolGateway executes tools. It does not decide agent reasoning steps.

### Put all loop logic directly in AgentRuntime

Rejected. AgentRuntime would become a large workflow class and future
planner/scheduler/sleep logic would become hard to isolate.

### Introduce LangGraph immediately

Deferred. A graph runtime may be useful later, but the first post-MVP step only
needs explicit strategy boundaries. The design should not require LangGraph to
validate ToolGateway and safe-tool loops.

## Testing requirements

Required tests before implementation:

```text
unit:
  strategy registry selects memory_augmented_answer by default in the PM-03 baseline
  unknown strategy is rejected
  memory_augmented_answer keeps max_tool_calls=0
  tool-capable strategy requires ToolGatewayPort
  budget exhaustion stops strategy

contract:
  LoopStrategy executes through ports only
  memory_augmented_answer preserves existing event chain

architecture:
  AgentRuntime does not import concrete loop adapters with tool implementations
  loop strategies do not import storage/provider/tool adapters directly
  ToolGateway does not import or select loop strategies

e2e:
  MVP user-turn still succeeds with memory_augmented_answer
  fake tool-capable loop can call fake ToolGateway and then produce final answer
  malformed tool proposal fails safely
```

Real tools, shell commands, MCP servers and external integrations are not
required for CI.

## Rollout plan

1. Extract current deterministic workflow into a named
   `MemoryAugmentedAnswerLoop` without changing behavior.
2. Add strategy selection and registry.
3. Add loop/step domain schemas.
4. Add loop-level events while preserving the existing canonical event chain.
5. Defer step-level events until the first tool-capable loop.
6. Add fake tool-capable loop tests.
7. Add `tool_react_loop` only after ToolGateway fake/safe tools are available.
8. Defer planner-executor until a later ADR.

## Deferred

- detailed `tool_react_loop` protocol;
- planner-executor task tables;
- graph checkpointing;
- LangGraph adapter;
- WebSocket/control channel;
- long-running task persistence;
- autonomous scheduling.
