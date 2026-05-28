# 06 — Agent Runtime and Loop Architecture

## 1. Purpose

Define the architecture of the agent runtime loop.

Key decision:

> LangGraph is the execution substrate, not the agent architecture.

The assistant's agent loops are defined by our own graph templates, loop strategies, budgets, policy hooks and emitted runtime events.

## 2. Why Not ReAct in Phase 1

ReAct is useful for tool-use loops, but Phase 1 has no tools.

Using ReAct everywhere would add:

- unpredictability;
- unnecessary model calls;
- harder safety;
- risk of self-loop;
- harder tests;
- premature tool/action semantics.

Phase 1 uses deterministic memory-augmented workflow.

## 3. Loop Strategy Concept

Runtime supports the idea of `LoopStrategy`.

Phase 1:

```text
memory_augmented_answer_workflow
```

Future:

```text
direct_answer_workflow
tool_react_loop
planner_executor_loop
sleep_consolidation_loop
approval_gated_loop
structured_extraction_workflow
```

## 4. Phase 1 Workflow

```text
receive_message
  ↓
load_conversation_context
  ↓
retrieve_memories
  ↓
build_prompt
  ↓
call_model_router
  ↓
postprocess_response
  ↓
persist_events
  ↓
stream/return_response
```

Properties:

```text
max_model_calls: 1
max_tool_calls: 0
allow_memory_write: false
allow_cloud: false by default
```

## 5. LoopStrategySelector

Phase 1 selector can be deterministic:

```text
if request.type == "chat":
  memory_augmented_answer_workflow
```

Future selector may use classification, but policy constraints always apply.

```python
class LoopStrategySelector(Protocol):
    async def select(self, request: RuntimeRequest, context: RuntimeContext) -> LoopStrategy: ...
```

## 6. Runtime State

Canonical state should be small and explicit.

```python
class AgentRuntimeState(TypedDict):
    conversation_id: str
    user_id: str
    request_id: str
    user_input: str
    recent_messages: list[Message]
    retrieved_memories: list[MemoryHit]
    selected_loop: str
    model_profile: str
    response_draft: str | None
    final_response: str | None
    policy_decisions: list[PolicyDecision]
    errors: list[RuntimeErrorRecord]
```

Future fields may include tool observations, plan state and task state.

## 7. Future ReAct Loop

ReAct enters only after ToolGatewayPort exists.

Constraints:

```text
max_steps
max_wall_time_sec
max_model_calls
max_tool_calls
allowed_tools
policy_check_per_action
approval_for_high_risk_tools
all actions audited
```

No unrestricted tool loop is allowed.

## 8. Future Planner-Executor

Planner-executor is reserved for:

- multi-step tasks;
- infrastructure operations;
- tasks requiring approval;
- long-running workflows;
- decomposition into subtasks.

## 9. Sleep/Reflection

Sleep/reflection is a bounded workflow, not free-running self-dialogue.

It should:

- select events for period;
- summarize episodes;
- propose memory candidates;
- detect unresolved tasks;
- emit report;
- avoid external effects without approval.

## 10. Runtime Events

All loops emit RuntimeStreamEvents:

```json
{"type": "request.processing.started"}
{"type": "context.assembly.started"}
{"type": "memory.retrieved"}
{"type": "context.assembled"}
{"type": "model.request.created"}
{"type": "token"}
{"type": "model.response.received"}
{"type": "assistant.message.created"}
{"type": "request.processing.completed"}
```

## 11. Testing Requirements

Tests must verify:

- selected loop for chat requests;
- max_model_calls = 1 in Phase 1;
- no tool calls in Phase 1;
- no autonomous memory writes in Phase 1;
- policy decision is recorded for model calls;
- RuntimeStreamEvents emitted in expected order.


## 8. Context Assembly Integration

Phase 1 agent loop does not perform ad-hoc prompt construction.

The graph node responsible for model input construction calls `ContextAssemblerPort`.

```text
receive_message
  -> persist user.message
  -> select_loop_strategy
  -> assemble_context
  -> call_model_router
  -> stream_response
  -> persist assistant.message / events
```

This keeps the agent loop independent from the internal implementation of context management.

The ContextAssembler may internally use ConversationStorePort, MemoryReadPort, PolicyPort and prompt templates, but Agent Runtime sees only the assembled context contract.


## Event trace for Phase 1 loop

The deterministic memory-augmented workflow emits a canonical event chain for every user turn:

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

All events in this chain share the same `request_id`. `causation_id` links the direct predecessor. `correlation_id` may equal `request_id` in Phase 1 and is reserved for longer workflows later.

Agent Runtime is responsible for preserving this causal chain, but it does not own event storage details directly; it emits through `EventLogPort`.


## Runtime budgets and future loop strategies

Phase 1 runtime budget applies only to `memory_augmented_answer`.

It is not a global architecture limit.

Future loop strategies must define their own:

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

Examples:

```text
tool_react_loop
planner_executor_loop
sleep_consolidation_loop
maintenance_workflow
```

Real agent loops must be bounded, auditable and policy-gated.
