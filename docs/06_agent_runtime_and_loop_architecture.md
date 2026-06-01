# 06 — Agent Runtime and Loop Architecture

## 1. Purpose

Define the architecture of the agent runtime loop.

Key decision:

> MVP runtime uses a custom deterministic workflow; LangGraph is deferred.

The assistant's agent loops are defined by our own workflow templates, loop strategies, budgets, policy hooks and emitted runtime events.

Current baseline: post-MVP Alpha. The original MVP loop remains
`memory_augmented_answer`; PM-03 added bounded `tool_react_loop` after
ToolGateway and the loop-strategy boundary were in place.

## 2. Why Not ReAct in Original Phase 1

ReAct is useful for tool-use loops, but original Phase 1 had no tools.

Using ReAct everywhere would add:

- unpredictability;
- unnecessary model calls;
- harder safety;
- risk of self-loop;
- harder tests;
- premature tool/action semantics.

Original Phase 1 uses deterministic memory-augmented workflow. Current
post-MVP Alpha adds `tool_react_loop` as the bounded agent-loop implementation
vehicle. PM-08k supersedes selector-first production routing, and PM-08l hardens
that loop with explicit states, a single final-answer path and typed observation
recovery before PM-09 voice work starts.

## 3. Loop Strategy Concept

Runtime supports the idea of `LoopStrategy`.

Phase 1:

```text
memory_augmented_answer_workflow
```

Current Alpha and future:

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
assemble_context via ContextAssemblerPort
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

## 5. Bounded Agent Loop Policy Modes

auto, chat and tools are policy modes of one bounded agent loop, not a semantic
selector that chooses between separate natural-language strategies before the
loop. `ToolReactLoop` remains the public implementation vehicle during PM-08l,
but ToolReactLoop decomposition is now required: state transitions, event
recording, final answer generation and observation recovery belong in explicit
runtime components such as `FinalAnswerStep` and `ToolObservationRecoveryPolicy`.

User-facing modes:

```text
auto  -> bounded agent loop with normal tool policy
chat  -> bounded agent loop with tool use disabled
tools -> bounded agent loop with tool use available/required by request policy
```

The loop boundary lives on the server side. CLI, API, future voice clients and
future integrations must not implement their own safety-critical routing logic.

PM-08a through PM-08h introduced loop selection, classifier and corpus
infrastructure. PM-08k supersedes classifier-first routing for the production
natural-language path: the runtime must not call a separate route classifier
before the bounded agent loop. Historical classifier fixtures may remain as
evaluation evidence, but they are not authorization and not a voice prerequisite.

Expected PM-08k/PM-08l behavior:

```text
ordinary chat or explanation
  -> bounded agent loop answers without tool use unless useful and allowed

project docs question
  -> bounded agent loop may use retrieval/context assembly

live project inspection
  -> bounded agent loop may propose a tool
  -> proposal is validated by schema, policy, approval and ToolGateway

live system diagnostics
  -> bounded agent loop may propose a diagnostics tool
  -> proposal is validated by schema, policy, approval and ToolGateway

tools disabled for a live-state request
  -> fail clearly, ask for permission or explain that live inspection is
     unavailable; do not invent live facts
```

Model-origin tool proposals are not execution authorization. Policy constraints
always apply after the model proposes an action.

```python
class AgentLoopPort(Protocol):
    async def run(self, request: AgentLoopRequest) -> AgentLoopResult: ...
```

Loop decisions and tool proposals are auditable and redacted. They record public
phase, selected mode, proposed tool, validation outcome and policy outcome
without logging the raw full prompt.

The request model must distinguish:

```text
requested_mode          user-facing mode: auto/chat/tools
tool_policy             disabled/available/required
agent_phase             selecting/retrieving/tool_running/streaming/etc.
proposed_tool           model-origin tool proposal, not authorization
proposal_status         accepted/rejected/approval_required/unavailable
policy_outcome          policy/approval result
```

Fallback behavior is part of the agent-loop contract. Unclear or unsupported
live-state requests must not silently become hallucinated ordinary chat.

## 6. Runtime State

Canonical state should be small and explicit.

```python
class AgentRuntimeState(TypedDict):
    conversation_id: str
    user_id: str
    request_id: str
    user_input: str
    requested_mode: str
    tool_policy: str
    agent_phase: str
    proposed_tool: str | None
    proposal_status: str | None
    model_profile: str
    response_draft: str | None
    final_response: str | None
    policy_decisions: list[PolicyDecision]
    tool_observation_refs: list[ToolObservationRef]
    errors: list[RuntimeErrorRecord]
```

Future fields may include plan state and task state, but future planner steps
must still execute through the same bounded loop, PolicyPort and ToolGatewayPort
path.

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

- `auto`, `chat` and `tools` enter the bounded agent loop as policy modes;
- chat mode disables tool calls while keeping the same request lifecycle;
- auto mode can answer ordinary chat and project-docs questions without a tool observation;
- tools mode requires a valid completed observation when tools are available and allowed;
- max_model_calls = 1 in the original Phase 1 `memory_augmented_answer` loop;
- original MVP loop does not make tool calls;
- `tool_react_loop` uses ToolGatewayPort and explicit budgets;
- RAG questions use ContextAssembler/content retrieval and do not require a tool observation by default;
- live-state requests do not silently hallucinate when tools are disabled or unavailable;
- no autonomous memory writes in Phase 1;
- policy decision is recorded for model calls;
- request-plan and loop events are recorded without raw full prompts;
- RuntimeStreamEvents emitted in expected order.


## 12. Context Assembly Integration

Phase 1 agent loop does not perform ad-hoc prompt construction.

The runtime step responsible for model input construction calls `ContextAssemblerPort`.

```text
receive_message
  -> persist user.message
  -> build agent request plan from auto/chat/tools policy mode
  -> enter bounded agent loop
     -> context_assembling step calls ContextAssemblerPort
     -> proposing/finalizing steps call ModelRouterPort
     -> optional tool steps execute only through ToolGatewayPort
  -> stream_response
  -> persist assistant.message / events
```

This keeps the agent loop independent from the internal implementation of context management.

The ContextAssembler may internally use ConversationStorePort, MemoryReadPort, PolicyPort and prompt templates, but Agent Runtime sees only the assembled context contract.


## 13. Event trace for Phase 1 loop

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


## 14. Runtime budgets and future loop strategies

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
