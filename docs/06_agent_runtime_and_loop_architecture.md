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

live-state question such as current local time, daemon status or host diagnostics
  -> bounded agent loop must collect relevant tool evidence before any answer
     that asserts a current live fact, when an allowed local tool can observe
     that state
  -> ordinary final_answer without observation is not valid for that claim

current available local tools question
  -> bounded agent loop may answer deterministically from the current
     ToolRequestPlan only
  -> do not use RAG, a global registry or hidden/disabled tools as the source

live project inspection
  -> bounded agent loop may propose a tool
  -> proposal is validated by schema, policy, approval and ToolGateway

live system diagnostics
  -> bounded agent loop may propose a diagnostics tool
  -> proposal is validated by schema, policy, approval and ToolGateway

tools disabled or no allowed local tool for a live-state request
  -> current implementation does not provide a hard evidence gate
  -> future unavailable/clarification behavior must be added before docs claim
     this path is guarded
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
live-state requests must not silently become hallucinated ordinary chat. A
request may fail clearly, ask for permission or say that live evidence is
unavailable, but it must not assert current state without a matching completed
observation when an allowed local tool can observe that state.

Live-state evidence guards and deterministic final answers are separate
responsibilities. The guard is deliberately broad and safety-oriented; the
deterministic finalizers are deliberately narrow and source-backed. The guard
must not become a classifier, route resolver or direct natural-language tool
execution path.

The live-state evidence guard algorithm is:

1. Build the candidate live-state tool set from explicit request metadata and
   allowed local tools. Initial live-state/calendar evidence tools include
   `datetime.now`, `calendar.diff`, `datetime.diff`, `datetime.until`, `daemon.status`,
   `tool.system.read.resources`, `tool.system.read.network`, `tool.system.read.hardware`,
   `tool.system.read.sensors` and `tool.system.read.process`. A new tool with
   a similar prefix is not live-state evidence until it has an explicit typed
   provenance mapping.
2. If a relevant local tool is allowed, do not silently downgrade the request
   to hallucinated ordinary chat. The loop may collect evidence, answer that
   live evidence is unavailable after a failed/denied/unavailable observation,
   ask for permission/clarification or fail with a typed, user-actionable
   reason.
3. Lightly normalize user text for matching: lowercase, trim edge punctuation
   and collapse whitespace. Do not use this step to infer a semantic route.
4. Match broad live-state intent families, not exact deterministic-answer
   phrases. Required families include:

   - current time/date wording, including "what time is it", "current time",
     "local time", `сколько времени`, `который час`,
     `текущее время`, `в данный момент` and `сейчас`;
   - current local machine, process or daemon state, including CPU/processor,
     memory/RAM, load/usage, battery, network/VPN/IP, disk, hardware,
     process/service/PID and daemon/status wording;
   - current live values combined with arithmetic, threshold, comparison or
     derived numeric wording, such as requests that calculate from the current
     time, a countdown value or current resource values.
   - timestamp/calendar interval wording, such as "how many hours between",
     `количество микросекунд между`, `seconds since` or `недель прошло с`.

5. Return typed guard metadata, not only a boolean. The metadata should identify
   the live-state family, whether completed evidence is required and the
   candidate tool names that may satisfy the evidence requirement. This metadata
   is a finalization guard only; it must not directly select, execute or
   authorize a tool.
6. When a positive live-state intent match has a relevant allowed local tool,
   the loop must not accept a `final_answer` that asserts current live state
   before a matching completed tool observation exists. Process
   existence/status evidence must match the requested process name or PID;
   process resource claims such as per-process CPU or memory require a typed
   process-resource observation such as `system.process_resource_snapshot`, not
   a global machine resource snapshot.
7. Live-derived numeric transformations must not be performed in model prose.
   If the requested answer is not a direct typed field from completed live
   evidence, the loop requires a completed `calculator.evaluate` observation
   whose expression is grounded in request-relevant typed numeric fields from
   completed live observations plus explicit constants from the user request.
   Operation-implied structural constants, such as the denominator for an
   average over covered live operand groups, are allowed only when they are
   derived from those covered operand groups.
   Timestamp/calendar interval requests use `calendar.diff` after the model
   supplies explicit timezone-aware ISO endpoints. Fixed-duration fallback
   requests may use `datetime.diff` for microseconds, milliseconds, seconds,
   minutes, hours, days and weeks. Calendar-aware units such as months,
   quarters and decades require `calendar.diff`. If an endpoint is the current
   moment, the loop also requires a completed `datetime.now` observation and
   the diff endpoint must match that observation. Relative named events such as
   "last Thanksgiving" require a separate typed event-date evidence source
   before their timestamps can satisfy the guard; `calendar.diff` and
   `datetime.diff` compute intervals only and do not prove that a
   model-supplied timestamp is the true date of an external event.
   The expression's operation family must match the requested transform; extra
   operations are not valid provenance. The calculator grammar, not a
   per-operation finalizer, determines which bounded mathematical operations are
   supported.
   This rule is intended to stay tool-agnostic, but new tools such as web search
   or code sandboxes may satisfy it only after an explicit typed provenance
   extension defines their live/current-state schemas and request-relevant
   source fields. Until that extension exists for a tool, a prose search
   snippet, code output or model claim is not evidence for current state.
8. After a completed observation exists, finalization follows the normal bounded
   loop. A narrow deterministic finalizer may synthesize a short answer only
   from completed typed evidence, for example formatting a completed
   `datetime.now` observation as the current local time. More complex live-state
   answers use the final-answer model with the observation in context.
9. Location or scope wording must not disable the evidence guard. For example,
   `сколько времени в данный момент?` still requires `datetime.now` evidence.
   A request such as `сколько времени в Париже?` is still live-state, but it is
   not eligible for the local-time deterministic finalizer unless an appropriate
   timezone/world-clock observation is available.

The available-tools deterministic finalizer is runtime metadata, not
retrieval. It may answer only a single-intent question about tools available to
this assistant for the current request. Its source of truth is
`ToolRequestPlan.allowed_tool_names` plus the matching
`allowed_tool_summaries`; disabled, hidden or globally registered tools must not
be disclosed. Architecture, implementation and external catalog questions such
as Python or AWS tools remain ordinary model/RAG-capable questions. Compound
requests remain in the bounded loop so live-state evidence guards still apply.

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
- auto mode does not finalize live-state claims without relevant completed
  tool evidence when an allowed local tool can supply that evidence;
- tools mode requires a valid completed observation when tools are available and allowed;
- max_model_calls = 1 in the original Phase 1 `memory_augmented_answer` loop;
- original MVP loop does not make tool calls;
- `tool_react_loop` uses ToolGatewayPort and explicit budgets;
- RAG questions use ContextAssembler/content retrieval and do not require a tool observation by default;
- failed, denied or unavailable live-state observations do not silently
  hallucinate live facts;
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
