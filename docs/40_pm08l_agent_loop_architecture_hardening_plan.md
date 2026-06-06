# 40 — PM-08l Agent Loop Architecture Hardening Plan

PM-08l is the final pre-PM09 architecture-hardening slice. It is not a voice
implementation slice and it must not add new tools, model providers or a new
semantic router.

The purpose of PM-08l is to make the post-PM-08k request path structurally ready
for voice:

```text
natural-language input or future voice transcript
  -> one bounded agent loop
  -> explicit step state machine
  -> optional tool action
  -> typed observation
  -> recovery, clarification or final answer
```

## Problem Statement

PM-08k moved production natural-language handling in the right direction: normal
typed input enters the bounded agent loop, and classifier-first route
adjudication no longer owns the default request path.

However, the current `ToolReactLoop` remains a transitional implementation. It
still has architectural pressure points that make PM-09 risky:

- the loop is a large orchestration method that owns state transitions, event
  recording, context assembly, structured proposals, fallback behavior, final
  answer generation, tool execution and failure handling;
- final-answer generation is reachable through multiple branches instead of one
  finalization path;
- `auto` mode can still depend on a structured tool-proposal call before an
  ordinary answer, which makes simple chat vulnerable to proposal-format
  failures;
- failed, denied, unavailable and malformed tool outcomes are not modeled as a
  clear recovery policy;
- live chat has already exposed this failure shape:
  - "загрузка цп" ended as `tool loop failed (tool_observation_failed)`;
  - "где раки зимуют?" ended as `tool loop failed (tool_observation_failed)`;
  - "Двадцать два в третьей степени." ended as
    `tool loop failed (tool_observation_failed)`;
  these cases must become regression tests for auto-mode fallback, typed
  unavailable/failed observations and final-answer recovery;
- streaming currently exposes lifecycle events and then emits the final answer
  as a terminal text delta, which is acceptable for a transitional CLI but needs
  an explicit voice-readiness contract.

PM-08l hardens the architecture without changing the PM-08k direction.

ReAct is a design influence, not the Jarvis runtime protocol. PM-08l must not
introduce free-form `Thought/Action/Observation` transcript parsing. The target
is a bounded typed agent loop with explicit states, typed tool proposals, typed
observations, policy gates and durable request lifecycle events.

## Target Contract

The production request path must satisfy these rules:

- `auto`, `chat` and `tools` are policy modes for the same bounded agent loop.
- `chat` disables tool calls but still uses the same request lifecycle.
- `auto` allows tools when policy and budget allow. Ordinary chat may answer
  without a tool observation, but live-state claims must not finalize without
  relevant completed tool evidence when an allowed local tool can observe that
  state.
- `tools` requires at least one valid tool observation before the final answer
  when tools are available and allowed.
- Tool choice belongs inside the bounded agent loop. Tool execution belongs
  behind schema validation, allowlists, `PolicyPort`, approvals and
  `ToolGatewayPort`.
- Malformed explicit `tool_call` proposals fail closed.
- Malformed non-tool proposal output in `auto` may fall back to a normal final
  chat path when that does not violate `tools` mode or budgets.
- Denied, unavailable and failed tool outcomes must become typed observations
  that drive deterministic recovery, clarification or controlled failure. They
  must not collapse into only a generic `tool loop failed` surface.
- Budget exhaustion after useful completed observations should finalize through
  the same final-answer path when safe; budget exhaustion before required
  observations must fail closed or clarify.
- Voice must submit transcripts through the same request lifecycle and loop. It
  must not add a voice-specific classifier, route resolver or tool bypass.

Live-state intent detection is a broad evidence requirement, not a direct
execution route. It must follow the full guard algorithm in
`docs/06_agent_runtime_and_loop_architecture.md`: derive candidate live-state
tools from explicit metadata and allowed local tools, lightly normalize user
text, match broad live-state intent families, return typed guard metadata, block
unevidenced live-state finalization, and finalize only after matching completed
evidence. It must cover current time/date wording, local machine or daemon
state wording, current live values combined with arithmetic, threshold or
comparison wording, unavailable-tool behavior and location/scope wording.

Current available-tools questions are answered from runtime request metadata,
not retrieval or a global registry. The deterministic finalizer may expose only
`ToolRequestPlan.allowed_tool_names` and matching safe summaries for the
current request. It must reject architecture/docs questions, external tool
catalogs and compound requests back to the ordinary bounded loop.

## Architecture Shape

`ToolReactLoop` may remain the public class name during PM-08l, but it should
stop being the place where every responsibility accumulates.

Target internal components:

```text
ToolReactLoop / AgentLoop orchestrator
  AgentLoopState
  AgentLoopStep
  ToolProposalStep
  FinalAnswerStep
  ToolObservationRecoveryPolicy
  LoopFailurePolicy
  LoopEventRecorder
```

The implementation should avoid a large rename before behavior is green. A later
mechanical rename from `ToolReactLoop` to `AgentLoop` is allowed only after the
contracts and tests are stable.

## Plan-then-Execute Compatibility

The bounded typed agent loop is the execution primitive for a user turn and for
future executable plan steps. It is not the final architecture for every
long-horizon task.

Simple requests must not require a planner:

```text
simple request
  -> bounded typed agent loop
  -> final answer
```

Compound tasks may later use a plan-and-execute shell around the same bounded
executor:

```text
compound request
  -> planner produces typed plan
  -> execute step 1 via bounded typed agent loop
  -> typed observation
  -> execute step 2 via bounded typed agent loop
  -> typed observation
  -> replan, stop or final answer
```

PM-08l must preserve this future shape without implementing the full planner.
The loop state machine, events and context inputs should be compatible with a
future scoped step request:

```text
PlanStepExecutionRequest:
  parent_request_id
  plan_id
  step_id
  step_goal
  allowed_tools
  budget
  approval_policy
```

This is a compatibility target, not a PM-08l production API. PM-08l must not add
a planner-executor, hidden plan classifier or planner-specific natural-language
router. A future planner must call the same bounded loop and the same
PolicyPort/ToolGatewayPort path used by normal user turns.

Canonical state taxonomy:

```text
idle                no active request in the shell/client
request_started     request accepted and loop is about to run
context_assembling  ContextAssembler is building proposal/final context
proposing           model is deciding final_answer vs tool_call
tool_validating     schema, allowlist, policy, approval and budget gates run
waiting_approval    user approval is required before execution can continue
tool_running        ToolGateway invocation is active
observing           typed tool observation is recorded for the loop
finalizing          final context and chat answer are being produced
completed           terminal success
failed              terminal controlled failure
cancelled           terminal user/system cancellation
```

Mode and budget matrix:

```text
chat:
  tools are disabled for the request
  final answer may be produced without observations
  proposed tool_call fails closed as tool_policy_disabled

auto:
  tools may be used when allowed and budgeted
  ordinary final answers may be produced without observations
  live-state claims require relevant completed tool evidence when an allowed
    local tool can observe that state
  malformed non-tool proposal may fall back to final chat
  malformed explicit tool_call fails closed
  budget exhausted after useful completed observations may finalize

tools:
  at least one valid completed tool observation is required before final answer
    when tools are available and allowed
  no allowed tools, tools_enabled=false or max_tool_calls=0 fails closed before
    final answer
  policy denial before the first valid observation fails closed or asks
    clarification
  APPROVAL_REQUIRED is a nonterminal waiting_approval state, not a completed
    observation
```

## Milestone Workflow

Each milestone must use the standard TDD workflow and have its own gate:

```text
1. Write or update tests first.
2. Confirm the red failure is meaningful.
3. Implement the smallest production change.
4. Run the relevant verification for the milestone.
5. Start two read-only review agents from scratch after tests are green.
6. Fix relevant P0/P1 findings, then repeat verification and review.
7. Commit the milestone only when there are no relevant P0/P1 findings.
```

Review-agent prompts must say:

```text
Tests are already green.
Do not run tests.
Do not edit files.
Perform read-only review only.
Focus on agent-loop architecture, contracts, regressions, security/privacy,
operability, missing integration/e2e coverage and facade isolation.
Report P0/P1/P2/P3 with file/line references.
```

PM-08l uses five commit gates:

```text
Milestone 0: contract freeze
Milestone 1: state machine and final answer unification
Milestone 2: auto/tools semantics and observation recovery
Milestone 3: streaming and event semantics
Milestone 4: final pre-PM09 gate
```

## Milestone 0 — Contract Freeze

Goal: align documentation and architecture guardrails around the PM-08l contract
before production refactoring.

Tests/verification:

- documentation acceptance proves PM-09 depends on PM-08l;
- architecture tests prove no classifier/request-resolver/direct-tool path is
  reintroduced.

Implementation:

- amend ADR-031 so accepted ADR history cannot be read as permission to restore
  selector-first production natural-language routing;
- update PM-08l plan references in roadmap/status docs;
- replace old generic PM-04-era budget-exhaustion wording with the precise
  PM-08l budget matrix;
- document that `ToolReactLoop decomposition` is now in scope for PM-08l.

Gate:

```text
make test-architecture
```

## Milestone 1 — State Machine and Final Answer Unification

Goal: make loop transitions explicit and route every final answer through one
shared finalization path.

Tests first:

- step order is observable as `started -> proposal/final/tool -> observation ->
  final -> completed`;
- canonical loop state names match the PM-08l state taxonomy;
- step failures pass through `LoopFailurePolicy`;
- the loop does not complete or fail outside the event recorder/failure policy;
- tools disabled uses the same finalizer;
- structured `final_answer` proposal uses the same finalizer;
- tool-budget exhausted after completed observations uses the same finalizer;
- malformed non-tool fallback in `auto` uses the same finalizer;
- finalizer performs context assembly, chat model call, assistant message
  creation and terminal request events exactly once.

Implementation:

- introduce `AgentLoopState` and `AgentLoopStep` domain/runtime-internal types;
- extract event recording into `LoopEventRecorder`;
- introduce `FinalAnswerStep`;
- remove duplicated final-chat branches from the loop;
- keep `ToolReactLoop` as the thin orchestrator;
- keep final answers on the main chat model/profile.

Gate:

```text
make test-unit
make test-e2e
make test-golden
make test-architecture
```

## Milestone 2 — Auto/Tools Semantics and Observation Recovery

Goal: make ordinary chat resilient when tools are merely available and stop
treating every non-completed observation as the same generic loop failure.

Tests first:

- an ordinary chat request with tools available completes without requiring a
  valid tool call;
- a non-tool factual/language question such as "где раки зимуют?" must not fail
  only because an available tool proposal/execution path fails;
- natural-language arithmetic such as "Двадцать два в третьей степени" either
  uses an allowed calculator tool and finalizes or falls back/clarifies through
  the final-answer path; it must not surface as generic
  `tool_observation_failed`;
- a time/tool request can use a tool and then finalize;
- malformed explicit `tool_call` fails closed;
- `tools` mode does not silently fall back to direct final answer before a valid
  observation;
- `tools` mode with no allowed tools fails closed;
- `tools` mode with `tools_enabled=false` fails closed;
- `tools` mode with `max_tool_calls=0` fails closed;
- policy denial before the first valid observation fails closed or asks
  clarification;
- `APPROVAL_REQUIRED` enters `waiting_approval` and does not count as a completed
  observation;
- denied tool observations produce a controlled terminal state or clarification
  path with a typed reason;
- unavailable tool observations preserve the tool name, request id and safe
  diagnostic reason;
- failed tool observations obey retry/failure budgets;
- failed CPU/system-diagnostics observations for "загрузка цп" do not become only
  generic `tool loop failed`; they either recover with typed partial/unavailable
  evidence, ask clarification or fail with a user-actionable typed reason;
- tool failures for requests that can be answered without that tool return to the
  final-answer/recovery policy instead of forcing terminal loop failure;
- approval denied/cancelled/expired leaves the request in a controlled state;
- observation refs enter final context only through `ContextAssembler`.

Implementation:

- make proposal parsing and fallback policy explicit;
- preserve strict behavior for explicit malformed tool calls and required-tool
  mode;
- ensure no deterministic semantic router is introduced;
- introduce `ToolObservationRecoveryPolicy`;
- encode `COMPLETED`, `DENIED`, `UNAVAILABLE`, `FAILED`,
  `APPROVAL_REQUIRED`, `APPROVAL_DENIED`, `APPROVAL_EXPIRED` and
  `APPROVAL_CANCELLED` transitions;
- keep tool observations typed and redacted.

Gate:

```text
make test-unit
make test-contract
make test-e2e
make test-golden
make test-architecture
```

## Milestone 3 — Streaming and Event Semantics

Goal: make lifecycle streaming stable enough for CLI and future voice clients.

Tests first:

- public stream emits stable request, loop, context, model, tool and terminal
  phases;
- completed path emits one terminal completion;
- failed path emits one terminal failure;
- reconnect after terminal buffer cleanup can replay terminal request state from
  persisted request status, events and conversation messages;
- restart/reconnect recovery does not require same-process stream memory;
- cancel path leaves the interactive CLI usable;
- final answer streaming limitations are documented if true token streaming is
  still deferred.

Implementation:

- remove accidental duplicate terminal stream events;
- keep activity phases tied to real lifecycle events;
- keep stream replay backed by durable request state and event log, not volatile
  loop-local buffers;
- document any remaining non-token-streaming limitation as PM-09 input.

Gate:

```text
make test-unit
make test-contract
make test-e2e
make test-architecture
```

## Milestone 4 — Final Pre-PM09 Gate

Goal: prove the hardened loop is ready for voice to sit on top of it.

Tests first:

- DB-backed transcript-like API turn with no tool use;
- DB-backed transcript-like API turn with ToolGateway execution;
- DB-backed denied/unavailable/approval path;
- DB-backed reconnect/replay after terminal stream cleanup or daemon restart;
- direct `AgentRuntime` construction remains aligned with default agent loop;
- request-plan tool availability cannot drift from the actual ToolGateway
  registry.

Evidence targets:

- `test_runtime_app_factory_api_no_tool_turn_persists_transcript`;
- `test_runtime_app_factory_api_tool_turn_replays_after_new_app_instance`;
- `test_runtime_app_factory_api_unavailable_tool_turn_replays_after_new_app_instance`;
- `test_runtime_app_factory_api_denied_approval_finishes_controlled`;
- `test_default_runtime_can_execute_default_agent_loop_without_tool_registry`;
- request-plan drift guard tests for missing gateway tools and metadata
  mismatches.

Implementation:

- update PM-09 entry docs to depend on the hardened PM-08l agent loop;
- update known limitations with any remaining true streaming or clarification
  gaps;
- remove or quarantine obsolete PM-04 wording that contradicts PM-08l.

Final gate:

```text
make test-unit
make test-contract
make test-golden
make test-integration
make test-e2e
make test-architecture
git diff --check
```

Then run two read-only review agents from scratch. PM-08l is complete only when
the verification gate is green and there are no relevant P0/P1 findings.

## Out of Scope

- voice audio capture;
- STT/TTS provider implementation;
- wake-word implementation;
- full planner-executor or plan-and-execute implementation;
- provider-native tool calling migration;
- LangGraph or durable workflow checkpointing;
- new production tools;
- semantic classifier or deterministic natural-language route resolver;
- broad model benchmarking.

## Post-Gate Follow-Ups

Final PM-08l review accepted these as non-blocking P2/P3 follow-ups. They must
not block the current PM-08l gate, but should be handled before broadening the
voice surface beyond the initial PM-09 foundation:

- extract live-state math evidence policy, calculator expression matching and
  proposal-output contract rendering out of `ToolReactLoop` into explicit
  runtime services, keeping the public loop class as orchestration only;
- split `tool_loop_evidence.py` into narrower helpers for intent-family
  detection, candidate evidence planning, observation matching, unavailable
  recovery pruning and calculator/live-state math matching. Until then,
  architecture tests enforce a size budget so the helper does not keep
  accumulating unrelated responsibilities;
- replace brittle lexical live-state finalization guards with typed
  request-plan/evidence metadata where possible. Keep the pre-answer evidence
  guard broad and safety-oriented, and keep deterministic finalizers narrow,
  source-backed and post-observation only. Include current-time, sensor,
  temperature and multi-expression threshold coverage;
- add direct unit coverage for `InProcessDarwinResourceProvider`, including
  memory accounting and provider failure paths;
- validate `system.resource_overview` snapshots before marking provider output
  as parsed live evidence;
- add dedicated unit coverage for shared privacy redaction helpers.

## Completion Criteria

PM-08l is complete only when:

- `ToolReactLoop` is a thin bounded-agent-loop orchestrator or has an equivalent
  internal decomposition;
- final answers use one finalization path;
- `auto/chat/tools` behavior is covered by tests and does not rely on
  classifier-first routing;
- tool observations drive explicit recovery, clarification or controlled
  failure;
- lifecycle streaming is stable and documented;
- the loop is compatible with a future plan-and-execute shell that executes
  scoped steps through the same bounded loop and ToolGateway path;
- transcript-like API/e2e turns prove voice can use the same runtime path;
- PM-09 remains blocked until the PM-08l final gate passes.
