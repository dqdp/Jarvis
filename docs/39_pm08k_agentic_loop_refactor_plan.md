# 39 — PM-08k Agentic Loop Refactor Plan

## Status

Planned implementation plan.

This document is the execution plan for implementing ADR-037 and the PM-08k
agentic-loop-first decision. It is intentionally separate from the research
document so the implementation can be reviewed as a concrete refactor sequence.

## Target Contract

Production natural-language request handling must follow one path:

```text
typed input / API request / future voice transcript
  -> request lifecycle
  -> bounded agent loop
  -> model answers or proposes a tool
  -> schema / allowlist / PolicyPort / ToolGatewayPort gates
  -> typed tool observation
  -> same bounded agent loop
  -> final answer
```

The production path must not run a semantic pre-router before the agent loop:

- no runtime LLM route classifier;
- no route-threshold tuning such as `0.87` or `0.9`;
- no `RequestResolver` / `RouteDecision` as a target runtime layer;
- no broad deterministic natural-language intent guards;
- no direct natural-language tool execution path.

Deterministic code remains for control and safety only:

- slash commands, cancel/exit and approval controls;
- voice-channel lifecycle events such as wake word/name detection,
  push-to-talk, silence timeout, stop/cancel/barge-in and short activation
  acknowledgement;
- mode validation for `auto`, `chat` and `tools`;
- non-TTY/plain CLI behavior;
- tool allowlists and argument schemas;
- policy, permission, sensitivity, budgets and approvals;
- redaction, audit and event-shape constraints.

These direct control-plane paths may update channel or session state without
entering ReAct because they are not user requests for knowledge or tool
execution. Once a voice or text input contains a substantive user request, it
must enter the same bounded agent loop as every other natural-language request.

## Current Hotspots

The refactor is concentrated in these areas:

```text
src/assistant_core/runtime/request_metadata.py
  currently performs loop selection through classifier/selector and writes
  classifier/direct-plan metadata.

src/assistant_core/app_factory.py
  currently wires RequestResolverIntentClassifier / HybridRequestResolver into
  the API app.

src/assistant_core/runtime/loop_selection.py
src/assistant_core/runtime/request_resolver.py
src/assistant_core/runtime/model_intent_classifier.py
  become historical/evaluation or deleted/quarantined code, not production
  request handling.

src/assistant_core/runtime/direct_tools.py
  should be removed from the default natural-language path.

src/assistant_core/runtime/loops/tool_react.py
  already contains the bounded tool loop, but still consumes selector candidate
  tools and direct plans. It also currently uses the structured model profile as
  the loop model for selected tool paths.

src/assistant_core/domain/loop_selection.py
  remains historical/compatibility only unless replaced by a narrower execution
  planning domain.

tests/unit/test_model_intent_classifier.py
tests/unit/test_request_resolver.py
tests/evaluation/test_tool_intent_routing_corpus_eval.py
  must be deleted, quarantined or rewritten as historical/evaluation tests.
```

## Design Clarifications Before Implementation

These choices must be treated as part of the refactor contract, not as
implementation details to rediscover mid-slice.

### Agent Loop Naming and Migration

PM-08k should not create a second production loop beside the current bounded
tool loop. The pragmatic sequence is:

```text
1. Keep the existing ToolReactLoop class as the implementation vehicle.
2. Make it satisfy the broader bounded agent loop contract.
3. Add tests and metadata that describe it as the agent loop for
   natural-language requests.
4. Rename to AgentLoop only after behavior is green and the rename is a
   mechanical cleanup.
```

This avoids a large rename while the semantics are still moving.

### Model Profile Contract

The agent loop must not become a small structured-classifier loop.

```text
reasoning and final answers -> local_main / chat-purpose profile
tool proposal formatting     -> structured output only when safely bounded
tool execution               -> ToolGatewayPort only
```

If the implementation temporarily uses structured model calls for tool proposal
JSON, it must still use the main reasoning model for ordinary reasoning/final
answers. `local_structured` must not become the default answer model for all
natural-language turns.

### Mode Semantics

`auto`, `chat` and `tools` are policy modes for the same agent loop:

```text
chat:
  tools disabled
  context/RAG still allowed through ContextAssembler
  final answer may be produced without tool observations

auto:
  tools available when policy/budget allow
  model may answer directly or propose a tool
  tool use is not required

tools:
  tools available when policy/budget allow
  at least one tool observation is required before final_answer when the allowed
    tool list is non-empty
  fails closed when tools are disabled, budget cannot execute tools or no tools
    are allowed
```

This keeps explicit `tools` useful for debugging and future voice/tool tests
without turning it into a semantic router.

### Allowed Tool Surface

The agent loop receives tools from runtime metadata, not from a natural-language
classifier.

Initial PM-08k rule:

```text
allowed_tool_names = enabled registry tools
  filtered by config
  filtered by runtime budget
  filtered by coarse policy/sensitivity gates
```

Do not introduce semantic narrowing by user text in PM-08k. If the full enabled
tool list becomes too large for local models, solve that as a later tool-surface
compression problem using static tool grouping, summaries or explicit user
mode, not a hidden intent router.

### Replacement Metadata and Events

Removing `loop_selection_*` metadata must not make CLI/SSE status worse.
Introduce redacted request-plan metadata instead:

```text
requested_loop_mode
selected_loop_strategy
agent_tool_policy
agent_allowed_tool_count
agent_allowed_tool_names if safe and bounded
selected_model_profile
request_plan_status
request_plan_reason_code
```

Do not write:

```text
intent_family
classifier confidence
classification_source
route labels
direct_tool_plan
raw user prompt
```

SSE/CLI activity phases should continue to show selecting/planning, context
assembly, tool running, waiting approval, streaming, failed, cancelled and done,
but those phases are observability, not routing authorization.

### Quarantine Gate

Do not delete classifier-era modules before production imports are gone.

Required sequence:

```text
1. Add architecture tests proving production code does not import them.
2. Rewire production path.
3. Run focused and architecture tests.
4. Only then delete/quarantine classifier-era modules and tests.
```

This avoids mixing a behavior refactor with a large deletion diff.

## Milestone Workflow

PM-08k must be implemented as gated milestones. Each milestone is a commit
boundary and must follow the same workflow:

```text
1. Write or update milestone tests first.
2. Run the focused tests and confirm meaningful red failures.
3. Implement the smallest production/doc change for that milestone.
4. Run the milestone verification gate until green.
5. Run two independent read-only review agents.
6. Relevant P0/P1 findings block the milestone.
7. Fix P0/P1 findings, rerun verification and rerun review for the affected
   milestone.
8. Commit the milestone only when verification is green and there are no
   relevant P0/P1 findings.
9. Start the next milestone only after the commit.
```

P2/P3 findings may be fixed inside the milestone if low-risk and scoped. They
do not require another review pass unless the user explicitly asks for one or
the fix changes the milestone architecture.

### Milestone 1 — Request Plan Contract

Goal:

```text
Introduce AgentRequestPlan and replacement request-plan metadata while keeping
old production behavior otherwise intact.
```

Scope:

- add `AgentRequestPlan` or equivalent narrow planning object;
- add metadata names for `agent_tool_policy`, allowed tool count/names and
  request-plan status/reason;
- add red tests proving no classifier fields belong in the new plan;
- keep old selector/classifier wiring in place until Milestone 2.

Verification gate:

```text
.venv/bin/python -m pytest -m unit tests/unit/test_runtime_request_metadata.py
.venv/bin/python -m pytest -m architecture tests/architecture/test_boundaries.py
git diff --check
```

Milestone is complete only when the new contract exists, is tested, and does
not yet require removing old production wiring.

### Milestone 2 — Production Wiring Without Classifier

Goal:

```text
Make production request creation use AgentRequestPlan instead of
RequestResolver/IntentClassifier/LoopStrategySelector.
```

Scope:

- refactor `runtime_request_metadata()`;
- remove production imports of `IntentClassifierPort`, `LoopStrategySelector`,
  `DeterministicIntentClassifier` and `DirectToolPlanner`;
- remove `build_intent_classifier()` from production app factory wiring;
- keep classifier-era modules present but unused;
- add architecture tests proving production wiring does not import classifier
  or request-resolver modules.

Verification gate:

```text
.venv/bin/python -m pytest -m unit tests/unit/test_runtime_request_metadata.py
.venv/bin/python -m pytest -m contract tests/contract/test_api_lifecycle_contract.py
.venv/bin/python -m pytest -m contract tests/contract/test_app_factory_contract.py
.venv/bin/python -m pytest -m architecture tests/architecture/test_boundaries.py
git diff --check
```

Milestone is complete only when production `auto/chat/tools` planning no longer
uses classifier/request-resolver wiring.

### Milestone 3 — Agent Loop Semantics

Goal:

```text
Make the existing ToolReactLoop satisfy the bounded agent loop contract for
auto/chat/tools without direct natural-language execution.
```

Scope:

- use main reasoning/chat profile for final-answer reasoning;
- keep structured output only as bounded proposal formatting where needed:
  `{"action":"final_answer"}` is a readiness signal, not the user-facing answer;
- enforce `chat`, `auto` and `tools` mode semantics;
- use request-plan allowed tools instead of selector candidate tools;
- remove direct plan consumption from the loop;
- ensure malformed proposals, denied tools and unsupported event/date requests
  fail closed or clarify.

Verification gate:

```text
.venv/bin/python -m pytest -m unit tests/unit/test_tool_react_loop.py
.venv/bin/python -m pytest -m contract tests/contract/test_tool_react_loop_contract.py
.venv/bin/python -m pytest -m contract tests/contract/test_sse_stream_contract.py
.venv/bin/python -m pytest -m architecture tests/architecture/test_boundaries.py
git diff --check
```

Milestone is complete only when natural-language requests execute through the
bounded agent loop and tool proposals cannot bypass schema/allowlist/policy/
ToolGateway gates.

### Milestone 4 — Quarantine Classifier-Era Code

Goal:

```text
Delete, quarantine or rewrite classifier-era modules and tests after production
imports are gone.
```

Scope:

- remove or isolate `request_resolver.py`, `model_intent_classifier.py`,
  classifier-specific `loop_selection.py` behavior and `direct_tools.py`;
- delete or quarantine tests that only prove rejected classifier-first behavior;
- keep evaluation fixtures only when explicitly labeled historical/evaluation;
- update docs and architecture tests to protect the new production boundary.

Verification gate:

```text
.venv/bin/python -m pytest -m unit tests/unit
.venv/bin/python -m pytest -m contract tests/contract
.venv/bin/python -m pytest -m architecture tests/architecture
git diff --check
```

Milestone is complete only when classifier-era code is no longer a production
dependency and remaining historical/evaluation artifacts cannot block PM-09.

### Milestone 5 — Full PM-08k Gate

Goal:

```text
Prove the complete PM-08k refactor is green and ready to unblock PM-09.
```

Scope:

- run full verification;
- run two independent read-only review agents;
- fix relevant P0/P1 findings and repeat the affected milestone gate;
- commit only after no relevant P0/P1 findings remain.

Verification gate:

```text
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m architecture tests/architecture
git diff --check
```

Milestone is complete only when the full suite is green, review is clean for
relevant P0/P1 findings and the final PM-08k commit is made.

## Refactor Sequence

### PM-08k.1 — Red Tests and Architecture Guards

Write failing tests before changing production code.

Required unit/contract tests:

```text
test_default_auto_request_builds_agent_loop_plan_without_classifier
test_chat_mode_builds_agent_loop_plan_with_tools_disabled
test_tools_mode_builds_agent_loop_plan_with_tools_available_or_required
test_request_metadata_does_not_write_loop_selection_direct_tool_plan
test_request_metadata_does_not_write_classifier_confidence_or_intent_family
test_voice_transcript_uses_same_agent_request_plan_shape
test_natural_language_calculator_request_has_no_direct_plan
test_calendar_event_request_has_no_pre_router_guess
```

Required architecture tests:

```text
test_app_factory_does_not_import_request_resolver_or_model_intent_classifier
test_request_metadata_does_not_import_intent_classifier_or_loop_selector
test_production_runtime_does_not_import_direct_tool_planner
test_cli_does_not_import_runtime_tool_adapters_or_route_classifiers
```

Expected red failures:

```text
app_factory still wires RequestResolverIntentClassifier;
request_metadata still imports LoopStrategySelector and DeterministicIntentClassifier;
request_metadata still writes classifier metadata and direct plans;
tool_react still consumes direct plan metadata;
auto mode can still be selected by pre-agent semantic classification.
```

### PM-08k.2 — Introduce Agent Request Planning

Replace semantic loop selection with a narrow request execution plan.

Add a small runtime/domain object such as `AgentRequestPlan` with only:

```text
requested_mode
selected_loop_strategy
tool_policy: disabled | available | required
allowed_tool_names
selected_model_profile
budget
redacted metadata
```

Mapping rules:

```text
auto  -> bounded agent loop, tools available when policy/budget allow
chat  -> bounded agent loop, tools disabled
tools -> bounded agent loop, tools available/required when policy/budget allow
```

The plan must not contain:

```text
intent_family
confidence
classification_source
candidate_capabilities from a classifier
route labels
direct_tool_plan
```

### PM-08k.3 — Rewire Request Metadata

Refactor `runtime_request_metadata()` so it:

- validates `auto/chat/tools`;
- computes tool policy and allowed tool names from registry, settings, budget
  and policy;
- selects the bounded agent loop for natural-language requests;
- selects the main agent/reasoning model profile by default;
- writes only request-plan metadata needed by runtime, UI and audit.

Remove production dependencies on:

```text
IntentClassifierPort
LoopStrategySelector
DeterministicIntentClassifier
DirectToolPlanner
CapabilityRoutingRegistry as a semantic router
```

`CapabilityRoutingRegistry` may remain only as tool metadata/allowlist source if
that boundary stays useful.

### PM-08k.4 — Rewire App Factory

Remove production wiring for:

```text
build_intent_classifier()
RequestResolverIntentClassifier
HybridRequestResolver
ModelRouteResolver
model_route_adjudicator_enabled
deterministic_fast_path_threshold
```

The API app should receive request-planning dependencies, not an intent
classifier. If compatibility requires keeping a parameter temporarily, it must
be unused in production and covered by a deprecation/quarantine test.

### PM-08k.5 — Harden the Bounded Agent Loop

Refactor `ToolReactLoop` or introduce a renamed `AgentLoop` so it becomes the
single bounded agent loop.

Required behavior:

- use the main reasoning model for ordinary agent reasoning;
- expose allowed tools to the loop through request-plan metadata;
- reject model-proposed tools not present in the allowed list;
- validate arguments before ToolGateway execution;
- send denied/unavailable/malformed observations back through deterministic
  failure handling or clarification;
- keep typed tool observations as the only tool result format.

Important design point:

```text
Do not replace the classifier with a small structured model that answers every
turn. The agent loop must use the main reasoning model for reasoning/final
answers, while structured output may be used only as a bounded proposal format
if the model/profile supports it safely.
```

### PM-08k.6 — Remove Direct Natural-Language Execution

Remove direct execution from the default request path:

- no direct calculator execution from natural language;
- no direct date/event countdown execution;
- no direct diagnostics execution;
- no direct plan metadata consumed by the loop.

If an optimization such as exact current time or explicit symbolic calculator is
kept, it must be isolated behind a separate follow-up ADR and must not interpret
arbitrary natural language.

### PM-08k.7 — Quarantine or Delete Classifier-Era Code

After production wiring is green:

- delete modules that have no remaining use;
- or move them to clearly named historical/evaluation code paths;
- update tests so historical classifier fixtures do not gate PM-09 readiness;
- keep only regression evidence that explains why classifier-first routing was
  rejected.

Candidate modules/tests:

```text
src/assistant_core/runtime/request_resolver.py
src/assistant_core/runtime/model_intent_classifier.py
src/assistant_core/runtime/loop_selection.py
src/assistant_core/runtime/direct_tools.py
tests/unit/test_request_resolver.py
tests/unit/test_model_intent_classifier.py
tests/unit/test_loop_selection.py
tests/unit/test_direct_tool_planner.py
tests/evaluation/test_tool_intent_routing_corpus_eval.py
```

Do not delete all at once unless architecture tests prove there are no remaining
production imports.

### PM-08k.8 — Voice Readiness Contract

Before PM-09 starts, add tests proving the future voice path will be a client
channel over the same request lifecycle:

```text
transcript -> same request body shape -> same AgentRequestPlan -> same agent loop
```

There must be no voice-specific:

- route classifier;
- deterministic semantic router;
- tool allowlist bypass;
- policy or ToolGateway bypass.

## Cross-Milestone Verification Notes

Each milestone defines its own required verification gate above. The commands
below show the intended broadening order across the full PM-08k refactor, but
they do not replace per-milestone gates, review agents or commit boundaries.

Suggested order:

```text
.venv/bin/python -m pytest -m unit tests/unit/test_runtime_request_metadata.py
.venv/bin/python -m pytest -m unit tests/unit/test_tool_react_loop.py
.venv/bin/python -m pytest -m contract tests/contract/test_api_lifecycle_contract.py
.venv/bin/python -m pytest -m contract tests/contract/test_tool_react_loop_contract.py
.venv/bin/python -m pytest -m architecture tests/architecture
.venv/bin/python -m pytest
```

Milestone 5 is the final PM-08k gate. Earlier milestones must not be carried
forward uncommitted merely because the final full-suite gate is still pending.

## Completion Criteria

PM-08k refactor is complete only when:

- production request handling enters the bounded agent loop by default;
- `auto/chat/tools` are policy modes, not semantic routers;
- production wiring does not import classifier/request-resolver modules;
- classifier thresholds do not affect runtime request handling;
- direct natural-language execution is removed from the default path;
- model-origin tool proposals always pass schema, allowlist, PolicyPort and
  ToolGatewayPort gates;
- unsupported or risky requests fail closed or ask clarification instead of
  guessing;
- PM-09 voice documentation points to the same request lifecycle and agent loop.
