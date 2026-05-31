# 38 — PM-08k Request Routing Architecture Review and Classifier Calibration

## Status

Implemented.

This document defines the implemented pre-voice hardening slice after PM-08j and
before PM-09. It records the PM-08k research decision, implementation contract
and acceptance criteria now used by the runtime.

The slice starts with a research gate. The first PM-08k question is not "which
small model should classify requests?" but "should Jarvis use a mandatory
front-gate LLM classifier at all?"

## Problem

The current model-facing structured classifier contract exposes too much of the
internal loop-selection domain directly to a small local model.

The model is currently asked to return a shape close to `IntentClassification`
and `CapabilityCandidate`, including:

```text
intent_family
candidate_capabilities
capability
tool_names
risk_classes
requires_live_state
requires_execution
fallback_preference
reason_code
scope_hint
evidence_codes
```

This makes the local model responsible for concepts that should remain owned by
Jarvis runtime code:

- capability identifiers;
- stable tool names;
- risk classes;
- policy and fallback semantics;
- direct execution metadata;
- diagnostic scope labels.

Recent local evaluation showed the practical failure mode: small local models
often return JSON, but invent out-of-contract labels such as
`general_inquiry`, `content.retrieve`, `datetime.now` as a capability, or
`low` as a risk class. This is not only a JSON-formatting problem. It is a
model-facing contract problem.

The broader risk is architectural: putting an additional LLM classifier in front
of every user turn adds latency, duplicates reasoning and creates a new failure
point before the main runtime sees the request. PM-08k must therefore evaluate
whether the mandatory classifier step is the right architecture, not only
whether its schema should be simplified.

## PM-08k.0 — Industry Research Gate

PM-08k starts with a read-only research and architecture review stage.

Research inputs:

- Jarvis local evaluation evidence from PM-08h/PM-08i;
- current Jarvis runtime boundaries from ADR-035 and PM-08;
- industry patterns for tool use, routing, fallback and abstention.

External references to review:

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
  describes tool calling as an application/model loop: the application gives the
  model tools, receives a tool call, executes code application-side and sends
  tool output back. It also supports `tool_choice` modes such as `auto`, `none`,
  forced function and allowed tools.
- [OpenAI JSON mode notes](https://help.openai.com/en/articles/8555517-function-calling-in-the-openai-api)
  distinguish valid JSON from schema-correct output and require application-side
  handling of edge cases.
- [Anthropic tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
  uses a similar client-tool loop: Claude decides whether a tool can help,
  constructs a tool request, client code executes it and returns results. The
  docs also call out token cost from tool definitions and tool-use messages.
- [Semantic Kernel planning](https://learn.microsoft.com/en-us/semantic-kernel/concepts/planning)
  moved from prompt-based planners toward native function calling as the primary
  planning/execution mechanism.
- [Rasa fallback classifier](https://rasa.com/docs/rasa/reference/rasa/nlu/classifiers/fallback_classifier/)
  and [Rasa fallback policy](https://rasa.com/docs/rasa/reference/rasa/core/policies/fallback/)
  treat low-confidence or ambiguous intent/action predictions as fallback cases,
  not as forced best-effort classifications.
- [LlamaIndex RouterQueryEngine](https://developers.llamaindex.ai/python/framework-api-reference/query_engine/router/)
  uses selectors that choose among candidate query engines based on candidate
  metadata and the query.
- [Haystack ConditionalRouter](https://docs.haystack.deepset.ai/docs/conditionalrouter)
  demonstrates deterministic pipeline routing with explicit conditions, fallback
  routes and optional output type validation.

Research questions:

1. Is a mandatory front-gate LLM classifier needed for Jarvis, or should routing
   be mostly deterministic/metadata-driven with optional model adjudication?
2. Should the main model be allowed to decide tool use for some paths, or should
   Jarvis keep all tool routing outside the answer model for now?
3. Which routes are cheap and safe enough for deterministic fast path?
4. Which routes are better handled by a non-LLM semantic/prototype classifier
   with aggressive abstain behavior?
5. Which cases truly require local LLM route adjudication?
6. When should Jarvis ask a clarification question instead of routing?
7. Which architecture minimizes latency without increasing false live-state
   positives or policy bypass risk?

Stage output:

```text
PM-08k.0 research note
architecture decision matrix
recommended request-routing architecture
recorded decision that mandatory front-gate LLM classifier is rejected as default
updated PM-08k implementation plan if the architecture changes
```

Candidate architectures to compare:

```text
A. Mandatory LLM classifier before every auto request
B. Deterministic-first router with optional local LLM route adjudicator
C. Deterministic + embedding/prototype router + optional local LLM adjudicator
D. Main-model tool-calling path with Jarvis-side policy/tool execution
E. Hybrid: deterministic fast path, direct ordinary chat, local LLM only for
   ambiguous/risky routes, clarification when confidence is insufficient
```

Initial hypothesis:

```text
E is the likely target.

Routing should be cheap and conservative. Execution remains policy-gated.
Reasoning should mostly happen once. A local LLM route classifier, if retained,
is an adjudicator for ambiguous cases rather than the mandatory first step for
every request.
```

PM-08k must not proceed to implementation until this research gate either
confirms the current direction or updates the architecture decision.

## PM-08k.0 Research Outcome

Decision:

```text
The mandatory front-gate LLM classifier is rejected as the default architecture.

Jarvis should move toward a Hybrid Request Resolver:
  deterministic-first routing for obvious safe routes;
  direct ordinary chat when no tool/live-state signal is present;
  optional non-LLM semantic routing only when calibrated to abstain;
  optional local LLM route adjudication only for ambiguous cases;
  clarification or explicit unavailable result when confidence is insufficient
    and execution/live-state risk exists.
```

Rationale:

- A mandatory LLM classifier adds a model call before every normal answer,
  including ordinary chat turns that do not need routing.
- PM-08i local evidence shows current small structured classifier candidates are
  not reliable as direct model-only routers: `qwen3.5:2b` took 103.951 seconds
  across 10 sampled cases and failed 10/10; `qwen3.5:0.8b` took 78.048 seconds
  and also failed 10/10 in the recorded comparison fixture. Additional local
  investigation saw the same failure shape for `qwen3.5:4b`: slower than 2B and
  still not contract-correct. PM-08k implementation must reproduce any such
  measurements in a calibration report before changing defaults.
- The failure is not just malformed JSON. The models can return JSON while
  inventing route, capability, tool-name or risk-class labels.
- Industry tool-use APIs do not generally require a separate classifier call in
  front of every request. OpenAI and Anthropic document model/tool loops where
  the application supplies tools, the model may choose a tool, and application
  code remains responsible for execution and results. They also expose controls
  such as allowed tools, forced/none/auto tool choice and client-side execution
  boundaries.
- Rasa-style NLU treats low confidence and ambiguity as fallback conditions,
  not as a reason to force a best-effort route.
- Haystack and LlamaIndex show routing as a first-class pipeline/metadata
  primitive; the router does not need to be an LLM call in all cases.
- Semantic Kernel's evolution toward native function calling is relevant, but
  Jarvis should not jump directly to main-model tool calling before local model
  support, policy interaction, typed observations and provider neutrality are
  tested for this runtime.

### Decision Matrix

| Option | Verdict | Why |
| --- | --- | --- |
| A. Mandatory LLM classifier before every auto request | Reject as default | Highest latency, duplicates reasoning, adds a pre-answer failure point, and current local models do not follow the broad contract reliably. |
| B. Deterministic-first router with optional local LLM route adjudicator | Accept as baseline direction | Fits local-first operation, keeps obvious safe routes cheap, preserves policy/ToolGateway boundaries, and uses the model only when cheaper layers abstain. |
| C. Deterministic + embedding/prototype router + optional local LLM adjudicator | Evaluate before runtime default | Promising for speed and coverage, but local probes showed dangerous near-miss errors around conceptual vs live-state requests unless it abstains aggressively. |
| D. Main-model tool-calling path with Jarvis-side policy/tool execution | Defer | Strong industry pattern, but it is a larger runtime change and would expose tool schemas to the answer model. Keep as a later ADR/PM after local support and policy semantics are proven. |
| E. Hybrid Request Resolver | Accept | Best fit for Jarvis now: cheap deterministic gates, direct ordinary chat, optional calibrated semantic layer, optional thin local LLM adjudicator, and clarification/unavailable for risky ambiguity. |

### Accepted Architecture

The target PM-08k architecture is:

```text
user request
  -> explicit user/API mode override
  -> deterministic route guards
       slash/CLI commands remain CLI-local commands
       obvious safe builtins
       obvious diagnostics
       obvious project docs
       explicit ordinary-chat hints
  -> direct ordinary chat when no live-state/tool/RAG signal is present
  -> optional non-LLM semantic resolver, only when calibrated to abstain
  -> optional local LLM route adjudicator with thin route schema
  -> clarification or unavailable result for ambiguous risky routes
  -> LoopStrategySelector / PolicyPort / ToolGatewayPort remain authoritative
```

The local LLM route adjudicator, if retained, is no longer a mandatory
front-gate classifier. It is a late-stage resolver for ambiguous cases after
cheap deterministic and non-LLM layers abstain.

### Agreed Design Choices

PM-08k implementation planning should follow these decisions:

1. Introduce `RequestResolver` rather than stretching `IntentClassifierPort`
   into a broader orchestration boundary.

```text
RequestResolver
  -> RouteDecision | Abstain | Clarify | Unavailable
  -> route registry maps accepted routes to IntentClassification
  -> LoopStrategySelector validates and selects the concrete loop
```

2. Use this resolver pipeline:

```text
explicit mode / slash command
  -> deterministic route guards
  -> ordinary chat bypass
  -> optional semantic resolver
  -> optional LLM adjudicator
  -> Clarify / Unavailable
```

Slash commands remain CLI-local commands and do not become backend safety
routing.

3. Obvious chat must not call the classifier.

Normative rule: obvious chat must not call the classifier.

Examples:

```text
Расскажи, как решаются кубические уравнения.
Объясни, что такое VPN.
Как работает аккумулятор?
Напиши пример на Python.
```

If there is no live-state, tool, project-docs or project-inspection signal, the
request should route directly to ordinary chat. This ordinary chat bypass is the
latency-critical happy path.

4. Use deterministic guards for obvious, low-risk, high-value routes:

```text
current_time
date_countdown
calculator
daemon_status
explicit ordinary-chat hints
strong project-docs markers
clear live-state diagnostics with current/local wording
```

Process, network, temperature and project inspection routes should be more
conservative because their near-miss surface is larger.

5. Resolvers must abstain instead of forcing a best guess. If ambiguity is
harmless, ordinary chat is acceptable. If ambiguity could trigger live-state or
tool behavior, return `Clarify` or `Unavailable`.

Example:

```text
память сейчас -> likely system_memory
как работает память -> ordinary_chat
проверь память -> system_memory
расскажи как проверить память -> ordinary_chat
```

If the resolver cannot distinguish ordinary explanation from live-state
diagnostics, it should ask whether the user wants an explanation or a current
machine check.

6. The non-LLM semantic layer starts as evaluation/calibration only. It may
become runtime default only if it demonstrates high precision on selected cases,
low false live-state positives, a clear abstain threshold and materially lower
latency than the LLM adjudicator.

7. The LLM adjudicator is optional and late-stage. If retained, it is called only
after deterministic and non-LLM resolvers abstain. It uses the thin route schema,
has a strict timeout, and invalid output becomes `Abstain` or `Unavailable`.

8. Blocking PM-09 metrics include:

```text
false live-state positives on near-miss corpus
direct_plan correctness for ci_baseline
exact tool_names after deterministic route mapping
ordinary chat bypass model-call avoidance
p95 resolver latency without model call
abstain rate reported explicitly
Clarify / Unavailable behavior for risky ambiguity
invalid model output never creates tool metadata
```

### Main-Model Tool Calling

main-model tool calling is deferred.

Reason:

- OpenAI, Anthropic and Semantic Kernel make tool calling a strong industry
  direction, but Jarvis currently has an explicit `ToolGatewayPort`,
  policy/audit model, typed observation path and provider-neutral local-first
  constraints.
- PM-08k can borrow the shape of tool choice controls from industry, especially
  allowed/none/auto semantics, without moving tool selection into the main
  answer model yet.
- A later ADR can evaluate provider-native/main-model tool calling once the
  local models, tool schemas, streaming UX, policy gates and typed observation
  contracts are ready for that larger change.

### PM-08k Follow-Up Slices

PM-08k.1 — Hybrid request resolver contract:

```text
Introduce RequestResolver and define RouteDecision, Abstain, Clarify and
Unavailable result types, explicit clarification/unavailable states, ordinary
chat bypass behavior and the ordered resolver pipeline. Keep CLI commands out of
backend safety routing.
```

PM-08k.2 — Deterministic and route-registry implementation:

```text
Move obvious safe routes into deterministic route guards backed by registry
metadata. Preserve PolicyPort, ToolGatewayPort and direct-plan validation as
authoritative execution gates.

Initial deterministic route set:
  current_time
  date_countdown
  calculator
  daemon_status
  explicit ordinary-chat hints
  project_docs_question with strong docs/ADR/roadmap markers
  clear live-state diagnostics with current/local wording
```

PM-08k.3 — Optional semantic/model adjudication and calibration:

```text
The non-LLM semantic layer starts as evaluation/calibration only, not runtime
default. Evaluate embedding/prototype routing and a thin local LLM route schema
only for cases where deterministic routing abstains. Compare thresholds, false
positives, abstain rates and latency before enabling any runtime default.
The LLM adjudicator is optional and late-stage; it must not become a mandatory
front gate again.
```

PM-08k.4 — Runtime default selection and acceptance gate:

```text
Enable only the resolver layers that pass calibration. Keep ambiguous semantic
and LLM layers disabled or opt-in until they prove lower latency without
increasing false live-state positives.
```

## Direction

If PM-08k keeps a model-backed routing layer, it should introduce a thinner
model-facing classifier contract. The model should classify into a small closed
route vocabulary and abstain when uncertain. Runtime code then maps that route
into the existing domain objects.

Target shape:

```json
{
  "route": "ordinary_chat",
  "confidence": 0.0,
  "requires_live_state": false,
  "is_conceptual_question": true,
  "abstain": false
}
```

The exact route enum is a PM-08k design input, but it should use product-level
routes rather than runtime internals. Example route families:

```text
ordinary_chat
project_docs_question
project_inspection
current_time
date_countdown
calculator
daemon_status
system_os_version
system_cpu_overview
system_memory
system_disk
system_battery
system_temperature
system_processes
system_network
system_vpn
unknown
```

PM-08k should use a medium-grained route taxonomy. It should be more specific
than one broad `system_diagnostics` bucket, but it should not mirror adapter
internals such as `free_memory_bytes`, `cpu_core_count` or command-specific
parsers. The recommended first route set is the list above.

The model must not output:

```text
capability
tool_names
risk_classes
fallback_preference
approval_possible
direct_tool_plan
raw shell commands
tool arguments
provider-specific request dictionaries
```

## Runtime Mapping

Jarvis should map route classifications into the existing selector domain
deterministically:

```text
model-facing route
  -> route registry mapping
  -> IntentClassification
  -> CapabilityCandidate
  -> LoopStrategySelector
  -> policy / ToolGateway / direct-plan checks
```

The route registry, not the model, owns:

- `IntentFamily`;
- `Capability`;
- stable tool names;
- risk classes;
- fallback behavior;
- scope labels;
- direct execution eligibility.

`LoopStrategySelector`, `PolicyPort`, `ToolGatewayPort` and
`CapabilityRoutingRegistry` remain authoritative. The model proposes a route;
it never authorizes execution.

## Classifier Stack

Subject to PM-08k.0, PM-08k should evaluate a three-layer classifier stack:

```text
1. Deterministic high-precision fast path
2. Non-LLM classifier that can abstain
3. Optional local LLM route classifier for ambiguous cases
```

The non-LLM layer can be one of:

- embedding nearest-neighbor/prototype routing over the intent corpus;
- a small statistical classifier if a dependency is accepted later;
- another deterministic route scorer with explicit abstain thresholds.

The non-LLM layer must be calibrated for abstention. It must not become a broad
always-answer classifier for conceptual/live-state near misses.

## Calibration

Thresholds such as `0.87` must be calibrated against recorded evidence, not
chosen by latency preference alone.

After PM-08k.0 selects the candidate routing architecture, PM-08k should compare
at least:

```text
deterministic-only route coverage
embedding/non-LLM route precision and coverage by abstain threshold
local LLM route classifier behavior for qwen3.5:2b and qwen3.5:4b
optional smaller candidates only if installed and explicitly evaluated
end-to-end domain mapping after route -> IntentClassification conversion
latency per classifier layer
```

Required metrics:

```text
route accuracy
mapped IntentClassification accuracy
tool-name/direct-plan correctness after deterministic mapping
false-positive live-state rate
ordinary conceptual near-miss false-positive rate
abstain rate
coverage
precision at each threshold
p50/p95 classifier latency
model-call count avoided by fast paths
```

The default runtime configuration should not change merely because a smaller
model is faster. A new default requires evidence that the route contract,
mapping and fallback behavior are at least as safe as the previous baseline.

## PM-08k Slice Contract

PM-08k is complete only when:

- PM-08k.0 research gate is complete;
- the architecture decision rejects the mandatory front-gate LLM classifier as
  the default and accepts the Hybrid Request Resolver direction;
- the model-facing route schema exists and is smaller than the internal
  `IntentClassification` domain object if a model adjudicator remains in scope;
- route parsing rejects unknown route labels and malformed booleans;
- invalid model output falls back or abstains instead of creating tool metadata;
- route-to-domain mapping is deterministic and registry-backed;
- model output cannot directly specify tool names, capabilities, risk classes,
  policy outcome or direct execution;
- deterministic, non-LLM and local LLM classifiers can be evaluated through one
  comparable report format;
- the pre-voice corpus report includes calibration data for threshold choices;
- PM-09 is blocked until PM-08k is either implemented or explicitly rejected by
  an updated architecture decision.

## Tests First

Unit tests:

```text
test_model_route_schema_contains_only_route_confidence_flags_and_abstain
test_model_route_schema_uses_closed_route_enum
test_model_route_parser_rejects_unknown_route
test_model_route_parser_rejects_non_boolean_flags
test_model_route_output_cannot_supply_tool_names_or_capabilities
test_route_registry_maps_current_time_to_safe_datetime_candidate
test_route_registry_maps_system_memory_to_read_resources_candidate
test_route_registry_keeps_project_docs_as_chat_with_rag_context
test_route_registry_maps_unknown_or_abstain_to_safe_fallback
test_conceptual_live_state_near_miss_does_not_map_to_tool_route
```

Documentation/architecture tests:

```text
test_pm08k_docs_start_with_industry_research_gate
test_pm08k_research_gate_compares_mandatory_classifier_and_hybrid_router
test_pm08k_research_gate_records_source_links_and_architecture_decision
```

Evaluation tests:

```text
test_classifier_calibration_report_records_route_accuracy_and_latency
test_classifier_calibration_report_compares_deterministic_embedding_and_llm
test_classifier_calibration_report_records_threshold_precision_coverage
test_classifier_calibration_report_blocks_default_model_change_on_regression
```

Architecture tests:

```text
test_model_facing_route_schema_does_not_expose_policy_or_tool_execution_fields
test_route_registry_depends_on_capability_metadata_not_tool_adapters
test_model_route_classifier_depends_on_model_router_port_not_provider_adapters
test_cli_does_not_import_route_registry_or_classifier_implementation
```

Expected red phase:

```text
PM-08k.0 industry research gate does not exist
model-facing route schema does not exist
route registry does not exist
current model-backed classifier still asks the model for capability candidates
calibration report does not compare route-level classifiers
PM-09 dependency chain does not mention PM-08k
```

## Non-Goals

PM-08k does not add:

- voice audio handling;
- cloud classifier calls;
- fine tuning or training pipelines;
- broad benchmark infrastructure;
- new tools;
- direct execution bypasses;
- CLI-owned routing logic.

## Discussion Questions

Before implementation, decide:

1. How aggressive should the deterministic part of the accepted Hybrid Request
   Resolver be before it must abstain?
2. Should the main answer model ever receive tool schemas and choose tool calls,
   or should Jarvis keep tool-routing outside the answer model until later?
3. Is the proposed route enum granular enough, or should diagnostics routes be
   grouped more coarsely?
4. Should PM-08k introduce a new `local_classifier` profile or keep using
   `local_structured`?
5. Should embedding/prototype routing be a runtime layer or only an evaluation
   baseline for now?
6. Should `0.87` be a candidate threshold in calibration, or should threshold
   selection be entirely report-driven?
7. Should PM-09 be hard-blocked on PM-08k, or may PM-09 proceed if PM-08k is
   explicitly rejected after discussion?
