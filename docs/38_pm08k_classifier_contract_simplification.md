# 38 — PM-08k Agentic Loop-First Request Handling

## Status

Updated plan.

This document defines the implemented pre-voice hardening slice after PM-08j and
before PM-09. It records the PM-08k research decision, implementation contract
and acceptance criteria for replacing classifier-first routing with an
agentic-loop-first runtime path.

The concrete refactor sequence is tracked separately in
`docs/39_pm08k_agentic_loop_refactor_plan.md`.

The slice starts with a research gate. The first PM-08k question is not "which
small model should classify requests?" but "should Jarvis use a mandatory
front-gate LLM classifier at all?"

The current decision is stronger than the earlier Hybrid Request Resolver plan:
Jarvis should remove runtime LLM route adjudication and broad deterministic
intent routing from the default natural-language path. The bounded agent loop is
the central request handler for typed input and future voice transcripts.

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

The broader risk is architectural: putting any semantic router in front of every
user turn adds latency, duplicates reasoning and creates a new failure point
before the main agent loop sees the request. PM-08k must therefore remove the
extra runtime classification layer, not merely simplify its schema.

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
   be removed in favor of one bounded agent loop?
2. Should model understanding happen inside the same loop that can answer,
   retrieve context, propose tools and handle observations?
3. Which behavior must remain deterministic because it is control, safety,
   policy, schema validation, budget or redaction rather than semantic routing?
4. Which historical classifier/prototype fixtures should remain as evaluation
   evidence with aggressive abstain behavior?
5. When should Jarvis ask a clarification question instead of executing a tool?
6. Which architecture minimizes latency without increasing false live-state
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

Initial hypothesis before research:

```text
The early PM-08k draft expected E, the Hybrid Request Resolver, to be the likely
target.

The PM-08k.0 research outcome below supersedes that hypothesis. The accepted
direction is agentic-loop-first request handling, with no semantic route
classifier before the loop.
```

PM-08k must not proceed to implementation until this research gate either
confirms the current direction or updates the architecture decision.

## PM-08k.0 Research Outcome

Decision:

```text
The mandatory front-gate LLM classifier is rejected as the default architecture.

Jarvis should move to agentic-loop-first request handling:
  every natural-language typed request or voice transcript enters the bounded
    agent loop by default;
  the model inside that loop decides whether to answer or propose a tool call;
  PolicyPort, ToolGatewayPort, approval, budgets and tool schemas remain
    authoritative outside the model;
  deterministic code handles control and safety, not semantic request routing;
  runtime LLM route adjudication and route-threshold tuning are removed from the
    production path.
```

Rationale:

- A mandatory LLM classifier adds a model call before every normal answer,
  including ordinary chat turns that do not need routing.
- PM-08i local evidence shows current small structured classifier candidates are
  not reliable as direct model-only routers: `qwen3.5:2b` took 103.951 seconds
  across 10 sampled cases and failed 10/10; `qwen3.5:0.8b` took 78.048 seconds
  and also failed 10/10 in the recorded comparison fixture. Additional local
  investigation saw the same failure shape for `qwen3.5:4b`: slower than 2B and
  still not contract-correct. These measurements are retained as historical
  evidence for rejecting classifier-first routing. They are not a calibration
  gate for changing PM-08k production defaults or starting PM-09.
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
| B. Deterministic-first router with optional local LLM route adjudicator | Reject for runtime default | Still creates a second semantic decision point before the agent loop and keeps threshold/schema complexity alive. |
| C. Deterministic + embedding/prototype router + optional local LLM adjudicator | Evaluation only | Useful as research evidence, but not the production request path before voice. |
| D. Main-model tool-calling path with Jarvis-side policy/tool execution | Accept as target direction | Best matches industry tool-loop patterns and keeps request understanding in one place while preserving Jarvis-side execution control. |
| E. Hybrid Request Resolver | Reject as PM-08k target | Better than a mandatory classifier, but still too complex and fragile for voice because it retains pre-agent semantic routing. |

### Accepted Architecture

The target PM-08k architecture is:

```text
user text or voice transcript
  -> API/CLI request lifecycle
  -> bounded agent loop
      model may answer ordinary chat directly
      live-state claims require completed local evidence when observable
      model may propose a tool call from the supplied/allowed tools
  -> PolicyPort / ToolGatewayPort validate and execute tool calls
  -> tool observation returns to the same loop
  -> final answer
```

There is no local LLM route adjudicator in the runtime default. There is also no
deterministic natural-language intent router whose job is to understand the
user before the agent loop.

### Agreed Design Choices

PM-08k implementation planning should follow these decisions:

1. Treat the bounded agent loop as the request-understanding boundary for
   natural-language input.

```text
normal typed input / voice transcript
  -> bounded agent loop
  -> model decides answer vs tool proposal
  -> PolicyPort / ToolGatewayPort validate execution
```

2. Deterministic code is allowed only for control and safety:

```text
slash commands
cancel / exit / approval controls
plain / non-TTY behavior
permission and sensitivity gates
tool allowlist and schema validation
budgets and approval requirements
redaction and event-shape constraints
```

It must not be a hidden natural-language intent router.

3. Remove the runtime LLM route adjudicator and its threshold semantics from the
production request path. Classifier comparison fixtures may remain historical
evidence, but PM-08k implementation must not require a model call before the
agent loop.

4. Direct execution is not part of the default natural-language path. If a
future optimization reintroduces direct answers for current time or explicit
calculator expressions, it needs a separate ADR and must prove that it cannot
truncate mixed natural-language expressions or bypass PolicyPort/ToolGateway.
Narrow deterministic finalization inside the bounded loop is different: it may
format completed typed evidence, or answer current available-tools questions
from the current `ToolRequestPlan.allowed_tool_names` and matching safe
summaries. It must not use RAG, a global registry, hidden/disabled tools or
external tool catalogs as the source of current availability.

5. Unsupported event/date questions must not be guessed by a pre-router. They
enter the same agent loop. If a tool cannot resolve the event, the agent should
surface an explicit unresolved/clarifying result rather than inventing a date.

6. Blocking PM-09 metrics include:

```text
all natural-language paths enter the same bounded loop
no runtime LLM route classifier is called before the loop
no deterministic natural-language intent router exists in the default path
tool proposals are constrained by allowlists and validated schemas
PolicyPort and ToolGatewayPort remain the only execution gates
unsupported/risky tool attempts fail closed or request clarification
voice transcripts use the same request lifecycle as typed input
```

### Provider-Native Tool Calling

Provider-native tool calling is deferred.

Reason:

- OpenAI, Anthropic and Semantic Kernel make tool calling a strong industry
  direction. Jarvis should follow the loop shape while keeping execution behind
  its own `ToolGatewayPort`, policy/audit model, typed observation path and
  provider-neutral local-first constraints.
- PM-08k adopts model-in-loop tool choice for the bounded Jarvis agent loop,
  not provider-native uncontrolled execution. The application supplies bounded
  tools, validates proposals and owns every side effect.
- Later ADRs can evaluate provider-native tool APIs, but PM-08k should first
  remove the separate classifier/router layer so the text and voice paths share
  one agentic control loop.

### PM-08k Follow-Up Slices

PM-08k.1 — Agentic loop default contract:

```text
Define the default runtime path as bounded agent loop first for natural-language
typed input and future voice transcripts. Remove the requirement for
RequestResolver, RouteDecision, model-route adjudication and natural-language
deterministic guards in the production path.
```

PM-08k.2 — Control and safety determinism:

```text
Keep deterministic code for:
  slash/control commands
  policy and permission checks
  sensitivity handling
  tool allowlists and schemas
  budgets and approvals
  event/log redaction
  non-TTY/plain behavior

Do not use deterministic code as a hidden semantic router for normal language.
```

PM-08k.3 — Remove classifier and threshold runtime complexity:

```text
Remove runtime LLM route adjudicator wiring, model-route schema parsing,
deterministic fast-path threshold behavior and model-route calibration gates from
the production request path. Keep historical classifier comparison fixtures only
as evidence for why this direction was rejected.
```

PM-08k.4 — ReAct/tool loop hardening:

```text
Ensure the bounded loop can handle ordinary answers, project-docs questions,
safe builtins and read-only tools without a pre-router. Unsupported events,
malformed tool arguments and denied tool calls fail closed or ask for
clarification through the loop.
```

## Rejected Route Schema

PM-08k no longer keeps a model-backed routing layer in runtime. The old thin
route schema below is retained only as historical research output and should not
be implemented as production request handling.

Rejected historical shape:

```json
{
  "route": "ordinary_chat",
  "confidence": 0.0,
  "requires_live_state": false,
  "is_conceptual_question": true,
  "abstain": false
}
```

The earlier route enum used product-level routes rather than runtime internals.
It is useful as evaluation vocabulary, not as production routing state. Example
historical route families:

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

This route taxonomy is not the PM-08k target. If retained, it belongs in
offline/evaluation reports only. Runtime request handling should expose
available tools to the bounded agent loop, validate any model-origin proposal,
and fail closed or clarify when the request cannot be handled safely.

The model must not output route-layer internals such as:

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

There is no route-classification mapping in the default runtime path.

```text
agent loop tool proposal
  -> allowlist/schema validation
  -> PolicyPort
  -> ToolGatewayPort
  -> typed ToolObservation
  -> same agent loop
```

The model may propose a tool call, but it never authorizes execution. Runtime
code owns:

- enabled tool registry and allowed tool names;
- argument schema validation and normalization;
- `Capability` and risk classes;
- policy and approval decisions;
- sensitivity ceilings and redaction;
- direct side effects and audit events.

In short: a model-origin tool proposal is not authorization.

## Classifier Experiments

Classifier and threshold experiments are historical/evaluation-only after this
decision. They may stay as fixtures that explain why Jarvis rejected a separate
classifier path, but they must not control production request handling.

Evaluation reports may still compare old router ideas for learning purposes:

```text
deterministic route coverage
embedding/non-LLM route precision
local LLM route classifier behavior
latency per experimental classifier layer
false live-state positives
```

These reports no longer define PM-09 readiness. PM-09 readiness depends on the
agent loop handling typed text and voice transcripts through the same request
lifecycle with safe tool execution gates.

## PM-08k Slice Contract

PM-08k is complete only when:

- PM-08k.0 research gate is complete;
- the architecture decision rejects both mandatory front-gate LLM classification
  and the Hybrid Request Resolver as runtime defaults;
- natural-language typed input enters the bounded agent loop by default;
- voice transcripts are documented to use the same lifecycle and loop;
- runtime model-route adjudication, route thresholds and route-schema parsing
  are removed from the production path;
- deterministic code is limited to control/safety/policy responsibilities;
- tool proposals cannot directly bypass PolicyPort, ToolGatewayPort, schemas,
  sensitivity ceilings, budgets or approvals;
- unsupported/risky tool attempts fail closed or ask for clarification through
  the agent loop;
- PM-09 is blocked until PM-08k is either implemented or explicitly rejected by
  an updated architecture decision.

## Tests First

Unit tests:

```text
test_default_request_path_uses_agent_loop_without_route_classifier
test_voice_transcript_uses_same_agent_loop_request_path
test_slash_commands_remain_client_controls_not_backend_routing
test_tool_proposal_requires_allowlisted_tool_name
test_tool_proposal_arguments_validate_before_execution
test_policy_denial_skips_tool_adapter_execution
test_unsupported_calendar_event_does_not_guess_date
test_mixed_natural_language_calculator_request_does_not_truncate_expression
```

Documentation/architecture tests:

```text
test_pm08k_docs_start_with_industry_research_gate
test_pm08k_research_gate_compares_mandatory_classifier_hybrid_router_and_agent_loop
test_pm08k_research_gate_records_source_links_and_architecture_decision
```

Architecture tests:

```text
test_runtime_default_does_not_import_model_route_classifier
test_loop_selector_does_not_execute_tools
test_agent_loop_tool_calls_go_through_toolgateway
test_cli_does_not_import_runtime_tool_adapters_or_route_classifiers
```

Expected red phase:

```text
PM-08k.0 industry research gate does not exist
runtime still wires model-route adjudicator before the agent loop
deterministic natural-language guards still select semantic tool routes
direct calculator path can truncate mixed natural-language arithmetic
unsupported event/date questions can bypass unresolved/clarification handling
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

1. Should any direct-answer optimization survive PM-08k, or should every
   natural-language request use the agent loop?
2. How should the agent loop present available tools without overloading local
   model context?
3. Which unsupported tool outcomes should ask clarification versus return a
   final unavailable answer?
4. Which historical classifier tests should be deleted, and which should remain
   as regression evidence for the rejected architecture?
