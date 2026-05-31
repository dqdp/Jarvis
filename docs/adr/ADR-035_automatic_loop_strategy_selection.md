# ADR-035 — Automatic Loop Strategy Selection

## Status

Accepted.

Date: 2026-05-30

## Context

Post-MVP Alpha now has the foundations needed for useful agent behavior:

```text
memory_augmented_answer
tool_react_loop
ToolGatewayPort
approval flow
project read-only shell tools
read-only system diagnostics tools
Project Docs RAG
```

However, the user-facing CLI should not require the user to understand internal
runtime strategies. A user should be able to type:

```text
check CPU temperature
look at the project docs about permissions
show what process is listening on this port
explain this architecture question
```

without explicitly saying:

```text
use tool_react_loop
use memory_augmented_answer
use this tool
use RAG
```

Modern agent harnesses usually hide this distinction. They either run a
tool-capable loop by default, or route through a graph/planner step that decides
whether a normal answer, retrieval, a tool call, a handoff or an approval is
needed.

Jarvis should follow the same user experience while keeping local-first safety,
ports/adapters boundaries and deterministic tests.

## Decision

Introduce automatic server-side loop selection.

The default user-facing mode becomes:

```text
auto
```

`auto` is not a third execution algorithm. It is a routing mode that resolves to
one concrete loop strategy before `AgentRuntime` invokes a loop:

```text
auto
  -> LoopStrategySelector
      -> memory_augmented_answer
      -> tool_react_loop
      -> future planner_executor_loop
```

The selector lives on the backend, not in the CLI. CLI and API clients may expose
explicit overrides for debugging or advanced use, but normal chat should not
require them.

## Selected approach

Use a backend selector with a replaceable intent-classification port:

```text
LoopStrategySelector
  -> IntentClassifierPort
  -> PolicyPort validation
  -> concrete LoopStrategy
```

PM-08a introduces the port and selector together. The first implementation used
in CI must be fake/deterministic so tests do not require a real LLM. Runtime
must also support a local structured model-backed classifier adapter behind the
same port. The deterministic classifier is a bootstrap/fallback adapter, not the
target routing mechanism.

This keeps the architecture from depending on keyword lists as the core routing
mechanism. Keyword or lexical hints may exist inside an initial deterministic
classifier adapter, but they are an implementation detail and a conservative
fallback, not the long-term decision boundary.

In other words, PM-08 must not implement a deterministic-only selector as the
target architecture. It implements `LoopStrategySelector + IntentClassifierPort`
first, then chooses the safest available classifier adapter for each runtime.
In local interactive runtime, the preferred adapter is the local structured
classifier when a structured local model profile is configured; otherwise the
runtime falls back to the conservative deterministic adapter.

The classifier may propose intent and candidate capabilities, but it must not
execute tools and must not grant permissions.

The selector pipeline is:

```text
1. honor explicit user/API override when present;
2. apply hard safety/config gates;
3. ask IntentClassifierPort for intent, confidence and candidate capabilities;
4. validate candidates against policy, enabled tools, budgets and model profiles;
5. choose concrete loop strategy or fail/ask/fallback according to confidence.
```

Before a request is persisted, the selected concrete loop must also be
executable under the active runtime budget. `tool_react_loop` is unavailable
when its runtime budget is missing, `allow_tools` is false, or
`max_tool_calls <= 0`; this must produce a redacted loop-selection failure
rather than an accepted request that fails later in runtime execution.

## Routing model

The final persisted request metadata payload is intentionally compact:

```text
requested_loop_mode
selected_loop_strategy
selected_model_profile, resolved after selected_loop_strategy is known
loop_selection_status
loop_selection_reason_code
loop_selection_confidence
loop_selection_intent_family
loop_selection_requires_tools
loop_selection_requires_live_state
loop_selection_policy_outcome
loop_selection_approval_possible
requested_model_profile
working_directory, execution-only and redacted from public API payloads
working_directory_scope
```

The redacted selection decision/event payload may additionally include:

```text
requested_mode
selected_loop_strategy
selected_model_profile, resolved after selected_loop_strategy is known
intent_family
reason_code
confidence
candidate_capabilities
requires_tools
requires_live_state
requires_approval_possible
answer_without_tools_would_be_misleading
fallback_behavior
```

Initial reason codes:

```text
explicit_memory_loop
explicit_tool_loop
tool_intent_project_read
tool_intent_system_diagnostics
tool_intent_safe_builtin
project_docs_question
ordinary_chat
tools_disabled_for_tool_intent
unknown_intent_default_chat
classifier_low_confidence
classifier_unavailable
```

The selected loop strategy is persisted in request metadata and emitted through
redacted audit/runtime events. Raw full prompts are not logged by default.

## Domain model

PM-08 introduces explicit domain objects for selection and classification.

### LoopSelectionMode

User-facing routing mode:

```text
auto
chat
tools
```

`invalid_override` is an internal audit-only mode used for rejected malformed
overrides. It is not accepted as a user-facing mode and never reaches runtime
execution.

`LoopSelectionMode` is not the same thing as `LoopStrategyName`.

### LoopSelectionRequest

Input to `LoopStrategySelector`:

```text
request_id
conversation_id
user_id
requested_mode
user_input
current_message_sensitivity
active_project_namespace
working_directory optional, for policy scope only
permission_mode
available_capabilities
available_tools_summary
runtime_budget_summary
model_profile_override optional
metadata
```

`user_input` is available to the selector/classifier as transient execution
input, but it must not be copied into selection events, long-lived audit payloads
or request metadata as a raw full prompt.

`working_directory` may be passed to `PolicyPort` for shell and system
diagnostics scope checks. It is not classifier evidence and must not be emitted
in loop-selection event payloads. It may be persisted in request metadata so the
accepted request can later execute tools with the same caller-provided scope,
but public request payloads must redact it and expose only scope presence.

The request lifecycle must not silently replace a missing `working_directory`
with the daemon current working directory. Tool-loop selection that requires
filesystem, shell or system-diagnostics scope must use an explicit caller
scope, or fail through policy with a redacted `request.loop_selection.failed`
event.

Pre-submit selection failures are correlated with a deterministic failure
request id derived from the client message id and a redacted payload
fingerprint. Accepted requests keep the stable request id derived from
`conversation_id + client_message_id`, so a failed attempt cannot poison a later
accepted retry's audit chain.

### IntentFamily

Constrained classifier intent:

```text
ordinary_chat
project_docs_question
project_inspection
system_diagnostics
safe_builtin_tool
code_execution
external_integration
planner_task
background_workflow
unknown
```

PM-08 only routes the implemented families:

```text
ordinary_chat
project_docs_question
project_inspection
system_diagnostics
safe_builtin_tool
unknown
```

Future families may exist in the enum so new tools can be planned without
changing the selector contract, but they must remain disabled until their
tool/capability slices exist.

### CapabilityCandidate

A classifier does not choose a concrete tool command. It proposes capability
candidates:

```text
capability
intent_family
confidence
requires_live_state
requires_execution
requires_write
tool_names
risk_classes
scope_hint
evidence_codes
```

`tool_names` are optional references to registered tools that may satisfy the
capability. They are not execution instructions.

`evidence_codes` are stable, non-sensitive labels such as:

```text
live_state_request
project_file_lookup
system_metric_request
documentation_question
code_execution_request
```

Do not store raw prompt snippets as evidence.

### IntentClassification

Output from `IntentClassifierPort`:

```text
intent_family
confidence
candidate_capabilities
requires_live_state
requires_execution
answer_without_tools_would_be_misleading
reason_code
fallback_preference
```

`fallback_preference` is one of:

```text
chat
fail_unavailable
ask_clarification
```

PM-08 may implement `chat` and `fail_unavailable` only. Clarification flow can
be added later when the CLI/API UX is ready.

### LoopSelectionDecision

Output from `LoopStrategySelector`:

```text
requested_mode
selected_loop_strategy
selected_model_profile optional, populated after selected_loop_strategy is known
intent_family
reason_code
confidence
candidate_capabilities
requires_tools
requires_live_state
policy_outcome
approval_possible
fallback_behavior
decision_status
```

`decision_status` is one of:

```text
selected
fallback_chat
rejected_by_policy
tools_unavailable
invalid_override
classifier_unavailable
```

The runtime may execute a loop only when `decision_status` is `selected` or
`fallback_chat`.

## Confidence model

Confidence is normalized to:

```text
0.0 <= confidence <= 1.0
```

Initial bands:

```text
high confidence:   >= 0.75
medium confidence: >= 0.45 and < 0.75
low confidence:    < 0.45
```

PM-08 default behavior:

```text
high confidence tool intent
  -> select tool_react_loop if policy/config allow

high confidence chat or project_docs_question
  -> select memory_augmented_answer

medium confidence
  -> conservative fallback to memory_augmented_answer

low confidence
  -> memory_augmented_answer with classifier_low_confidence reason
```

If `answer_without_tools_would_be_misleading=true` and tools are disabled or
denied, the selector must not silently fallback to chat as if live state was
checked. It returns `tools_unavailable` or `rejected_by_policy`.

Future aggressive routing may lower thresholds or use clarification, but it must
be a configuration change with tests, not hidden behavior. The presence of a
model-backed classifier does not by itself make routing aggressive; confidence,
policy and live-state fallback rules still govern the decision.

## IntentClassifierPort

`IntentClassifierPort` resolves user intent into a constrained domain shape.

Input:

```text
request text
requested mode
active project namespace
available intent families
available capabilities and tool descriptions
permission mode summary
```

Output:

```text
intent_family
requires_live_state
candidate_capabilities
confidence
answer_without_tools_would_be_misleading
reason_code
classification_source
```

`reason_code` is diagnostic evidence only. It must not encode trust or
provenance by string prefix. Selector decisions that depend on provenance must
use explicit source metadata such as:

```text
fake
deterministic
model
guardrail
fallback
```

Direct-execution eligibility is a separate policy/runtime decision. It must not
be inferred from `reason_code` and must not be granted solely because a model
classifier returned a candidate tool or scope hint.

Initial intent families:

```text
ordinary_chat
project_docs_question
project_inspection
system_diagnostics
safe_builtin_tool
code_execution later
external_integration later
unknown
```

Classifier implementations:

```text
FakeIntentClassifier
  deterministic test fixture for unit/contract/e2e tests

DeterministicIntentClassifier
  conservative local bootstrap/fallback using capability metadata and obvious
  hints

ModelBackedIntentClassifier / LocalStructuredIntentClassifier
  local model adapter returning strict JSON through ModelRouterPort; preferred
  runtime classifier when a local structured model profile is available
```

The selector must treat classifier output as advice. It remains responsible for
policy validation, disabled-tool handling and safe fallback.

The model-backed classifier contract is intentionally narrow:

```text
input:
  user request text
  allowed intent families
  available capability metadata
  permission-mode summary

output:
  constrained IntentClassification JSON only
```

It must not output raw shell commands, raw tool arguments, executable code or
provider-specific request dictionaries. Candidate `tool_names` are stable
registry references only. Diagnostic details such as `os_version`,
`cpu_overview` or `free_memory` are represented as stable `scope_hint` labels
and interpreted by deterministic planner/runtime code, not by ad hoc shell text
from the model.

If the local model call fails, times out or returns invalid JSON, the adapter
must fall back to a deterministic classifier when one is configured. If no
fallback is configured, it returns `classifier_unavailable`/`unknown` with
`fail_unavailable` preference rather than pretending live state was checked.

Because a small local classifier can still produce a false negative for live
local system-state requests, the adapter may apply a narrow post-classification
guardrail: when the model says ordinary chat but the request clearly asks for
current local machine/system state, the adapter may correct the result to
`system_diagnostics`. This guardrail is not the main selector architecture and
must remain category-level, policy-gated and covered by tests.

Routing quality is tracked with a multilingual tool-intent corpus. CI validates
the corpus schema and a deterministic/guardrail baseline. Real local model
coverage is an opt-in evaluation test and must not be required for CI.

## Direct tool execution and typed observations

Automatic routing may choose a fast direct-tool path for known, low-risk,
read-only questions such as:

```text
current time
OS version
battery charge
disk free space
VPN status
process name search
CPU overview
```

This direct path is a latency optimization, not a replacement for the normal
bounded ReAct loop. It must stay constrained:

- only allowlisted capabilities/tools/scopes may use direct execution;
- model-origin classifier output may propose candidate tools, but must not by
  itself grant direct execution;
- an explicit direct-scope allowlist may short-circuit obvious direct intents
  before the structured classifier call;
- ToolGateway and policy remain authoritative for execution.

Direct answers must not depend on an expanding set of command-output regexes in
`tool_react_loop`. Command output formats vary by OS version, locale and tool
implementation. Therefore the target design is:

```text
Tool adapter / normalizer
  -> typed ToolInvocationResult payload
  -> typed ToolObservation payload
  -> typed ToolObservationRef
  -> direct answer formatter
```

Typed observation v1 uses one shared contract across CLI, API, context events
and future UI surfaces:

```text
structured_content
structured_schema
structured_schema_version
parse_status
parse_warnings
```

`parse_status` values:

```text
parsed
partial
unparsed
not_applicable
```

Direct-answer behavior:

```text
parsed
  -> answer deterministically from structured_content

partial
  -> answer only from available typed fields and include a cautious warning

unparsed
  -> route bounded/redacted raw observation through normal ReAct/model analysis
     when policy and model budget allow; otherwise return a clear
     unparsed/unavailable result

not_applicable
  -> use the ordinary tool/ReAct path for tools that do not expose typed
     direct-answer contracts
```

Raw bounded stdout/stderr may still be kept for audit, debugging and ordinary
ReAct fallback, but it is not the primary direct-answer contract. If a tool
cannot provide typed data for the requested direct scope, Jarvis must either:

```text
route the bounded observation through the normal model/ReAct path when allowed;
or return a clear unparsed/unavailable result.
```

It must not hallucinate live state from unknown or unrecognized command output.
Adding a new diagnostics command should add adapter/normalizer fixture tests,
not a new command-specific parser branch inside the loop.

Diagnostics normalizers live near the adapters, not inside the loop runtime:

```text
tools/system_diagnostics/normalizers/os_version.py
tools/system_diagnostics/normalizers/battery.py
tools/system_diagnostics/normalizers/disk.py
tools/system_diagnostics/normalizers/vpn.py
tools/system_diagnostics/normalizers/process.py
tools/system_diagnostics/normalizers/cpu.py
tools/system_diagnostics/normalizers/sensors.py
```

The same provider-neutral schema is used across platforms:

```text
sw_vers / uname / os-release -> system.os_version v1
pmset / upower              -> system.battery_charge v1
df on macOS/Linux          -> system.disk_free v1
```

Process command lines, network evidence and similar host details are
sensitivity-aware fields. They should be omitted or redacted by default unless
the capability contract explicitly needs them and policy permits disclosure.

## Capability routing metadata

Every auto-routable capability/tool should expose routing metadata:

```text
capability
intent_families
description
requires_live_state
requires_execution
requires_write
risk_classes
positive_examples
negative_examples
default_selection_policy
```

This metadata is used by classifier implementations. It is not an authorization
source; policy still decides.

When adding a new tool such as a code sandbox, the implementation must add:

```text
capability: tool.code_sandbox.execute
intent_families: [code_execution]
requires_live_state: false
requires_execution: true
requires_write: false unless mounted write outputs are enabled
risk_classes: [compute_execution]
positive_examples:
  run this Python snippet
  execute this test and show output
  check this function with a sandboxed test
negative_examples:
  explain this code
  write an example but do not run it
default_selection_policy:
  developer_local: approval_required or allow only for read-only sandbox
  locked_down: approval_required or deny
  automation: deny by default
```

The selector should not need bespoke code for the code sandbox. It should see a
`code_execution` classification and candidate capability, then validate the
capability through policy and route to the appropriate tool-capable loop.

## Default behavior

### Ordinary chat

Requests that look like explanation, brainstorming, drafting, summarization or
general conversation use:

```text
memory_augmented_answer
```

This loop remains the reliable baseline and keeps:

```text
max_tool_calls = 0
```

### Project Docs RAG

Project documentation questions do not require `tool_react_loop`.

RAG is part of context assembly:

```text
memory_augmented_answer
  -> ContextAssembler
      -> ContentRetrievalPort
```

Examples:

```text
what does ADR-029 say about permissions?
summarize our PM-07 RAG design
where do docs describe shell sandbox rules?
```

The selector may tag these as `project_docs_question`, but the execution loop is
still `memory_augmented_answer` unless a live tool is also required.

### Live local inspection

Requests that need current host or project state use:

```text
tool_react_loop
```

Examples:

```text
check CPU temperature
show process status
what is listening on port 8080?
inspect the current git diff
look for where LoopStrategyName is defined
```

The loop still executes only through `ToolGatewayPort`; it cannot import shell,
diagnostics or tool adapters directly.

### Write-like or risky operations

Write-like requests are not silently executed by automatic routing.

Examples:

```text
reindex project docs
delete a memory
change a file
run a network command
```

The selector may identify required capabilities, but `PolicyPort` and approval
rules remain authoritative. If approval is required, the normal approval flow is
used. If the action is denied, the request fails with a clear policy result or
the assistant explains that it cannot perform the action.

### Tools disabled

If a request clearly requires tools but tools are disabled by policy, Jarvis must
not silently answer as if it had inspected live state.

Allowed outcomes:

```text
return a policy/configuration error;
or answer that live inspection is unavailable because tools are disabled.
```

Do not fallback to hallucinated diagnostics.

## Explicit overrides

API and CLI may expose explicit modes:

```text
auto
chat
tools
```

Mapping:

```text
auto  -> selector chooses concrete loop
chat  -> memory_augmented_answer
tools -> tool_react_loop
```

Explicit override is useful for debugging, tests and advanced users. It is not
the normal user path.

Overrides remain policy-gated:

- `tools` fails when tools are disabled;
- `tools` fails when the selected tool loop has no executable runtime budget;
- `chat` cannot execute tools;
- `auto` cannot bypass capability policy;
- `model_profile` override must still match the selected loop purpose.

## Boundary rules

Rules:

- CLI does not classify intent for safety-critical routing.
- API transport layer only accepts/validates requested mode; it does not own
  routing policy.
- `LoopStrategySelector` depends on domain schemas, settings, tool metadata and
  `IntentClassifierPort` output plus `PolicyPort`-level capability checks.
- `LoopStrategySelector` must not call model providers directly.
- `LoopStrategySelector` must not execute tools.
- `LoopStrategySelector` must not call storage adapters directly.
- `LoopStrategySelector` must not assemble prompts.
- `LoopStrategySelector` must not store memories or content chunks.
- `IntentClassifierPort` implementations must return constrained domain
  classifications, not free-form execution plans.
- `IntentClassifierPort` implementations must not execute tools or mutate state.
- `AgentRuntime` still executes only concrete loop strategies.
- `memory_augmented_answer` must not gain hidden tool behavior.
- `tool_react_loop` remains the only initial tool-capable user-turn loop.
- Provider-native tool calling is still normalized into domain tool proposals
  and executed through `ToolGatewayPort`.

## Relation to model routing

Loop selection and model routing are related but separate.

The selector chooses:

```text
selected_loop_strategy
```

The request metadata layer then resolves a model profile compatible with that
strategy:

```text
memory_augmented_answer -> chat profile
tool_react_loop         -> structured profile
```

No automatic cloud fallback is introduced. If the required local profile is not
available, the request fails clearly.

## Relation to RAG

RAG is not a tool-loop trigger by itself.

Project Docs RAG remains:

```text
ContextAssembler -> ContentRetrievalPort
```

This avoids turning every documentation question into a multi-step tool loop and
keeps citations/context manifests deterministic.

Future broad RAG or external source retrieval may add tools, but those source
adapters must be documented separately.

## Observability

Add redacted selection events:

```text
request.loop_selection.started
request.loop_selection.completed
request.loop_selection.failed
```

Minimum completed payload:

```text
request_id
conversation_id
requested_mode
selected_loop_strategy
selected_model_profile
reason_code
confidence
candidate_capabilities
policy_outcome
```

Do not include raw full prompt text.

## Alternatives considered

### Always run tool_react_loop

Rejected as the default.

It gives a simple user experience, but it makes ordinary chat more expensive
and brittle, especially with small local models. It also increases accidental
tool proposals and approval prompts.

`tools` remains available as an explicit override.

### Require users to choose loop_strategy

Rejected for normal UX.

The CLI would expose internal architecture concepts and users would need to know
whether RAG, tools or ordinary chat is appropriate.

Explicit override remains available for debugging.

### Let the model decide everything

Rejected.

A model-backed `IntentClassifierPort` adapter helps classify varied natural
language requests, but it must not be the authority that grants tool access,
bypasses policy or chooses unsafe fallback. It returns constrained intent data;
the backend selector and policy engine make the final decision. CI uses fake
model-router tests for this adapter and must not require a real LLM call.

### Put routing into CLI

Rejected.

Routing must be consistent across CLI, HTTP API, future voice, scheduler and
external clients.

## Testing requirements

PM-08 must be test-first.

Unit tests:

```text
test_selector_uses_intent_classifier_for_auto_mode
test_fake_intent_classifier_drives_selector_decision
test_auto_selects_memory_loop_for_ordinary_chat
test_auto_selects_memory_loop_for_project_docs_question
test_auto_selects_tool_loop_for_project_shell_read_intent
test_auto_selects_tool_loop_for_system_diagnostics_intent
test_auto_reports_tools_disabled_for_tool_intent
test_classifier_low_confidence_falls_back_to_chat
test_classifier_tool_intent_is_clamped_by_policy
test_explicit_chat_override_selects_memory_loop
test_explicit_tools_override_selects_tool_loop
test_selector_does_not_treat_rag_as_tool_loop
test_selector_outputs_reason_code_and_candidate_capabilities
```

Contract tests:

```text
test_message_without_loop_strategy_uses_auto_mode
test_auto_mode_persists_selected_loop_metadata
test_model_profile_matches_selected_loop
test_tools_disabled_does_not_silently_fallback_to_chat
```

CLI tests:

```text
test_cli_submit_message_omits_loop_strategy_for_default_auto
test_interactive_mode_command_switches_between_auto_chat_tools
test_cli_approval_prompt_can_approve_deny_and_cancel
test_cli_cancel_active_request_keeps_interactive_session_usable
test_cli_tool_flow_renders_action_approval_observation_without_raw_json_noise
```

Architecture tests:

```text
test_cli_does_not_import_loop_selector
test_selector_depends_on_intent_classifier_port_not_model_provider
test_selector_does_not_import_tool_adapters
test_selector_does_not_import_storage_adapters
test_intent_classifier_port_does_not_import_tool_adapters
test_memory_augmented_answer_still_does_not_import_toolgateway
```

E2E tests with fake providers:

```text
test_cli_plain_question_uses_memory_loop
test_cli_project_docs_question_uses_rag_without_tool_loop
test_cli_system_diagnostics_question_uses_tool_loop
test_cli_tool_intent_approval_flow_still_works
```

No test may require a real LLM call.

## Rollout plan

Implement PM-08 as ordered sub-slices:

```text
PM-08a Loop selection domain and selector contract
PM-08b API/request lifecycle auto mode
PM-08c CLI auto mode and mode controls
PM-08d CLI tool/RAG/approval readiness surface
```

1. Add domain/config representation for `auto`.
2. Add `IntentClassifierPort` and constrained intent domain objects.
3. Add fake and deterministic classifier implementations for tests and the first
   conservative runtime baseline.
4. Add `LoopStrategySelector` that consumes classifier output and validates it
   against policy/configuration.
5. Change request metadata resolution so missing `loop_strategy` means `auto`.
6. Persist selected concrete loop strategy and requested mode in request
   metadata.
7. Allocate a stable pre-submit request id from `conversation_id` and
   `client_message_id` for accepted requests, and a separate deterministic
   failure request id for rejected pre-submit selection attempts so failed
   attempts cannot collide with later accepted retries.
8. Add redacted selection events.
9. Add CLI/API explicit overrides for `auto`, `chat` and `tools`.
10. Add slash command `/mode auto|chat|tools` for interactive CLI.
11. Make CLI tool/RAG/approval rendering usable from the normal interactive
    flow.
12. Ensure `/cancel` and Ctrl-C leave the interactive session usable.
13. Keep `auto` as default for CLI and API.
14. Add local structured classifier adapter after the port, selector,
    observability and fallback behavior are green; keep fake/deterministic
    classifiers for CI and failure fallback.

## Deferred

- planner-executor routing;
- multi-agent handoff routing;
- remembered per-user routing preferences;
- UI for routing decisions;
- routing to background scheduler;
- routing to external integrations;
- cloud model fallback.
