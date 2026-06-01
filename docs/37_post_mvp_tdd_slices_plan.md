# 37 — Post-MVP TDD Slices Plan

## Status

Accepted planning baseline for the first post-MVP Alpha implementation slice.

Date: 2026-05-29

This document extends the MVP slice plan in
`docs/27_tdd_implementation_slices_plan.md`. It does not reopen MVP scope.

## 1. Purpose

Post-MVP work must remain TDD-first and architecture-preserving.

The immediate implementation path is:

```text
PM-01 Capability and policy foundation
PM-02 ToolGatewayPort with fake and safe tools
PM-03 LoopStrategy abstraction
PM-04 Safe-tool loop v1
PM-05 Approval model and CLI/API flow
PM-06a Project read-only shell tool
PM-06b Read-only system diagnostics tools
PM-07a Project Docs ingestion and citation index
PM-07b Project Docs retrieval and ContextAssembler integration
PM-08a Loop selection domain and selector contract
PM-08b API/request lifecycle auto mode
PM-08c CLI auto mode and mode controls
PM-08d CLI tool/RAG/approval readiness surface
PM-08e Model-backed intent classifier adapter
PM-08f Typed tool observations and direct-answer hardening
PM-08g Direct planner and capability routing registry cleanup
PM-08h Tool-intent corpus hardening and pre-voice corpus evaluation gate
PM-08i Interactive CLI shell UX hardening
PM-08j Canonical Jarvis runtime startup
PM-08k Agentic loop-first request handling cleanup
PM-08l Agent loop architecture hardening gate
PM-09 Voice gateway foundation
```

PM-01 through PM-09 are accepted in detail here. PM-08 is intentionally split
into smaller implementation sub-slices because voice must build on a working
CLI/API agent surface, not on a partially wired selector. Later slices should be
expanded before implementation.

## 2. Slice PM-01 — Capability and policy foundation

### Goal

Add the post-MVP capability and permission foundation from ADR-029 without
implementing tools, shell, RAG or new loop strategies.

After this slice, the system can answer:

```text
Is this capability allowed, denied or approval_required?
Which permission mode applies?
Which risk classes are involved?
Is the requested scope inside an allowed workspace?
Was the policy decision audited?
```

### Inputs

Required docs:

```text
docs/adr/ADR-029_capability_and_permission_model.md
docs/17_data_sensitivity_and_privacy_policy.md
docs/25_configuration_model.md
docs/26_testing_strategy.md
docs/36_post_mvp_plan_review.md
```

Related but out of PM-01 implementation scope:

```text
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/adr/ADR-031_agent_loop_strategy_architecture.md
```

### Tests first

Unit tests:

```text
test_permission_mode_developer_local_is_default
test_permission_modes_validate_known_values
test_unknown_capability_is_denied
test_safe_tool_capability_is_allowed
test_model_cloud_is_denied
test_secret_access_is_denied
test_shell_write_requires_approval
test_shell_destructive_is_denied
test_locked_down_requires_approval_for_shell_read
test_developer_local_allows_shell_read_inside_allowed_workspace
test_developer_local_denies_shell_read_outside_allowed_workspace
test_automation_denies_direct_memory_write
test_policy_decision_contains_stable_reason_and_scope
test_approval_required_outcome_contains_scoped_metadata
```

Contract tests:

```text
test_policy_port_evaluates_capability_request
test_config_policy_engine_returns_allow_deny_and_approval_required
test_capability_policy_uses_permission_mode
test_capability_policy_uses_sensitivity
test_capability_policy_uses_working_directory_scope
```

Audit tests:

```text
test_denied_capability_decision_emits_policy_capability_event
test_approval_required_capability_decision_emits_policy_capability_event
test_policy_capability_event_payload_is_redacted
```

Architecture tests:

```text
test_no_toolgateway_package_required_for_pm01
test_no_shell_adapter_package_required_for_pm01
test_runtime_does_not_import_tool_or_shell_adapters
```

### Expected red phase

The first test run should fail because:

```text
CapabilityPolicyRequest does not exist
PermissionMode does not exist
Capability / RiskClass domain values do not exist
PolicyPort has no evaluate_capability_request method
ConfigPolicyEngine does not evaluate capability policy
policy.capability.decision.recorded event is not defined/emitted
```

Bad red failures:

```text
test typo
wrong import path
real tool execution
real shell command execution
network dependency
```

### Implementation

Minimal production changes:

```text
domain/policy:
  Capability
  RiskClass
  PermissionMode
  CapabilityPolicyRequest
  PolicyDecision outcome allow/deny/approval_required for capability policy
  policy reason constants
  scoped decision metadata

ports/policy:
  evaluate_capability_request(...)

policy/engine:
  ConfigPolicyEngine capability rules
  permission mode defaults
  working_directory allowlist checks for tool.shell.read

config:
  permissions.mode
  permissions.modes.locked_down
  permissions.modes.developer_local
  permissions.modes.automation
  capabilities.tool.shell.read.allowed_roots
  capabilities.tool.shell.read.max_output_bytes
  capabilities.tool.shell.read.timeout_seconds

domain/events:
  policy.capability.decision.recorded

runtime/API wiring:
  only enough to audit policy decisions where PM-01 tests require it
```

### PM-01 default policy

`developer_local` is the Alpha default.

Expected outcomes:

```text
tool.safe -> allow
content.retrieve -> allow
context.inspect -> allow for non-secret manifest/ref data
tool.shell.read inside allowed root -> allow
tool.shell.read outside allowed root -> deny / outside_allowed_workspace
tool.shell.write in developer_local -> approval_required
tool.shell.write in locked_down -> deny
tool.shell.network -> deny
tool.shell.destructive -> deny
model.cloud -> deny
secret access -> deny
memory.write in automation direct autonomous context -> deny
```

### Acceptance criteria

PM-01 is complete only when:

```text
tests were added before production code;
red phase failed for missing capability/policy behavior;
all PM-01 unit, contract, audit and architecture tests are green;
no real tools are implemented;
no shell adapter is implemented;
no ToolGatewayPort implementation is required;
no RAG/content retrieval implementation is added;
no LoopStrategy extraction is performed;
architecture tests still pass;
existing MVP behavior remains green.
```

### Out of scope

Do not implement:

```text
ToolGatewayPort
tool registry
fake tools
safe tools
approval API
approval CLI UX
shell adapter
MCP
RAG/content ingestion
LoopStrategy abstraction
tool_react_loop
planner-executor
voice
cloud enablement
remembered approvals
```

### Architecture guardrails

Rules:

```text
AgentRuntime must not import tool/shell/MCP/RAG/voice adapters.
PolicyPort must remain domain-level and adapter-neutral.
ConfigPolicyEngine must not execute actions.
Working-directory policy must be a path/scope decision only.
No subprocess calls are allowed in PM-01.
No network calls are allowed in PM-01 tests.
```

## 3. Slice PM-02 — ToolGatewayPort with fake and safe tools

Depends on:

```text
ADR-030 accepted
PM-01 complete
```

### Goal

Introduce the ToolGateway execution boundary from ADR-030 with fake tools and a
minimal safe built-in tool set.

After this slice, callers can:

```text
list registered tools
inspect a tool spec
invoke fake and safe tools through ToolGatewayPort
receive normalized ToolObservation results
see policy/tool lifecycle audit events
verify output limits and timeout handling
```

### Inputs

Required docs:

```text
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/26_testing_strategy.md
docs/37_post_mvp_tdd_slices_plan.md
```

Related but out of PM-02 implementation scope:

```text
docs/adr/ADR-031_agent_loop_strategy_architecture.md
```

### Tests first

Unit tests:

```text
test_tool_spec_requires_name_capability_risk_and_schema
test_duplicate_tool_names_fail_registry_validation
test_disabled_tool_cannot_execute
test_unknown_tool_returns_failed_or_denied_domain_result
test_tool_arguments_validate_before_execution
test_tool_output_truncation_sets_truncated_true
test_tool_timeout_returns_timeout_observation
test_denied_policy_returns_denied_observation_without_adapter_execution
test_approval_required_returns_observation_without_adapter_execution
```

Contract tests:

```text
test_tool_gateway_lists_enabled_tools
test_tool_gateway_gets_tool_by_name
test_tool_gateway_invokes_fake_echo
test_tool_gateway_invokes_datetime_now
test_tool_gateway_invokes_calculator_evaluate
test_tool_gateway_invokes_daemon_status
test_tool_gateway_records_completed_lifecycle_events
test_tool_gateway_records_failed_lifecycle_events
test_tool_gateway_records_denied_lifecycle_events
```

Architecture tests:

```text
test_agent_runtime_does_not_import_tool_adapters
test_model_router_does_not_execute_tools
test_context_assembler_does_not_execute_tools
test_cli_api_do_not_import_concrete_tool_adapters
test_toolgateway_does_not_import_loop_strategies
```

E2E smoke:

```text
test_safe_tool_call_emits_policy_and_tool_lifecycle_events
test_denied_tool_call_emits_denied_event_and_skips_adapter_execution
test_approval_required_tool_call_does_not_execute_adapter
```

### Expected red phase

The first test run should fail because:

```text
ToolGatewayPort does not exist
ToolSpec / ToolCallRequest / ToolObservation do not exist
tool registry does not exist
fake tools do not exist
safe tools do not exist
tool lifecycle event types do not exist
ToolGateway does not call PolicyPort
ToolGateway does not write lifecycle events
```

Bad red failures:

```text
real shell execution
real network calls
real external service calls
tests depending on wall-clock exact value instead of injectable clock
tests bypassing PolicyPort to make invocation pass
```

### Implementation

Minimal production changes:

```text
domain/tools:
  ToolSpec
  ToolCallRequest
  ToolObservation
  ToolObservationStatus
  ToolRegistryEntry
  ToolCallId

ports/tools:
  ToolGatewayPort

tools/registry:
  in-process registry
  duplicate name validation
  enabled/disabled filtering

tools/gateway:
  policy evaluation before execution
  approval_required handling as ToolObservation
  denied handling as ToolObservation
  timeout/output limit enforcement
  lifecycle event emission

tools/fake:
  fake.echo
  fake.fail
  fake.timeout

tools/builtin:
  datetime.now
  calculator.evaluate
  daemon.status

domain/events:
  tool.call.requested
  tool.call.denied
  tool.call.started
  tool.call.completed
  tool.call.failed
  tool.call.timeout
  tool.call.cancelled
  tool.observation.recorded
```

### PM-02 safe tools

Included:

```text
fake.echo
fake.fail
fake.timeout
datetime.now
calculator.evaluate
daemon.status
```

Deferred:

```text
conversation.lookup
memory.lookup
context.inspect
shell
filesystem
MCP
network search
Telegram
Spotify
GitHub
```

### Tool result rules

PM-02 uses `ToolObservation` for normal tool outcomes:

```text
completed
denied
approval_required
failed
timeout
cancelled
```

Exceptions are reserved for programmer/configuration errors:

```text
invalid registry state
duplicate tool name at startup
adapter contract violation
invalid domain object construction
```

### Acceptance criteria

PM-02 is complete only when:

```text
tests were added before production code;
red phase failed for missing ToolGateway/tool behavior;
all PM-02 unit, contract, architecture and e2e smoke tests are green;
ToolGateway always calls PolicyPort before adapter execution;
denied and approval_required calls do not execute adapters;
tool lifecycle events are emitted;
outputs are bounded and truncation is represented;
no shell adapter is implemented;
no MCP/integration adapter is implemented;
no LoopStrategy extraction is performed;
no approval API/CLI UX is implemented;
existing MVP behavior remains green.
```

### Out of scope

Do not implement:

```text
shell execution
filesystem write tools
MCP gateway
Telegram/Spotify/GitHub integrations
network search
provider-native tool calling
approval endpoints
approval CLI UX
artifact storage
streaming tool output
tool_react_loop
LoopStrategy abstraction
planner-executor
RAG/content retrieval
```

### Architecture guardrails

Rules:

```text
AgentRuntime may depend on ToolGatewayPort only when a future tool-capable loop exists.
PM-02 must not wire ToolGateway into memory_augmented_answer.
Tool adapters must not call PolicyPort directly unless ToolGateway delegates classification explicitly.
Tool adapters must not write EventLog directly except through ToolGateway-managed hooks.
ModelRouter must not execute tools.
ContextAssembler must not execute tools.
CLI/API may list or invoke tools only through ToolGatewayPort, not concrete adapters.
```

## 4. Slice PM-03 — LoopStrategy abstraction

Depends on:

```text
ADR-031 accepted
PM-02 complete
```

### Goal

Extract the existing MVP deterministic workflow into an explicit loop strategy
without changing user-visible behavior.

After this slice:

```text
AgentRuntime selects a loop strategy.
memory_augmented_answer remains the default concrete strategy until PM-08 adds
user-facing auto selection.
memory_augmented_answer still has max_tool_calls=0.
The current MVP event chain remains compatible.
ToolGateway is still not used by the base loop.
```

### Inputs

Required docs:

```text
docs/adr/ADR-031_agent_loop_strategy_architecture.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/26_testing_strategy.md
docs/37_post_mvp_tdd_slices_plan.md
```

### Tests first

Unit tests:

```text
test_strategy_registry_selects_memory_augmented_answer_by_default
test_unknown_strategy_is_rejected
test_memory_augmented_answer_budget_keeps_max_tool_calls_zero
test_memory_augmented_answer_does_not_require_tool_gateway
test_loop_execution_request_requires_strategy_name_and_budget
test_loop_execution_result_reports_model_and_tool_call_counts
```

Contract tests:

```text
test_memory_augmented_answer_loop_preserves_existing_runtime_result
test_memory_augmented_answer_loop_preserves_existing_event_chain
test_agent_runtime_delegates_to_strategy_registry
test_strategy_uses_context_assembler_model_router_policy_and_stores_ports
```

Architecture tests:

```text
test_loop_strategies_do_not_import_storage_adapters
test_loop_strategies_do_not_import_provider_clients
test_memory_augmented_answer_loop_does_not_import_toolgateway
test_toolgateway_does_not_import_loop_strategies
```

E2E smoke:

```text
test_user_turn_lifecycle_still_uses_memory_augmented_answer
test_no_tool_events_are_emitted_for_memory_augmented_answer
```

### Expected red phase

The first test run should fail because:

```text
LoopStrategyName does not exist
LoopExecutionRequest does not exist
LoopExecutionResult does not exist
LoopBudget does not exist
LoopStrategyRegistry does not exist
MemoryAugmentedAnswerLoop does not exist as a separate strategy
AgentRuntime does not delegate to a strategy registry
```

Bad red failures:

```text
changed assistant response semantics
changed existing MVP event chain unexpectedly
ToolGateway required for memory_augmented_answer
real tool execution
provider-specific imports inside loop strategy
```

### Implementation

Minimal production changes:

```text
domain/loops:
  LoopStrategyName
  LoopExecutionRequest
  LoopExecutionResult
  LoopBudget
  LoopStatus

runtime/loops:
  LoopStrategy protocol/internal interface
  LoopStrategyRegistry
  MemoryAugmentedAnswerLoop

runtime/agent_runtime:
  select strategy
  delegate execution to MemoryAugmentedAnswerLoop
  preserve existing public AgentRuntime contract

domain/events:
  agent.loop.started
  agent.loop.completed
  agent.loop.failed
  agent.loop.cancelled
```

### Loop events in PM-03

PM-03 adds loop-level events only:

```text
agent.loop.started
agent.loop.completed
agent.loop.failed
agent.loop.cancelled
```

Deferred to PM-04:

```text
agent.step.started
agent.step.completed
agent.step.failed
```

Reason: step boundaries matter only once a loop can perform multiple
model/tool iterations.

### Acceptance criteria

PM-03 is complete only when:

```text
tests were added before production code;
red phase failed for missing loop strategy behavior;
existing MVP unit/contract/e2e behavior remains green;
memory_augmented_answer is selected by default in the PM-03 baseline;
memory_augmented_answer keeps max_tool_calls=0;
ToolGateway is not required by memory_augmented_answer;
no tool-capable loop is implemented;
no shell/MCP/RAG/integration behavior is added;
architecture tests still pass.
```

### Out of scope

Do not implement:

```text
tool_react_loop
tool proposal parsing
step-level events
planner-executor
ToolGateway use from memory_augmented_answer
shell
MCP
RAG
approval flow
background tasks
LangGraph adapter
graph checkpoints
```

### Architecture guardrails

Rules:

```text
AgentRuntime selects strategies but does not contain every loop algorithm.
Loop strategies use ports, not storage/provider/tool adapters.
memory_augmented_answer must not import ToolGatewayPort.
ToolGateway must not import or select loop strategies.
ContextAssembler remains responsible for prompt/context assembly.
ModelRouter remains responsible for provider-specific conversion.
```

## 5. Slice PM-04 — Safe-tool loop v1

Depends on:

```text
PM-01 complete
PM-02 complete
PM-03 complete
```

### Goal

Add the first bounded tool-capable loop strategy using only fake and safe tools.

After this slice, the runtime can:

```text
select a tool-capable strategy explicitly;
ask a model for a provider-neutral tool proposal;
parse the tool proposal;
execute the tool through ToolGatewayPort;
return the observation to the loop;
ask the model for a final answer;
stop safely on malformed actions, denied tools or budget exhaustion.
```

### Inputs

Required docs:

```text
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/adr/ADR-031_agent_loop_strategy_architecture.md
docs/37_post_mvp_tdd_slices_plan.md
```

### Tests first

Unit tests:

```text
test_tool_react_loop_requires_toolgateway
test_tool_react_loop_budget_requires_positive_step_and_tool_limits
test_tool_react_loop_rejects_unknown_tool_name
test_tool_react_loop_rejects_malformed_tool_proposal
test_tool_react_loop_stops_on_max_steps
test_tool_react_loop_stops_on_max_tool_calls
test_tool_react_loop_stops_on_consecutive_tool_failures
test_tool_react_loop_does_not_persist_tool_observation_as_message
```

Contract tests:

```text
test_tool_react_loop_executes_fake_tool_then_final_answer
test_tool_react_loop_executes_datetime_tool_then_final_answer
test_tool_react_loop_handles_denied_tool_observation
test_tool_react_loop_handles_approval_required_observation_without_execution
test_tool_react_loop_records_step_events
test_tool_react_loop_records_tool_observation_refs
```

Golden/context tests:

```text
test_tool_observation_ref_can_enter_context_as_tool_section
test_tool_observation_context_respects_budget
test_tool_observation_context_excludes_secret_observation
```

Architecture tests:

```text
test_tool_react_loop_uses_toolgateway_port_not_adapters
test_tool_react_loop_does_not_import_shell_mcp_or_integration_adapters
test_model_router_does_not_execute_tools
test_toolgateway_does_not_select_loop_strategy
```

E2E smoke:

```text
test_safe_tool_loop_user_turn_with_fake_tool_completes
test_safe_tool_loop_malformed_tool_request_fails_safely
test_safe_tool_loop_budget_exhaustion_fails_safely
```

### Expected red phase

The first test run should fail because:

```text
tool_react_loop does not exist
ToolProposal does not exist
tool proposal parser does not exist
agent.step.* events are not defined/emitted
ContextAssembler cannot include tool observation refs
LoopStrategyRegistry cannot select a tool-capable loop
```

Bad red failures:

```text
real shell execution
MCP/network/external service calls
ToolGateway bypassed
tool observations persisted as conversation messages
unbounded loop behavior
```

### Implementation

Minimal production changes:

```text
domain/loops:
  ToolProposal
  ToolProposalParseError
  ToolObservationRef
  LoopStep
  LoopStepStatus

runtime/loops:
  ToolReactLoop or equivalent first tool-capable strategy
  tool proposal parser for internal/fake structured model output
  budget and stopping condition enforcement
  denied/approval_required observation handling

context_assembly:
  tool observation refs as a bounded context section

domain/events:
  agent.step.started
  agent.step.completed
  agent.step.failed
```

### Tool proposal format

PM-04 should use a provider-neutral internal shape. It should not depend on
OpenAI/Ollama/provider-native tool calling.

Example logical shape:

```text
action: final_answer | tool_call
tool_name optional
arguments optional
final_answer optional
```

The exact implementation can use existing structured model support and fake
model providers.

### PM-04 allowed tools

Allowed:

```text
fake.echo
fake.fail
fake.timeout
datetime.now
calculator.evaluate
daemon.status
```

Denied/out of scope:

```text
shell
filesystem
MCP
network search
Telegram
Spotify
GitHub
memory.lookup
conversation.lookup
context.inspect
```

### Budget defaults

Initial safe-tool loop budget:

```text
max_steps: 4
max_model_calls: 4
max_tool_calls: 2
max_consecutive_failures: 1
max_wall_time_seconds: 60
```

These are starting values for Alpha tests. They may move to config in the
implementation slice if the existing runtime budget model makes that natural.

### Acceptance criteria

PM-04 is complete only when:

```text
tests were added before production code;
red phase failed for missing tool-capable loop behavior;
tool_react_loop or equivalent strategy is explicit and bounded;
all tool execution goes through ToolGatewayPort;
fake/safe tool e2e smoke passes;
malformed tool proposals fail safely;
budget exhaustion has deterministic safe terminal behavior;
tool observations are not normal conversation messages;
step events are emitted;
existing memory_augmented_answer behavior remains green;
architecture tests still pass.
```

### Out of scope

Do not implement:

```text
shell
filesystem tools
MCP
network search
Telegram/Spotify/GitHub
approval API/CLI UX
remembered approvals
planner-executor
RAG/content retrieval
artifact storage
provider-native tool calling
LangGraph adapter
graph checkpoints
```

### Architecture guardrails

Rules:

```text
ToolReactLoop uses ToolGatewayPort, not concrete adapters.
ToolGateway does not know about ToolReactLoop.
ModelRouter may return structured tool proposals but must not execute tools.
ContextAssembler may include tool observation refs but must not execute tools.
Tool observations remain runtime/context artifacts, not user-facing messages by default.
memory_augmented_answer remains max_tool_calls=0.
```

## 6. Slice PM-05 — Approval model and CLI/API flow

Depends on:

```text
PM-01 complete
PM-02 complete
PM-04 complete
```

### Goal

Add one-shot approval requests for actions whose policy outcome is
`approval_required`.

PM-05 uses a non-blocking approval model:

```text
ToolGateway returns approval_required ToolObservation.
Loop/CLI presents the approval request.
User grants or denies through HTTP/CLI.
Loop retries the tool call with approval_id if granted.
ToolGateway executes only after validating the granted approval.
```

ToolGateway must not block indefinitely waiting for approval.

### Inputs

Required docs:

```text
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/37_post_mvp_tdd_slices_plan.md
```

### Tests first

Unit tests:

```text
test_approval_request_requires_capability_risk_scope_and_expiry
test_approval_grant_changes_pending_to_granted
test_approval_deny_changes_pending_to_denied
test_approval_cancel_changes_pending_to_cancelled
test_expired_approval_cannot_be_granted
test_denied_approval_cannot_be_reused
test_granted_approval_cannot_be_reused_for_different_tool_call
test_approval_scope_must_match_tool_call_and_capability
test_empty_cli_approval_input_denies
test_yes_cli_approval_input_grants
```

Contract tests:

```text
test_approval_store_creates_pending_approval
test_approval_store_gets_approval_by_id
test_approval_store_grants_pending_approval
test_approval_store_denies_pending_approval
test_approval_store_expires_stale_approvals
test_toolgateway_returns_approval_required_without_execution
test_toolgateway_executes_after_matching_granted_approval
test_toolgateway_rejects_expired_or_mismatched_approval
```

API tests:

```text
test_get_approval_returns_redacted_payload
test_grant_pending_approval
test_deny_pending_approval
test_grant_expired_approval_returns_conflict
test_deny_already_granted_approval_returns_conflict
```

CLI tests:

```text
test_cli_renders_approval_prompt
test_cli_empty_approval_input_denies
test_cli_yes_approval_input_grants
test_cli_ctrl_c_denies_or_cancels_local_wait
test_cli_reports_expired_approval
```

E2E smoke:

```text
test_approval_required_tool_call_does_not_execute
test_granted_approval_allows_retry_execution
test_denied_approval_prevents_execution
test_expired_approval_prevents_execution
test_approval_events_are_emitted
```

### Expected red phase

The first test run should fail because:

```text
ApprovalRequest does not exist
ApprovalStorePort does not exist
approval lifecycle statuses do not exist
approval API routes do not exist
CLI approval prompt does not exist
ToolGateway cannot validate approval_id
approval lifecycle events are not emitted
```

Bad red failures:

```text
remembered approvals
WebSocket requirement
blocking ToolGateway wait
real shell execution
external side effects
raw secret payloads in approval events
```

### Implementation

Minimal production changes:

```text
domain/approvals:
  ApprovalRequest
  ApprovalStatus pending/granted/denied/expired/cancelled
  ApprovalDecision
  ApprovalScope

ports/approvals:
  ApprovalStorePort

storage:
  approval persistence
  expiry query/update

api:
  GET /v1/approvals/{approval_id}
  POST /v1/approvals/{approval_id}/grant
  POST /v1/approvals/{approval_id}/deny

cli:
  render approval prompt
  y/yes grants
  n/no/empty denies
  Ctrl-C denies or cancels local wait

tools/gateway:
  create/return approval_required observation
  validate granted approval_id on retry
  reject denied/expired/mismatched approval

domain/events:
  approval.required
  approval.granted
  approval.denied
  approval.expired
  approval.cancelled
```

### Approval lifecycle

Statuses:

```text
pending
granted
denied
expired
cancelled
```

Events:

```text
approval.required
approval.granted
approval.denied
approval.expired
approval.cancelled
```

Default TTL:

```text
5 minutes
```

PM-05 approvals are one-shot and scoped to one action/tool call.

### API shape

Initial endpoints:

```http
GET  /v1/approvals/{approval_id}
POST /v1/approvals/{approval_id}/grant
POST /v1/approvals/{approval_id}/deny
```

Response payloads must be redacted and must not include raw secrets or raw full
prompts.

### CLI UX

Initial prompt:

```text
approval> <capability> wants to perform <redacted_summary>
approve? [y/N]
```

Rules:

```text
y / yes -> grant
n / no / empty -> deny
Ctrl-C -> deny or cancel local wait
timeout -> expired
```

Default is deny.

### Acceptance criteria

PM-05 is complete only when:

```text
tests were added before production code;
red phase failed for missing approval behavior;
approval_required tool calls do not execute;
granted approval allows retry execution with matching approval_id;
denied approval prevents execution;
expired approval prevents execution;
approval payloads and events are redacted;
CLI can grant/deny an approval;
HTTP approval endpoints work;
no remembered approvals are implemented;
no WebSocket is required;
existing PM-04 safe-tool loop remains green.
```

### Out of scope

Do not implement:

```text
remembered approvals
approval rules UI
WebSocket/control channel
multi-user approval routing
role-based access control
mobile push
secret manager
destructive shell
external irreversible side effects
cloud enablement
```

### Architecture guardrails

Rules:

```text
Approval is a domain lifecycle, not an ad hoc CLI prompt.
ToolGateway returns approval_required instead of blocking indefinitely.
Loop/CLI retries with approval_id after grant.
Approvals are one-shot in PM-05.
Approval payloads are redacted.
Approval decisions are auditable.
```

## 7. Slice PM-06a — Project read-only shell tool

Depends on:

```text
PM-01 complete
PM-02 complete
PM-05 complete
ADR-033 accepted
```

### Goal

Add a safe project inspection tool behind `ToolGatewayPort`.

PM-06a lets an agent inspect the current project without giving it arbitrary
terminal access.

Allowed first command families:

```text
pwd
ls
rg
sed -n
head
tail
wc
git status
git diff
git show
git log
git branch
git ls-files
```

PM-06a does not include write shell, network clients, interpreters, package
managers, build tools, pipes, redirects or arbitrary shell strings.
`git status` and `git ls-files` require an explicit safe file pathspec after
`--` to avoid leaking stale tracked secret-like filenames from git metadata.
Direct reads of `.git` metadata through generic file readers are denied.
Executables are resolved through a minimal PATH and symlink targets must stay
under trusted runtime roots.
Non-recursive `ls` classification checks only entries that the requested
listing can disclose; recursive directory readers still scan descendants.

### Inputs

Required docs:

```text
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/adr/ADR-033_shell_sandbox_and_local_command_policy.md
docs/37_post_mvp_tdd_slices_plan.md
```

### Tests first

Unit tests:

```text
test_allows_pwd_inside_workspace
test_allows_ls_inside_workspace
test_allows_bare_ls_at_workspace_root_with_git_metadata
test_denies_ls_when_requested_listing_exposes_secret_like_entry
test_allows_rg_inside_workspace
test_allows_sed_n_with_bounded_range
test_allows_head_and_tail_with_bounded_line_count
test_allows_wc_inside_workspace
test_allows_git_status_short_with_explicit_safe_pathspec
test_allows_git_diff_read_only
test_allows_git_show_read_only
test_denies_shell_metacharacters
test_denies_path_outside_workspace
test_denies_symlink_escape_outside_workspace
test_denies_directory_with_symlink_descendant_to_secret_path
test_denies_secret_like_paths
test_denies_git_metadata_paths_for_direct_file_readers
test_denies_git_index_listing_without_explicit_safe_pathspec
test_denies_git_index_directory_pathspecs
test_shell_classification_redacts_secret_like_denied_cwd
test_subprocess_executor_rejects_allowlisted_dir_symlink_to_untrusted_target
test_subprocess_executor_executes_resolved_symlink_target
test_denies_write_commands
test_denies_network_commands
test_denies_interpreters
test_denies_git_write_subcommands
test_denies_package_build_runtime_managers
```

Contract tests:

```text
test_shell_read_tool_executes_allowed_argv_command
test_shell_read_tool_returns_bounded_stdout
test_shell_read_tool_returns_bounded_stderr
test_shell_read_tool_truncates_large_output_with_metadata
test_shell_read_tool_redacts_secret_like_denied_cwd_in_shell_events
test_shell_read_tool_redacts_secret_like_output_names
test_shell_read_tool_redacts_secret_like_certificate_output_names
test_shell_read_tool_observation_is_at_least_project_sensitivity
test_shell_read_tool_approval_required_observation_is_at_least_project_sensitivity
test_shell_read_tool_times_out
test_shell_read_tool_emits_classified_started_completed_events
test_shell_read_tool_emits_denied_event_without_execution
test_shell_read_tool_uses_minimal_environment
```

Architecture tests:

```text
test_agent_runtime_does_not_import_subprocess
test_loop_strategies_do_not_import_subprocess
test_only_shell_adapter_executes_subprocess
test_toolgateway_consults_policy_before_shell_execution
```

E2E smoke with fake model/tool request:

```text
test_agent_can_use_project_rg_tool_and_answer_with_observation
test_agent_cannot_use_denied_shell_command
test_shell_denial_is_returned_as_tool_observation
```

### Expected red phase

The first test run should fail because:

```text
ShellCommandClassifier does not exist
ShellReadTool does not exist
shell command policy config does not exist
allowlisted cwd validation does not exist
secret-like path denial does not exist
bounded subprocess adapter does not exist
shell audit events are not emitted
ToolGateway has no shell.read project tool
```

Bad red failures:

```text
test executes a dangerous real command
test depends on host-specific state
test reads secrets
test uses network
test bypasses ToolGateway
test weakens architecture boundaries
```

### Implementation

Minimal production changes:

```text
domain/tools/shell_policy:
  ShellCommand
  ShellCommandClassification
  ShellCommandDecision
  ShellCommandFamily
  ShellOutputLimits

ports/tools:
  ShellExecutorPort or adapter-private equivalent

tools/shell:
  ProjectShellReadTool
  ShellCommandClassifier
  bounded argv subprocess adapter

config:
  allowlisted shell roots
  allowed project command families
  output/time limits
  secret path patterns

ToolGateway registry:
  tool.shell.read.project

events:
  tool.shell.classified
  tool.shell.denied
  tool.shell.started
  tool.shell.completed
  tool.shell.failed
  tool.shell.timeout
  tool.shell.output_truncated
```

### Command execution rules

PM-06a rules:

```text
argv only
no shell strings
no pipes
no redirects
no command separators
no subshells
no environment assignment prefixes
cwd must be inside allowlisted project root
paths must remain inside allowlisted project root
secret-like paths are denied
output is bounded
wall-clock time is bounded
environment is minimal
```

Denied command families:

```text
write/destructive filesystem commands
network clients
interpreters
package managers
build tools
git write/state-changing subcommands
interactive terminal tools
```

### Permission mode behavior

Expected policy behavior:

```text
developer_local:
  tool.shell.read.project -> allow inside allowlisted roots

locked_down:
  tool.shell.read.project -> approval_required

automation:
  tool.shell.read.project -> allow only for configured roots and command families
```

Write shell remains unavailable in PM-06a under every mode.

### Acceptance criteria

PM-06a is complete only when:

```text
tests were added before production code;
red phase failed for missing shell policy/tool behavior;
allowed read-only commands execute through ToolGateway;
denied commands return denied ToolObservation without execution;
cwd and paths cannot escape allowlisted roots;
secret-like paths are denied;
stdout/stderr are bounded and truncation metadata is returned;
timeouts work;
shell events are emitted without raw secrets or unbounded output;
AgentRuntime and LoopStrategy do not import subprocess;
ToolGateway consults PolicyPort before execution;
existing PM-04/PM-05 safe-tool and approval flows remain green.
```

### Out of scope

Do not implement:

```text
write shell
destructive shell
arbitrary bash/zsh
interactive terminal sessions
network clients
package managers
build tools
remote execution
system diagnostics
MCP gateway
container sandbox
secret manager access
```

System diagnostics are PM-06b.

## 8. Slice PM-06b — Read-only system diagnostics tools

Depends on:

```text
PM-06a complete
ADR-033 accepted
```

### Goal

Add curated read-only diagnostics tools behind `ToolGatewayPort`.

PM-06b lets an agent inspect local runtime health without arbitrary shell
access. It is intended for debugging local model servers, daemon processes,
resource pressure, storage pressure, network listener state and thermal
pressure.

PM-06b adds these capabilities:

```text
tool.system.read.process
tool.system.read.resources
tool.system.read.hardware
tool.system.read.network
tool.system.read.sensors
```

### Inputs

Required docs:

```text
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/adr/ADR-033_shell_sandbox_and_local_command_policy.md
docs/37_post_mvp_tdd_slices_plan.md
```

### Tests first

Unit tests:

```text
test_allows_ps_snapshot
test_allows_pgrep
test_allows_uptime
test_allows_df
test_allows_du_inside_workspace
test_denies_du_outside_workspace
test_denies_secret_like_cwd
test_denies_du_secret_like_path
test_allows_macos_top_snapshot
test_allows_macos_vm_stat
test_allows_macos_sysctl_selected_keys
test_allows_linux_top_batch_snapshot
test_allows_linux_free
test_allows_linux_lscpu
test_allows_linux_lshw
test_allows_network_diagnostics_selected_flags
test_denies_linux_ifconfig_to_match_platform_allowlist
test_denies_interactive_diagnostics
test_denies_kill_sudo_and_system_mutations
test_denies_network_clients
test_allows_temperature_sensor_snapshot
test_temperature_snapshot_normalizes_celsius_when_possible
test_temperature_source_unavailable_is_non_fatal
test_denies_sudo_powermetrics
test_denies_sensor_write_paths
test_denies_long_running_sensor_polling
test_gpu_temperature_uses_nvidia_smi_query_mode
test_platform_specific_classifier_is_deterministic
```

Contract tests:

```text
test_system_diagnostics_tool_executes_allowed_argv_command
test_system_diagnostics_tool_returns_bounded_stdout
test_system_diagnostics_tool_truncates_large_output_with_metadata
test_system_diagnostics_tool_times_out
test_system_diagnostics_tool_redacts_process_command_line_secrets
test_system_diagnostics_tool_redacts_auth_flags_and_key_values
test_system_diagnostics_tool_redacts_network_sensitive_output
test_system_diagnostics_tool_redacts_credential_urls
test_sensor_command_stdout_returns_normalized_snapshot
test_nvidia_smi_temperature_query_returns_sensor_snapshot
test_powermetrics_permission_required_returns_unavailable_snapshot
test_sensor_backend_unavailable_returns_unavailable_observation
test_sensor_backend_snapshot_is_normalized
test_system_diagnostics_tool_emits_audit_events
test_system_diagnostics_tool_returns_denied_observation_without_execution
```

Architecture tests:

```text
test_agent_runtime_does_not_import_diagnostics_adapters
test_loop_strategies_do_not_import_diagnostics_adapters
test_only_shell_or_diagnostics_adapter_executes_subprocess
test_toolgateway_consults_policy_before_diagnostics_execution
```

E2E smoke with fake diagnostics adapters:

```text
test_agent_can_use_process_snapshot_and_answer_with_observation
test_agent_can_use_temperature_snapshot_when_available
test_agent_handles_unavailable_temperature_backend
test_agent_cannot_use_denied_diagnostics_command
```

### Expected red phase

The first test run should fail because:

```text
SystemDiagnosticsClassifier does not exist
SystemDiagnosticsTool does not exist
sensor diagnostics capability does not exist
platform-specific diagnostics policy does not exist
diagnostics redaction does not exist
temperature unavailable observation does not exist
ToolGateway has no system diagnostics tools
```

Bad red failures:

```text
test requires real host-specific sensor hardware
test requires sudo
test runs an interactive command
test mutates process or system state
test depends on live network access
test stores raw process command lines with secrets
```

### Implementation

Minimal production changes:

```text
domain/tools/system_diagnostics:
  SystemDiagnosticsCommand
  SystemDiagnosticsClassification
  SystemDiagnosticsDecision
  SystemDiagnosticsFamily
  SensorReading
  SensorSnapshot
  DiagnosticsOutputLimits

tools/system_diagnostics:
  SystemDiagnosticsClassifier
  ProcessDiagnosticsTool
  ResourceDiagnosticsTool
  HardwareDiagnosticsTool
  NetworkDiagnosticsTool
  SensorDiagnosticsTool
  bounded argv diagnostics adapter
  read-only thermal sysfs adapter for Linux

config:
  enabled diagnostics families
  selected platform command flags
  diagnostics output/time limits
  redaction patterns

ToolGateway registry:
  tool.system.read.process
  tool.system.read.resources
  tool.system.read.hardware
  tool.system.read.network
  tool.system.read.sensors

events:
  tool.system.diagnostics.classified
  tool.system.diagnostics.denied
  tool.system.diagnostics.started
  tool.system.diagnostics.completed
  tool.system.diagnostics.failed
  tool.system.diagnostics.timeout
  tool.system.diagnostics.output_truncated
  tool.system.diagnostics.unavailable
```

### Initial diagnostics command families

```text
ps
pgrep
uptime
df
du inside allowlisted workspace roots
macOS: top -l 1, vm_stat, sysctl selected keys, netstat, ifconfig, lsof
Linux: top -b -n 1, free, lscpu, lshw, ss/netstat, ip addr, lsof
```

Temperature and sensor diagnostics:

```text
macOS: powermetrics --samplers smc -n 1 if available without sudo
Linux: sensors
Linux: read-only /sys/class/thermal/thermal_zone*/temp adapter
Linux GPU temperature: nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

Sensor readings should normalize to Celsius when possible and include source
metadata and timestamp.

Guardrails:

```text
htop/watch/less/vim are denied by default
sudo and privilege escalation are denied
kill/renice/launchctl/systemctl mutations are denied
network diagnostics are read-only but privacy-sensitive
process command lines are truncated/redacted
environment variables are not returned
diagnostics output is bounded
temperature diagnostics are one-shot snapshots
fan control and power profile changes are denied
sensor write paths are denied
all attempts are audited
```

### Permission mode behavior

Expected policy behavior:

```text
developer_local:
  tool.system.read.process -> allow
  tool.system.read.resources -> allow
  tool.system.read.hardware -> allow
  tool.system.read.network -> allow with redaction and output caps
  tool.system.read.sensors -> allow with no privilege escalation

locked_down:
  all tool.system.read.* -> approval_required

automation:
  only explicitly configured diagnostics families
```

### Acceptance criteria

PM-06b is complete only when:

```text
tests were added before production code;
red phase failed for missing diagnostics behavior;
allowed diagnostics execute through ToolGateway;
denied diagnostics return denied ToolObservation without execution;
interactive diagnostics are denied;
mutating process/system commands are denied;
network diagnostics output is redacted and bounded;
process command lines are truncated/redacted;
environment variables are not returned;
temperature sensor snapshots work when available;
missing sensor backend is non-fatal and returns unavailable;
sudo and privilege escalation are denied;
sensor writes and fan/power mutations are denied;
timeouts work;
diagnostics events are emitted without raw secrets or unbounded output;
existing PM-06a shell policy remains green.
```

### Out of scope

Do not implement:

```text
interactive monitoring
long-running polling
process mutation
system service mutation
sudo or password prompts
fan control
power profile changes
network clients
remote diagnostics
write shell
```

## 9. Slice PM-07a — Project Docs ingestion and citation index

Depends on:

```text
PM-01 complete
ADR-034 accepted
```

Recommended order:

```text
after PM-06b
```

### Goal

Add the source registry, deterministic markdown chunking and citation index for
Project Docs RAG.

PM-07a creates the Content Retrieval ingestion foundation without adding
retrieval into model context yet.

Initial corpus:

```text
README.md
docs/*.md
docs/adr/*.md
```

### Inputs

Required docs:

```text
docs/20_post_mvp_rag_content_retrieval.md
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-034_content_retrieval_subsystem_and_project_docs_rag.md
docs/37_post_mvp_tdd_slices_plan.md
```

### Tests first

Unit tests:

```text
test_project_docs_source_allowlist_matches_readme_docs_and_adrs
test_secret_like_paths_are_not_ingested
test_secret_like_paths_include_common_key_material_names
test_secret_like_paths_include_separator_variants
test_secret_like_paths_include_space_and_backslash_variants
test_secret_like_content_is_not_ingested
test_secret_like_content_includes_quoted_assignment_keys
test_secret_like_content_includes_common_key_naming_variants
test_secret_like_content_includes_inline_object_assignments
test_secret_like_content_includes_markdown_decorated_assignments
test_secret_like_content_allows_schema_fields_that_mention_tokens
test_symlink_to_secret_like_path_is_not_ingested
test_symlinked_docs_directory_is_not_ingested
test_symlinked_docs_directory_adr_child_is_not_ingested
test_symlinked_adr_directory_is_not_ingested
test_markdown_chunker_splits_by_headings
test_markdown_chunker_splits_oversized_sections
test_chunk_preserves_heading_path
test_markdown_chunker_ignores_headings_inside_fenced_code
test_markdown_chunker_closes_fence_only_with_matching_marker
test_markdown_chunker_does_not_close_fence_on_marker_prefix_with_text
test_chunk_preserves_line_range_when_possible
test_citation_formats_path_and_line_range
test_changed_source_marks_old_chunks_stale
test_deleted_source_marks_chunks_deleted_or_stale
```

Integration tests:

```text
test_source_registry_creates_project_doc_sources
test_source_registry_updates_content_hash_on_change
test_ingestion_creates_content_sources_and_chunks
test_ingestion_does_not_persist_secret_like_content
test_reingestion_marks_old_chunks_stale
test_reingestion_reactivates_chunks_when_content_reverts
test_reingestion_recovers_after_failed_atomic_sync
test_failed_source_chunk_sync_does_not_publish_new_source_hash
test_real_source_chunk_sync_rolls_back_after_chunk_failure
test_source_chunk_sync_rejects_cross_source_chunks
test_source_chunk_sync_rejects_chunks_with_mismatched_source_metadata
test_source_chunk_sync_rejects_secret_sensitivity_content
test_unchanged_reingestion_does_not_churn_chunks
test_unchanged_reingestion_refreshes_source_last_seen
test_deleted_source_resurrection_reactivates_existing_chunks
test_deleted_source_marks_source_and_chunks_deleted_or_stale
test_project_docs_delete_pass_does_not_delete_other_content_corpora
test_content_tables_are_separate_from_memory_tables
```

Architecture tests:

```text
test_memory_subsystem_does_not_import_content_retrieval_storage
test_content_retrieval_does_not_write_memory_tables
```

### Expected red phase

The first test run should fail because:

```text
ContentSource does not exist
ContentChunk does not exist
ContentCitation does not exist
content source registry does not exist
markdown chunker does not exist
content storage tables do not exist
```

Bad red failures:

```text
test stores document chunks in memory tables
test uses real embedding/model provider
test indexes arbitrary files
test indexes source code/logs/PDFs/web pages
test requires pgvector
test touches ContextAssembler integration
```

### Implementation

Minimal production changes:

```text
domain/content_retrieval:
  ContentSource
  ContentChunk
  ContentCitation
  ContentSourceStatus active/stale/deleted/failed
  ContentChunkStatus active/stale/deleted

storage:
  content_sources
  content_chunks
  project docs ingestion adapter

ingestion:
  allowlisted source scanner
  secret-like path/content denial
  deterministic markdown heading chunker
  citation builder
  content_hash change detection
  stale/deleted source handling

event type contract:
  PM-07a defines event types only; ingestion EventLogPort emission is deferred.
  Event emission remains required before production observability, but it does
  not block PM-07b retrieval/context integration.
  content.source.discovered
  content.source.ingested
  content.source.updated
  content.source.deleted
  content.chunk.created
  content.chunk.stale
```

### Source registry

Initial source fields:

```text
source_id
source_type readme/project_doc/adr
uri/path
title
content_hash
last_seen_at
indexed_at
status active/stale/deleted/failed
sensitivity
metadata
```

### Chunking and citation rules

Rules:

```text
chunk markdown by heading section first
split oversized sections by configured size limit
preserve heading path
preserve source path
preserve line_start/line_end when possible
store source content_hash on every chunk
```

Citation format:

```text
path:line_start-line_end
```

Heading path may be included as metadata.

### Acceptance criteria

PM-07a is complete only when:

```text
tests were added before production code;
red phase failed for missing ingestion/index behavior;
README/docs/ADR corpus can be ingested;
secret-like paths and content are denied;
markdown chunks include citations and line ranges when possible;
changed sources mark old chunks stale;
deleted sources mark chunks deleted or stale;
Memory tables do not store document chunks;
retrieval and ContextAssembler integration are not implemented yet;
architecture guardrails pass.
```

### Out of scope

Do not implement:

```text
ContentRetrievalPort retrieval
content embeddings
ContextAssembler content sections
ContextManifest content hit refs
agent answer with citations
general file RAG
source code indexing
PDF ingestion
web ingestion
```

## 10. Slice PM-07b — Project Docs retrieval and ContextAssembler integration

Depends on:

```text
PM-07a complete
ADR-034 accepted
```

### Goal

Add retrieval, embeddings and context integration for the Project Docs citation
index created in PM-07a.

PM-07b lets the assistant retrieve cited source chunks from project
documentation without storing document chunks as Memory records.

### Tests first

Contract tests:

```text
test_content_retrieval_port_returns_content_hits_not_memory_hits
test_content_hit_contains_source_chunk_score_citation_and_hash
test_retrieval_excludes_stale_chunks
test_retrieval_excludes_deleted_chunks
test_retrieval_returns_citations
test_fake_embedding_provider_is_used_for_content_embeddings
test_embedding_failure_excludes_failed_chunks_from_retrieval
```

Integration tests:

```text
test_ingestion_creates_content_embeddings
test_embedding_failure_marks_chunk_or_embedding_failed
test_retrieval_uses_content_embeddings_not_memory_embeddings
test_content_tables_remain_separate_from_memory_tables
```

Golden context tests:

```text
test_context_assembler_includes_content_hits_in_separate_section
test_context_assembler_keeps_memory_hits_and_content_hits_separate
test_context_manifest_records_content_hit_refs
test_secret_content_hit_is_excluded_from_context
```

Architecture tests:

```text
test_agent_runtime_does_not_import_content_storage_adapters
test_context_assembler_does_not_import_content_sqlalchemy_models
test_memory_subsystem_does_not_import_content_retrieval_storage
test_content_retrieval_does_not_write_memory_tables
```

E2E smoke with fake embedding/model providers:

```text
test_agent_answers_from_project_doc_content_hit_with_citation
test_agent_does_not_treat_content_hit_as_memory
```

### Expected red phase

The first test run should fail because:

```text
ContentRetrievalPort does not exist
ContentHit does not exist
content_embeddings do not exist
retrieval adapter does not exist
ContextAssembler cannot include content hits
ContextManifest cannot record content hit refs
```

Bad red failures:

```text
test stores document chunks in memory tables
test uses real embedding/model provider
test requires pgvector
test bypasses ContextAssembler
```

### Implementation

Minimal production changes:

```text
domain/content_retrieval:
  ContentQuery
  ContentHit

ports/content_retrieval:
  ContentRetrievalPort

storage:
  content_embeddings

retrieval:
  local_embedding through EmbeddingPort
  PostgreSQL array similarity or deterministic adapter-level similarity
  stale/deleted chunk exclusion

context_assembly:
  separate Relevant Project Documentation section
  ContextManifest content hit refs

events:
  content.embedding.created
  content.embedding.failed
  content.retrieved
```

### Retrieval and storage rules

Rules:

```text
ContentHit is not MemoryHit
document chunks are not MemoryRecord
content embeddings are not memory embeddings
retrieval excludes stale/deleted/failed chunks
pgvector is optional adapter optimization, not PM-07 requirement
tests use fake embedding providers
```

### ContextAssembler behavior

ContextAssembler adds content hits in a separate section:

```text
Relevant Project Documentation
```

ContextManifest records:

```text
source_id
chunk_id
citation
score
sensitivity
content_hash
```

### Permission mode behavior

Expected policy behavior:

```text
developer_local:
  project docs RAG -> allow

locked_down:
  project docs RAG -> approval_required or deny by config

automation:
  project docs RAG -> allow only for configured corpora
```

### Acceptance criteria

PM-07b is complete only when:

```text
tests were added before production code;
red phase failed for missing retrieval/context behavior;
retrieval returns ContentHit with citation;
retrieval excludes stale/deleted/failed chunks;
ContextAssembler includes content hits separately from memories;
ContextManifest records content hit refs;
Memory tables do not store document chunks;
real model/embedding calls are not required for CI;
architecture guardrails pass.
```

### Out of scope

Do not implement:

```text
general file RAG
source code indexing
PDF ingestion
web ingestion
email or Telegram ingestion
MCP resource indexing
raw conversation vector indexing
event log vector indexing
cloud embeddings
mandatory pgvector dependency
provider-native file search
```

## 11. Slice PM-08 — Automatic loop selection and CLI readiness

### Goal

Make ordinary chat, Project Docs RAG and safe/read-only tools available through
one natural CLI/API chat surface.

After PM-08, the user should not need to know or type:

```text
memory_augmented_answer
tool_react_loop
specific internal tool names
```

The historical PM-08a/PM-08e selector-era default user-facing behavior was:

```text
auto
```

In that historical model, `auto` resolved on the backend to one concrete loop
strategy before runtime execution. PM-08k/ADR-037 supersedes this for production
natural-language handling: `auto` is now a policy mode of the bounded agent loop.

PM-08 must not implement a deterministic-only selector as the target
architecture. The slice must introduce `LoopStrategySelector` plus
`IntentClassifierPort`; CI uses fake classifier implementations, while runtime
uses a local structured model-backed adapter when available and a conservative
deterministic classifier adapter as bootstrap/fallback.

### Inputs

Required docs:

```text
docs/adr/ADR-035_automatic_loop_strategy_selection.md
docs/adr/ADR-031_agent_loop_strategy_architecture.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/adr/ADR-034_content_retrieval_subsystem_and_project_docs_rag.md
docs/06_agent_runtime_and_loop_architecture.md
docs/10_api_and_streaming.md
docs/22_api_shape_and_request_lifecycle.md
docs/26_testing_strategy.md
```

Prerequisites:

```text
PM-01 complete
PM-02 complete
PM-03 complete
PM-04 complete
PM-05 complete
PM-06a complete
PM-06b complete
PM-07a complete
PM-07b complete
```

### User-facing behavior

Default CLI/API behavior:

```text
no loop_strategy provided -> auto
```

Explicit modes:

```text
auto
chat
tools
```

Mapping:

```text
auto  -> LoopStrategySelector chooses concrete loop
chat  -> memory_augmented_answer
tools -> tool_react_loop
```

Examples:

```text
"explain the permission model"
  -> memory_augmented_answer

"what does ADR-034 say about project docs RAG?"
  -> memory_augmented_answer with ContentRetrievalPort context hits

"check CPU temperature"
  -> tool_react_loop using system diagnostics tools

"inspect where LoopStrategyName is defined"
  -> tool_react_loop using project read-only shell tools
```

### Delivery breakdown

PM-08 is delivered as eleven ordered sub-slices:

```text
PM-08a Loop selection domain and selector contract
PM-08b API/request lifecycle auto mode
PM-08c CLI auto mode and mode controls
PM-08d CLI tool/RAG/approval readiness surface
PM-08e Model-backed intent classifier adapter
PM-08f Typed tool observations and direct-answer hardening
PM-08g Direct planner and capability routing registry cleanup
PM-08h Tool-intent corpus hardening and pre-voice corpus evaluation gate
PM-08i Interactive CLI shell UX hardening
PM-08j Canonical Jarvis runtime startup
PM-08k Agentic loop-first request handling cleanup
```

PM-08a through PM-08h record the implemented selector/classifier-era path and
its test evidence. PM-08k supersedes that runtime direction for production
natural-language handling: classifier, threshold, `RequestResolver` and
`RouteDecision` artifacts may remain only as migration compatibility,
historical evidence or explicitly quarantined follow-up work. They are not a
PM-09 gate and must not be reintroduced as a pre-agent semantic router.

Do not start PM-09 voice implementation until PM-08d, PM-08e, PM-08f, PM-08g,
PM-08h, PM-08i, PM-08j, PM-08k and PM-08l are complete. Voice depends on the same
user-turn surface being usable from text first, on direct answers not inheriting
fragile stdout parsing, on registry-backed tool metadata, on corpus evidence
that covers typed and spoken-transcript-like requests, on a Codex-like
interactive CLI shell that is usable enough to dogfood before voice, on a
canonical local startup path that does not rely on manual
DB/migration/daemon orchestration, and on the PM-08k agentic-loop-first contract
that removes the separate classifier-first path before voice. PM-08l is the
final hardening gate that proves that contract through DB-backed transcript-like
API/e2e evidence and startup invariants before PM-09 starts.

### PM-08a — Loop selection domain and selector contract

Goal:

```text
Create the backend selection model without changing CLI/API defaults yet.
```

Tests first:

```text
test_loop_selection_mode_accepts_auto_chat_tools
test_loop_selection_request_rejects_missing_required_fields
test_intent_classification_requires_confidence_between_zero_and_one
test_capability_candidate_does_not_store_raw_prompt_evidence
test_loop_selection_decision_distinguishes_requested_mode_from_selected_loop
test_medium_confidence_tool_intent_falls_back_conservatively
test_misleading_without_tools_does_not_fallback_to_fake_chat
test_selector_uses_intent_classifier_for_auto_mode
test_fake_intent_classifier_drives_selector_decision
test_auto_selects_memory_loop_for_ordinary_chat
test_auto_selects_memory_loop_for_project_docs_question
test_auto_selects_tool_loop_for_project_shell_read_intent
test_auto_selects_tool_loop_for_system_diagnostics_intent
test_selector_passes_working_directory_to_real_policy_for_system_diagnostics
test_auto_reports_tools_disabled_for_tool_intent
test_tools_disabled_reason_takes_precedence_over_unavailable_capabilities
test_classifier_low_confidence_falls_back_to_chat
test_non_tool_intent_drops_tool_candidate_metadata_before_chat_fallback
test_classifier_tool_intent_is_clamped_by_policy
test_explicit_chat_override_selects_memory_loop
test_explicit_tools_override_selects_tool_loop
test_explicit_tools_override_fails_closed_without_candidate_allowlist
test_selector_does_not_treat_rag_as_tool_loop
test_selector_outputs_reason_code_and_candidate_capabilities
test_selector_does_not_log_raw_prompt_in_decision_payload
```

Architecture tests:

```text
test_selector_depends_on_intent_classifier_port_not_model_provider
test_selector_does_not_import_tool_adapters
test_selector_does_not_import_storage_adapters
test_selector_does_not_import_context_assembler_implementation
test_intent_classifier_port_does_not_import_tool_adapters
test_memory_augmented_answer_still_does_not_import_toolgateway
```

Expected red phase:

```text
LoopStrategySelector does not exist
IntentClassifierPort does not exist
LoopSelectionMode does not exist
LoopSelectionRequest does not exist
LoopSelectionDecision does not exist
CapabilityCandidate does not exist
IntentClassification does not exist
```

Implementation:

```text
domain:
  LoopSelectionMode
  LoopSelectionRequest
  LoopSelectionDecision
  stable reason_code strings
  IntentFamily
  IntentClassification
  CapabilityCandidate
  SelectionDecisionStatus
  SelectionFallbackPreference

runtime:
  IntentClassifierPort
  FakeIntentClassifier
  DeterministicIntentClassifier conservative baseline
  LoopStrategySelector
  classifier-output validation
  policy/config/budget gates, including working_directory policy scope
```

Acceptance:

```text
selector is fully testable without API, CLI or real LLM calls;
selector returns constrained decisions, not executable tool instructions;
selector distinguishes requested mode from selected concrete loop;
RAG is classified as memory loop with content retrieval, not tool loop;
tool intent can be rejected or marked unavailable without fake chat;
working_directory can be passed to PolicyPort without being stored in
loop-selection event payloads.
```

### PM-08b — API/request lifecycle auto mode

Goal:

```text
Wire backend auto-selection into the request lifecycle.
```

Tests first:

```text
test_message_without_loop_strategy_uses_auto_mode
test_auto_mode_persists_requested_mode_and_selected_loop_metadata
test_auto_mode_emits_loop_selection_event
test_model_profile_matches_selected_loop
test_tools_disabled_does_not_silently_fallback_to_chat
test_explicit_tools_mode_is_rejected_when_tools_disabled
test_request_metadata_keeps_selected_model_profile_resolution_after_loop_selection
test_tool_loop_budget_without_tool_calls_rejects_before_request_persistence
```

Architecture tests:

```text
test_api_transport_does_not_own_loop_selection_rules
test_request_metadata_does_not_call_tool_adapters
test_loop_selection_events_do_not_include_raw_prompt
```

Expected red phase:

```text
auto is not a known user-facing mode
missing loop_strategy still resolves directly to memory_augmented_answer
selected loop metadata is not persisted separately from requested mode
loop selection events do not exist
model profile resolution still happens before concrete loop selection
```

Implementation:

```text
api/request lifecycle:
  accept auto/chat/tools user-facing modes
  route missing loop_strategy to requested_mode=auto
  call LoopStrategySelector before AgentRuntime execution
  allocate a stable pre-submit request_id from conversation_id + client_message_id
    so accepted request lifecycle and idempotent retries share correlation
  allocate a separate deterministic pre-submit failure request_id for rejected
    selection attempts so a failed attempt cannot collide with a later accepted
    retry that reuses the same client_message_id
  do not default missing working_directory to the daemon cwd; tool intents that
    need scope must provide explicit working_directory or fail through policy
  persist explicit working_directory for execution only; redact it from public
    request payloads and expose working_directory_scope for status/debug output
  persist requested_mode separately from selected_loop_strategy
  validate the selected loop runtime budget before request persistence
  resolve model_profile after selected_loop_strategy is known
  emit redacted loop-selection started/completed/failed events
```

Acceptance:

```text
API clients can omit loop_strategy and get auto routing;
explicit chat/tools overrides still work;
selected concrete loop and selected model profile are visible in request metadata;
tools-disabled tool intent fails clearly rather than hallucinating live state;
tool-loop budget mismatch fails before request persistence or runtime execution;
no real LLM, shell or diagnostics calls are required for API contract tests.
```

### PM-08c — CLI auto mode and mode controls

Goal:

```text
Make auto the normal CLI behavior and keep explicit modes as debug controls.
```

Tests first:

```text
test_cli_submit_message_omits_loop_strategy_for_default_auto
test_http_submit_message_omits_loop_strategy_for_default_auto
test_cli_chat_accepts_loop_strategy_override
test_interactive_mode_defaults_to_auto
test_interactive_mode_command_switches_between_auto_chat_tools
test_interactive_help_lists_mode_command
test_interactive_status_displays_current_mode_without_internal_loop_names_by_default
```

Architecture tests:

```text
test_cli_does_not_import_loop_selector
test_cli_does_not_duplicate_selector_rules
```

Expected red phase:

```text
CLI still defaults to an internal concrete loop strategy
CLI cannot set or display auto/chat/tools mode
interactive help does not list mode controls
CLI status leaks internal loop names as the primary user-facing mode
```

Implementation:

```text
cli:
  default to auto by omitting loop_strategy from normal message submissions
  add --loop-strategy auto|chat|tools as debug override
  add /mode auto|chat|tools in interactive mode
  treat /mode auto as returning to default omitted loop_strategy
  show current user-facing mode in help/status
  keep routing rules on the backend
```

Acceptance:

```text
normal CLI chat does not require loop_strategy or tool names;
debug override exists but is not required for ordinary use;
CLI sends user-facing mode, not selector decisions;
CLI help/status is understandable without exposing architecture internals first.
```

### PM-08d — CLI tool/RAG/approval readiness surface

Goal:

```text
Prove that the text CLI can use the implemented RAG, tools, approvals and
cancellation paths through one normal chat surface before voice is added.
```

Tests first:

```text
test_cli_plain_question_uses_memory_loop
test_cli_project_docs_question_uses_rag_without_tool_loop
test_cli_system_diagnostics_question_uses_tool_loop
test_cli_project_shell_question_uses_tool_loop
test_cli_auto_tool_intent_submits_caller_working_directory
test_http_submit_message_sends_working_directory_when_provided
test_cli_tool_intent_approval_flow_still_works
test_cli_approval_prompt_can_approve_deny_and_cancel
test_cli_cancel_active_request_keeps_interactive_session_usable
test_cli_tool_flow_renders_action_approval_observation_without_raw_json_noise
test_cli_tool_unavailable_message_is_clear_when_policy_denies
test_request_stream_buffer_replays_public_tool_lifecycle_events_without_raw_output
test_tool_react_loop_streams_public_tool_lifecycle_events
```

Architecture tests:

```text
test_cli_rendering_does_not_execute_tools
test_cli_approval_controls_call_api_not_toolgateway_directly
test_voice_readiness_surface_does_not_add_voice_dependencies
```

Expected red phase:

```text
CLI e2e cannot trigger RAG/tools through auto mode
approval flow is only partly usable from interactive CLI
cancel/interrupt leaves the interactive session in a broken state
tool proposal/result rendering is too raw or ambiguous for normal usage
```

Implementation:

```text
cli:
  render selected mode/loop/result in user-facing language
  render tool proposals, approval prompts and observations consistently
  send caller working_directory with CLI chat submissions so backend policy can
    authorize project shell and system diagnostics scope
  support approve/deny/cancel from the normal interactive flow
  make Ctrl-C and /cancel return to a usable prompt
  keep RAG answers citation-oriented and not tool-loop-looking

tests:
  fake classifier, fake providers, fake tools and fake approvals only
```

Acceptance:

```text
plain text CLI can automatically answer ordinary questions;
plain text CLI can automatically retrieve Project Docs RAG citations;
plain text CLI can automatically route live project/system inspection to tools;
project/system tool routing has explicit caller scope from the CLI working directory;
approval-required tool flow is understandable and controllable from CLI;
cancel/interrupt does not break the session;
tool flow output is not raw JSON-first;
all PM-08a through PM-08d architecture tests pass.
```

Bad red failures for all PM-08 sub-slices:

```text
test requires a real LLM call
test calls real shell or diagnostics commands
test bypasses API/runtime boundaries
test expects RAG to be implemented as a tool
test logs raw full prompt text
test duplicates selector rules in CLI
```

PM-08 must introduce the `IntentClassifierPort` boundary and use fake
implementations so CI does not require a real LLM call. Runtime should prefer a
local structured model-backed classifier when a structured local model profile
is configured, with deterministic fallback for startup, failure and tests.
PM-08f must then harden the direct-tool answer path before PM-09 voice work:
fast direct execution may stay, but user-facing answers must consume typed tool
observations rather than command-specific stdout parsing inside the loop.
PM-08g and PM-08h are mandatory pre-voice hardening slices: routing metadata,
direct-tool eligibility and corpus expectations must be centralized and tested
before spoken turns start relying on automatic routing.

### PM-08e — Model-backed intent classifier adapter

Goal:

```text
Replace per-question keyword routing as the runtime direction without making
the selector itself model/provider aware.
```

Tests first:

```text
test_model_backed_classifier_maps_structured_payload_to_intent_classification
test_model_backed_classifier_sends_constrained_schema_to_model_router
test_model_backed_classifier_rejects_raw_command_tool_names
test_model_backed_classifier_falls_back_to_deterministic_classifier_on_router_error
test_model_backed_classifier_guardrail_corrects_local_system_state_false_negative
test_tool_intent_corpus_has_broad_multilingual_coverage
test_ci_baseline_classifier_routes_tool_intent_corpus_cases
test_request_metadata_accepts_injected_intent_classifier
```

Architecture tests:

```text
test_model_backed_intent_classifier_depends_on_model_router_port_not_provider_adapters
test_loop_selector_depends_on_intent_classifier_port_not_model_provider
```

Expected red phase:

```text
ModelBackedIntentClassifier does not exist
runtime_request_metadata cannot accept an injected IntentClassifierPort
app factory always hard-codes deterministic classification
```

Implementation:

```text
runtime:
  ModelBackedIntentClassifier behind IntentClassifierPort
  StructuredModelRequest against local_structured profile by default
  strict schema for IntentClassification JSON
  parser from JSON payload to domain objects
  fallback classifier on router failure or invalid payload
  narrow local-system-state false-negative guardrail after model output
  no provider-specific imports in runtime

request lifecycle:
  allow runtime_request_metadata/create_app to receive an IntentClassifierPort
  app_factory wires model-backed classifier with deterministic fallback
```

Acceptance:

```text
runtime can use local model classification for varied natural language wording;
CI uses fake model-router responses and does not call a real LLM;
selector remains provider-agnostic and policy-authoritative;
invalid model output cannot become a raw command/tool execution;
tool-intent routing has a corpus of multilingual formulations with CI-safe
baseline checks and opt-in local model evaluation;
live-state requests still fail unavailable rather than hallucinating when tools
are disabled or denied.
```

### PM-08f — Typed tool observations and direct-answer hardening

Goal:

```text
Keep the fast direct-tool path, but remove the architectural dependency on
loop-level stdout parsing for user-facing answers.
```

Rationale:

```text
The direct path is useful for low-latency common questions, but it must not grow
into a large set of regex parsers inside tool_react_loop. Command output formats
vary by OS version, locale and tool implementation. Parsing belongs in
capability-specific adapters/normalizers with contract fixtures. The loop should
orchestrate execution and answer assembly, not understand every command format.
```

Tests first:

```text
test_system_diagnostics_tool_returns_typed_os_version_payload
test_system_diagnostics_tool_returns_typed_battery_payload
test_system_diagnostics_tool_returns_typed_disk_payload
test_system_diagnostics_tool_returns_typed_vpn_payload
test_system_diagnostics_tool_returns_typed_process_search_payload
test_system_diagnostics_tool_returns_typed_memory_payload
test_direct_answer_uses_typed_payload_not_raw_stdout
test_direct_answer_falls_back_when_typed_payload_missing
test_direct_answer_does_not_parse_unrecognized_stdout_format
test_direct_answer_partial_payload_includes_cautious_warning
test_unparsed_direct_payload_routes_to_react_when_budget_allows
test_tool_react_loop_has_no_scope_specific_stdout_parsers
test_diagnostics_normalizers_cover_macos_and_linux_fixture_outputs
test_unparsed_tool_output_can_enter_bounded_react_context_as_data
```

Architecture tests:

```text
test_tool_react_loop_does_not_import_diagnostics_parsers
test_diagnostics_parsers_do_not_import_loop_runtime
test_tool_adapters_return_provider_neutral_typed_observations
```

Expected red phase:

```text
ToolObservation has only raw content/content_type for diagnostics answers
ToolInvocationResult and ToolObservationRef cannot carry typed payloads through
  the gateway/context/direct-answer pipeline
tool_react_loop owns os/battery/disk/vpn/process stdout parsing
diagnostics adapters do not expose typed normalized payloads
direct answers cannot distinguish typed data from raw fallback text
```

Planning decisions before implementation:

```text
typed payload shape:
  chosen:
    start with one generic provider-neutral envelope:
      structured_content
      structured_schema
      structured_schema_version
      parse_status
      parse_warnings
  rejected for PM-08f:
    a large class hierarchy per diagnostics command
  reason:
    the important boundary is typed data propagation and parse_status semantics;
    detailed domain classes can be added later only where they remove complexity

unparsed behavior:
  chosen:
    direct answers never infer live state from unparsed stdout
    if policy, budget and loop state allow it, bounded/redacted raw output may
      enter the ordinary ReAct/model analysis path as data
    otherwise return a clear unparsed/unavailable answer
  rejected:
    best-effort direct regex fallback inside tool_react_loop
  reason:
    fragile parsing was the failure mode that PM-08f is removing

partial behavior:
  chosen:
    answer only from present typed fields and include a cautious warning
  open detail:
    exact user-facing warning text can be finalized during implementation tests

raw content compatibility:
  chosen:
    keep content/content_type for bounded human/debug text, event payloads and
    ReAct fallback
  rejected:
    deleting raw content from ToolObservation in PM-08f
  reason:
    audit/debug and ordinary ReAct still need bounded source material

redaction:
  chosen:
    process command lines, network evidence, paths and host-specific identifiers
    are redacted or bounded by default unless the capability contract explicitly
    requires them and policy permits disclosure
  open detail:
    per-schema sensitivity labels may be added when a field needs finer control

scope:
  chosen:
    PM-08f covers existing direct diagnostics scenarios only:
      OS version
      battery
      disk
      VPN
      process search
      CPU/resources
      memory/resources
      sensors
  rejected:
    adding new tools or write-capable actions in PM-08f

sequencing:
  chosen:
    PM-08f does not absorb the full registry/planner cleanup. It may add typed
    fields and migration-compatible metadata hooks needed for PM-08g, but
    CapabilityRoutingRegistry and DirectToolPlan ownership remain a separate
    PM-08g slice.
  reason:
    PM-08f should stay focused on typed observation propagation and parser
    removal; mixing it with routing registry cleanup would make the slice too
    broad to verify cleanly.
```

PM-08k refinement:

```text
The default runtime no longer treats diagnostics or natural-language extraction
as direct-answer scenarios. Typed diagnostics payloads remain required because
they are the stable tool observation contract for ReAct context, audit output
and future renderers, but direct execution is now restricted to the smallest
obvious whitelist:
  current_time
  explicit symbolic calculator expressions

Natural-language arithmetic, calendar/event countdowns and all system
diagnostics must route through tool_react_loop with validated candidate tools,
then let the ReAct model choose the concrete tool call and arguments.
```

Implementation:

```text
domain/tools:
  add provider-neutral typed observation payload support through the full
  execution pipeline:
    ToolInvocationResult
      -> ToolObservation
      -> ToolObservationRef
      -> context/direct formatter/event payload
  use one field contract everywhere:
    structured_content
    structured_schema
    structured_schema_version
    parse_status:
      parsed
      partial
      unparsed
      not_applicable
    parse_warnings
  keep content/content_type for bounded human/debug text and backwards
    compatibility

tools/system_diagnostics:
  place normalization outside tool_react_loop, near the adapters:
    tools/system_diagnostics/normalizers/os_version.py
    tools/system_diagnostics/normalizers/battery.py
    tools/system_diagnostics/normalizers/disk.py
    tools/system_diagnostics/normalizers/vpn.py
    tools/system_diagnostics/normalizers/process.py
    tools/system_diagnostics/normalizers/cpu.py
    tools/system_diagnostics/normalizers/sensors.py
  normalize platform-specific command outputs into provider-neutral schemas:
    system.os_version v1:
      product_name
      version
      build
      platform
    system.battery_charge v1:
      percent
      state
      source
    system.disk_free v1:
      filesystems[]
        mount
        size
        used
        available
        used_percent
    system.vpn_status v1:
      connected
      interface_or_service optional
      evidence
    system.process_name_search v1:
      query
      matches[]
        pid
        name
        command optional, redacted/bounded
    system.cpu_overview v1:
      logical_cores
      physical_cores optional
      load_percent optional
      load_average optional
      user_percent optional
      system_percent optional
      idle_percent optional
      source
    system.memory_overview v1:
      total
      used optional
      available
      free optional
      used_percent optional
      swap_total optional
      swap_used optional
      source
    system.sensor_snapshot v1:
      sensors[]
        name
        kind
        temperature_c optional
        source
  share schemas across platforms:
    sw_vers / uname / os-release -> system.os_version v1
    pmset / upower              -> system.battery_charge v1
    df on macOS/Linux          -> system.disk_free v1
  keep raw stdout/stderr bounded and redacted in metadata/content only

runtime/loops:
  direct answer builders read typed payloads only
  CPU and memory direct-answer v1 is aggregate-only:
    no per-core usage
    no per-process memory
    no top-process list
    no thread-level statistics
    no pressure/stall breakdown
  parsed:
    answer deterministically from structured_content
  partial:
    answer only from available fields and include a cautious warning
  unparsed:
    do not invent a parsed answer
  when model calls are allowed, ordinary ReAct may analyze bounded raw
    observations as data after direct typed answering is unavailable
  when model calls are not allowed, return a clear unparsed/unavailable result
  keep scope-specific stdout parsing out of tool_react_loop
```

Acceptance:

```text
direct OS, battery, disk, VPN and process answers come from typed payloads;
direct CPU, memory and sensor answers come from typed payloads;
typed payloads propagate from ToolInvocationResult through ToolObservation and
ToolObservationRef into direct formatters and context/event payloads;
loop-level parsers for command-specific stdout are removed or reduced to generic
typed-payload formatting;
parse_status drives the direct-answer fallback matrix;
raw output format changes are caught by adapter/normalizer fixture tests;
unknown output format produces a clear unparsed/unavailable result or bounded
ReAct fallback, not hallucinated state;
ToolGateway remains the only execution boundary;
AgentRuntime and LoopStrategySelector remain independent of diagnostics
adapters;
CI does not require host diagnostics commands or a real LLM.
```

Out of scope:

```text
provider-native tool calling
MCP tool schema export
artifact storage for large raw outputs
write-capable tools
per-core CPU direct answer payloads
per-process memory/resource payloads
top-process resource summaries
voice input/output
```

### PM-08g — Direct planner and capability routing registry cleanup

Goal:

```text
Replace scattered direct-tool metadata and allowlists with one typed routing
registry and one direct-tool planning component.
```

Rationale:

```text
PM-08e/PM-08f make auto-routing useful, but direct eligibility is currently a
cross-cutting concern: request metadata builds tool summaries, the model
classifier knows a direct allowlist, and the tool loop validates persisted
metadata again. Before adding voice or more tools, this must become one
auditable decision point.
```

Tests first:

```text
test_capability_routing_registry_lists_enabled_tools_from_settings
test_capability_routing_registry_rejects_duplicate_tool_names
test_model_classifier_rejects_tool_names_not_in_available_registry
test_direct_tool_planner_allows_known_safe_scenario
test_direct_tool_planner_denies_model_origin_direct_execution
test_direct_tool_planner_denies_tool_scope_mismatch
test_direct_scope_evidence_comes_from_registry_backed_extractors
test_direct_argument_extractor_fixtures_cover_supported_scenarios
test_direct_tool_plan_round_trips_through_request_metadata
test_tool_react_loop_consumes_direct_tool_plan_not_loose_tool_name_metadata
test_direct_tool_plan_requires_process_search_pattern
```

Architecture tests:

```text
test_loop_selector_does_not_import_direct_tool_planner
test_direct_tool_planner_does_not_execute_tools
test_model_intent_classifier_does_not_own_direct_execution_allowlist
test_request_metadata_does_not_define_tool_registry_literals
```

Expected red phase:

```text
tool routing metadata is built inline in request_metadata
direct scenarios are inferred from loose metadata keys
model classifier accepts stable-looking but unregistered tool_names
tool_react_loop accepts direct tool metadata by tool name instead of a typed plan
```

Decision matrix:

```text
registry ownership:
  options:
    A. keep available_tools_summary assembled inline in request_metadata
    B. derive everything dynamically from ToolGateway at request time
    C. introduce CapabilityRoutingRegistry fed by registered tool metadata and
       settings enablement
  chosen:
    C
  reason:
    A keeps duplication; B makes settings/policy visibility unclear; C gives one
    auditable metadata source without making ToolGateway an authorization system
  resolved details:
    tool descriptors declare static routing metadata
    settings enable/disable capabilities and tool families
    CapabilityRoutingRegistry merges descriptors with active settings into the
    available registry used by classifiers and direct planning
    ToolGateway remains the execution boundary, not the metadata authority

tool_name validation:
  options:
    A. accept stable-looking labels and let later stages filter
    B. strip unknown tool names but keep the candidate
    C. reject the candidate when all proposed tool_names are unknown; strip only
       mixed unknown extras when at least one valid registered tool remains
  chosen:
    C
  reason:
    model output is advice, but registry membership should be enforced at the
    classifier/parser boundary before selection metadata is persisted
  resolved details:
    reject only the invalid candidate when possible, not the whole classification
    if other candidates remain valid
    if a tool-intent classification has no valid candidates after validation,
    return classifier_unavailable/unknown or fail_unavailable according to the
    existing fallback rules instead of silently fabricating a tool candidate

direct execution authority:
  options:
    A. model-origin classification can directly authorize direct execution
    B. model-origin classification can choose tool_react_loop, but direct
       execution needs DirectToolPlanner approval from deterministic/guardrail
       direct scope evidence
    C. remove direct execution and always use bounded ReAct
  chosen:
    B
  reason:
    direct execution is useful for latency, but it is an optimization granted by
    runtime policy/planning, not by the model
  resolved details:
    deterministic/guardrail direct scope evidence must come from registry-backed
    scenario extractors and argument extractors with fixture tests
    it must not become a new global keyword-list selector hidden inside
    DirectToolPlanner

DirectToolPlan shape:
  options:
    A. keep loose loop_selection_direct_tool_name metadata
    B. store a typed redacted DirectToolPlan in request metadata
    C. keep DirectToolPlan transient only and do not persist selection details
  chosen:
    B
  reason:
    execution, events and later debugging need a clear decision artifact, but the
    artifact must be redacted and non-executable by itself
  resolved details:
    request metadata and event logs store only a redacted DirectToolPlan summary:
      scenario
      tool_names
      capability labels
      scope_hint
      classification_source
      provenance/evidence labels
      redacted required argument labels or values approved by policy
    raw command output, raw user prompt and executable argv are not stored inside
    the DirectToolPlan

planner placement:
  options:
    A. put direct planning inside LoopStrategySelector
    B. put direct planning inside request metadata after loop selection
    C. put direct planning inside tool_react_loop only
  chosen:
    B
  reason:
    selector should choose a loop; the loop should execute a plan; request
    metadata is the boundary where selected loop, classifier output, registry and
    redacted persisted metadata meet

argument ownership:
  options:
    A. DirectToolPlan includes final argv
    B. DirectToolPlan includes scenario/scope/required argument values; direct
       tool helpers build argv from registered scenarios
    C. model classifier emits tool arguments
  chosen:
    B
  reason:
    plans should be auditable and non-provider-specific, while executable argv
    remains deterministic runtime code behind ToolGateway

process search:
  chosen:
    DirectToolPlanner must require an extracted process search pattern for the
    process_name_search direct scenario
  fallback:
    if no pattern is available, route to ordinary bounded ReAct/clarification
    instead of direct pgrep

implementation sequencing:
  chosen:
    implement PM-08g after PM-08f as a separate TDD slice
  allowed PM-08f preparation:
    typed observation fields and backward-compatible metadata fields that make
    DirectToolPlan migration straightforward
  rejected:
    doing registry cleanup opportunistically inside PM-08f without its own red
    phase and architecture tests
```

Implementation:

```text
runtime/routing:
  CapabilityRoutingRegistry
    reads settings and registered ToolGateway metadata
    produces available_tools_summary for classifiers
    owns stable tool_name -> capability/risk/intent metadata
    owns or references scenario/argument extractor descriptors for direct-capable
      tools

runtime/direct_tools:
  DirectToolPlan
    tool_names
    scenario
    scope_hint
    required_arguments
    classification_source
    provenance
  DirectToolPlanner
    accepts LoopSelectionDecision plus registry metadata
    consumes registry-backed scenario/argument extractor output
    grants direct execution only for allowlisted tool+scenario+scope combinations
    denies direct execution for model-origin classifier output unless a separate
      deterministic/guardrail direct scope approved it
    keeps policy and ToolGateway as execution authorities

runtime/request_metadata:
  requests available_tools_summary from the registry
  stores a redacted DirectToolPlan shape, not ad hoc direct_tool_name fields

runtime/loops:
  reads DirectToolPlan and delegates argument construction to direct-tool helpers
  does not reinterpret classifier candidates as execution authorization
```

Acceptance:

```text
there is one source of truth for auto-routable tool metadata;
unknown model-proposed tool_names are rejected or stripped before selection;
direct execution eligibility is represented as DirectToolPlan, not loose metadata;
direct planning validates tool, capability, scenario, scope and source together;
ToolGateway remains the only execution boundary;
adding a new direct-capable tool changes registry/planner fixtures, not selector
or CLI routing code;
adding a new direct-capable tool requires a descriptor plus registry/planner
fixture coverage before it can be auto-routed or direct-planned.
```

Out of scope:

```text
new tool families
write-capable direct tools
provider-native tool calling
voice input/output
```

### PM-08h — Tool-intent corpus hardening and pre-voice corpus evaluation gate

Goal:

```text
Turn the multilingual tool-intent corpus into a pre-voice quality gate for
automatic routing, direct planning and safe fallback behavior.
```

Rationale:

```text
Voice will make routing misses more visible and more expensive to correct in the
moment. Before PM-09, typed turns must prove that varied natural-language
requests route to the expected family, capabilities, tool names, direct plan or
safe fallback without relying on one-off fixes for each phrasing.
```

Tests first:

```text
test_tool_intent_corpus_has_required_categories_for_pre_voice_gate
test_tool_intent_corpus_exact_ci_baseline_matches_expected_tools
test_tool_intent_corpus_covers_negative_live_state_near_misses
test_tool_intent_corpus_covers_direct_plan_scenarios
test_tool_intent_corpus_covers_model_classifier_fake_payloads
test_tool_intent_corpus_covers_spoken_transcript_variants
test_tool_intent_corpus_asserts_policy_outcome_for_relevant_cases
test_pre_voice_local_model_eval_report_is_recorded
test_tool_intent_corpus_requires_languages_for_priority_categories
test_guardrail_baseline_does_not_turn_conceptual_questions_into_tools
test_pre_voice_routing_gate_blocks_missing_priority_categories
```

Evaluation tests:

```text
test_local_model_classifier_routes_pre_voice_corpus_opt_in
test_local_model_classifier_reports_failures_without_ci_network_or_real_llm
```

Decision matrix:

```text
baseline strictness:
  options:
    A. subset assertions for all corpus cases
    B. exact assertions for every corpus case
    C. exact assertions for critical ci_baseline/direct-plan cases, subset only
       for explicitly marked extensible cases
  chosen:
    C
  reason:
    critical routing must be stable, but future multi-tool or explanatory cases
    may legitimately add non-breaking candidates

corpus dimensions:
  chosen required expectation fields:
    intent_family
    capabilities
    tool_names
    scope_hint
    direct_plan expected/forbidden
    fallback_behavior
    policy_outcome optional
    approval_possible optional
    spoken_transcript_variants optional
  reason:
    before voice, correctness is not only intent family; it is also whether the
    system will execute directly, fall back, require approval, or refuse
    misleading live-state chat

mandatory pre-voice categories:
  chosen:
    safe.current_time
    safe.date_countdown
    safe.calculator
    safe.daemon_status
    system.os_version
    system.cpu_overview
    system.memory
    system.disk
    system.battery
    system.temperature
    system.processes
    system.network
    system.vpn
    project.inspection
    project.docs_question
    ordinary.conceptual
    ordinary.near_miss_live_state
    tools_disabled.live_state
    spoken_transcript_variants
  reason:
    these categories cover the expected typed CLI/voice surface before adding
    audio, and they directly match prior failures around local diagnostics,
    false live-state answers and per-phrase routing fixes

language coverage:
  options:
    A. require every category in every supported language
    B. require Russian and English for all priority categories, plus additional
       languages for representative live-state and ordinary-chat groups
    C. keep current ad hoc language coverage
  chosen:
    B
  reason:
    A is too heavy for the near term; C already missed phrasing diversity; B is
    a practical pre-voice quality gate

spoken transcript coverage:
  chosen:
    priority categories include mandatory transcript-like variants before PM-09:
      fillers and conversational prefixes such as "ээ", "ну", "слушай"
      wake-name prefixes such as "джарвис проверь" or "jarvis check"
      missing punctuation
      inconsistent casing
      mixed Russian/English terms
      common ASR-like inflections and phrasing variants
  advisory:
    representative misheard tool nouns may be included in opt-in/evaluation cases
    once real STT output shows concrete recurring errors, but they are not a hard
    PM-08h gate
  reason:
    PM-09 consumes STT transcripts, not clean typed prompts; the pre-voice gate
    must catch common transcript noise without turning speculative ASR mistakes
    into an unbounded required corpus

negative examples:
  chosen:
    every priority live-state category gets conceptual near-misses that must stay
    ordinary_chat, plus tools-disabled/fail-unavailable cases where relevant
  examples:
    explain what VPN means
    how does CPU temperature monitoring work
    what is disk space conceptually
  reason:
    voice and natural phrasing increase false-positive risk

model evaluation:
  chosen:
    CI uses deterministic/fake classifier and fake model-router payloads for
    the PM-08h-era corpus while those tests remain
    real local classifier model evaluation is opt-in and reports
    category/language failures as historical evidence
    after PM-08k, classifier model evaluation no longer defines PM-09 readiness
    voice readiness is proven by spoken-transcript-like requests entering the
    same bounded agent loop, policy gates and ToolGateway path as typed input
  rejected:
    requiring a real local model in CI
  reason:
    CI must stay deterministic and network-free, and PM-08k removes the
    classifier-first runtime path before voice

pre-voice gate:
  chosen:
    Before PM-08k, PM-08h exact ci_baseline and guardrail baseline protected
    classifier-era production behavior. After PM-08k, they do not define PM-09
    readiness unless they are rewritten as agent-loop transcript parity cases.
  open detail:
    PM-08k decides which historical classifier fixtures are deleted, quarantined
    as evaluation-only or rewritten as agent-loop transcript cases
```

Implementation:

```text
tests/fixtures/intent_routing:
  split corpus expectations into:
    intent_family
    capabilities
    tool_names
    scope_hint
    direct_plan expected/forbidden
    fallback_behavior
    policy_outcome optional
    approval_possible optional
    spoken_transcript_variants optional
    ci_baseline
    guardrail_baseline
    opt_in_model_eval

tests/unit:
  make critical CI baseline assertions exact for tool_names and direct_plan
  keep subset-style assertions only for explicitly marked extensible cases

tests/evaluation:
  keep real local model evaluation opt-in and non-CI
  report confusion by category/language/scope
  retain classifier comparison output only as historical/evaluation evidence
  rewrite PM-09 readiness evidence around spoken-transcript-like agent-loop
  requests, not a separate local classifier model pass
```

Acceptance:

```text
priority live-state categories have positive and negative examples in multiple
languages;
critical CI baseline cases assert exact expected tool_names and direct_plan
state;
calculator, daemon status, datetime, diagnostics, process search, VPN, disk,
battery, CPU, memory, temperature, project inspection and ordinary conceptual
near-misses are represented;
spoken-transcript-like variants are represented for priority categories;
relevant cases assert expected policy_outcome and approval_possible behavior;
historical local classifier evaluation is either quarantined as evaluation-only
or replaced with agent-loop transcript cases before PM-09 starts;
guardrail tests prove conceptual questions do not accidentally route to tools;
PM-09 cannot start until classifier-era corpus gates are either quarantined as
historical/evaluation evidence or replaced by green spoken-transcript-like
agent-loop cases.
```

Out of scope:

```text
requiring a real local model in CI
large benchmark harness
voice audio test data
```

### PM-08i — Interactive CLI shell UX hardening

Goal:

```text
Make the text CLI a Codex-like interactive dogfood shell before voice starts.
```

Inputs:

```text
docs/adr/ADR-036_interactive_cli_shell_architecture.md
docs/10_api_and_streaming.md
docs/22_api_shape_and_request_lifecycle.md
docs/26_testing_strategy.md
docs/37_post_mvp_tdd_slices_plan.md
```

Tests first:

```text
test_slash_command_registry_filters_by_partial_prefix
test_slash_command_completion_includes_descriptions_and_argument_hints
test_prompt_toolkit_reader_is_selected_for_tty
test_plain_flag_forces_line_reader_without_prompt_toolkit
test_status_line_renders_mode_readiness_conversation_phase_model_and_cwd_scope
test_status_line_redacts_secret_like_paths_and_truncates_long_values
test_shell_activity_transitions_follow_public_stream_events
test_shell_activity_never_renders_fake_percentages
test_cli_ctrl_c_clears_prompt_or_cancels_active_request
test_cli_cancel_active_request_keeps_interactive_session_usable
test_interactive_history_does_not_store_secret_or_memory_add_lines
test_non_tty_interactive_flow_remains_deterministic
```

Architecture tests:

```text
test_cli_does_not_import_loop_selector_or_runtime_tool_adapters
test_cli_does_not_import_storage_or_model_provider_clients
test_prompt_toolkit_imports_are_confined_to_cli_shell_modules
```

Expected red phase:

```text
SlashCommandRegistry does not exist
PromptToolkitLineReader does not exist
ShellActivityState does not exist
--plain is not accepted
TTY reader selection still uses the raw manual ANSI reader
status-line rendering does not exist
```

Implementation:

```text
dependencies:
  prompt_toolkit>=3.0

cli:
  add SlashCommandRegistry and SlashCommandDefinition
  add PromptToolkitLineReader for TTY mode
  keep InteractiveLineReader for non-TTY and --plain mode
  add ShellActivityState and status-line renderer helpers
  add global --plain flag
  route stream events into activity phases while preserving existing transcript
    rendering
  keep slash command execution in the current chat flow
```

User-facing behavior:

```text
TTY mode:
  prompt_toolkit prompt
  dynamic slash command menu while typing /...
  bottom toolbar with mode, readiness, conversation, request phase, model and
    redacted cwd scope
  in-memory history only

non-TTY or --plain:
  deterministic line-oriented input
  no animation or terminal-control assumptions
```

Acceptance:

```text
TTY mode uses prompt_toolkit-backed input and completion;
non-TTY and --plain keep deterministic line-oriented behavior;
status line shows only public/redacted shell state;
activity indicator is phase-based and never percentage-based;
slash command discovery filters dynamically and exposes descriptions/hints;
Ctrl-C, Ctrl-D, /cancel and approval prompts leave the session usable;
history remains in-memory and filters secret/memory-add input;
CLI stays client-only and does not import backend routing/tool/storage/provider
implementation modules.
```

Out of scope:

```text
Textual or full-screen TUI
alternate-screen layout
session dashboard panes
persistent command history
backend API changes
new tools or voice behavior
Rich dependency
```

### PM-08j — Canonical Jarvis runtime startup

Goal:

```text
Make the pre-voice Jarvis runtime startable through one canonical local command
path instead of manual DB, migration, daemon, health and CLI orchestration.
```

Inputs:

```text
docs/12_deployment_and_runtime.md
docs/26_testing_strategy.md
docs/34_post_mvp_roadmap.md
docs/37_post_mvp_tdd_slices_plan.md
Makefile
infra/compose/
```

Tests first:

```text
test_jarvis_compose_uses_persistent_database_not_test_database
test_jarvis_up_runs_database_migrations_before_daemon_start
test_jarvis_up_waits_for_daemon_health
test_jarvis_status_reports_pid_port_health_profile_and_log_path
test_jarvis_down_stops_daemon_from_pid_file
test_jarvis_cli_uses_canonical_base_url_without_plain_mode
test_jarvis_up_fails_loudly_when_prompt_toolkit_is_missing
test_jarvis_up_does_not_install_dependencies_or_pull_models_implicitly
test_jarvis_reset_is_explicitly_destructive_and_never_runs_from_up
```

Architecture tests:

```text
test_jarvis_runtime_does_not_reuse_test_postgres_volume_or_port
test_makefile_jarvis_targets_delegate_to_single_startup_script
test_jarvis_startup_script_does_not_import_runtime_or_storage_implementation
test_jarvis_startup_script_has_no_secret_defaults
```

Expected red phase:

```text
Jarvis runtime compose file does not exist
Jarvis runtime startup script does not exist
Makefile has no jarvis-up/status/logs/down/cli targets
daemon PID/log/health contract is undocumented and unimplemented
current manual flow reuses test-db-up and can be torn down by test cleanup
```

Implementation:

```text
infra:
  add infra/compose/jarvis-postgres.yml with persistent Jarvis runtime volume
  use a port distinct from TEST_DATABASE_URL, for example 55433
  use database name jarvis_local

scripts:
  add a small Python startup driver, for example scripts/dev/jarvis_runtime.py
  commands: bootstrap, up, down, status, logs, cli, reset
  store runtime files under .run/jarvis/
  write daemon.pid, daemon.log and redacted runtime metadata
  perform stale PID cleanup
  check PID/process ownership and port 8080 before daemon start
  poll /v1/health until ready or timeout

Makefile:
  add jarvis-bootstrap
  add jarvis-up
  add jarvis-cli
  add jarvis-status
  add jarvis-logs
  add jarvis-down
  add jarvis-reset
  keep Makefile as a thin wrapper around the Python driver
```

Operational contract:

```text
jarvis-bootstrap:
  may verify editable install/dependencies and print missing-model guidance
  may explicitly install dependencies only if invoked for that purpose
  may pull local models only with explicit user command or documented opt-in

jarvis-up:
  must not install dependencies or pull models implicitly
  starts persistent Jarvis runtime Postgres
  waits for database readiness
  runs migrations
  starts daemon against Jarvis runtime DATABASE_URL and selected profile
  waits for health
  prints the exact jarvis-cli command and log path

jarvis-cli:
  opens the normal prompt_toolkit CLI against the canonical daemon
  must not pass --plain

jarvis-status:
  reports daemon pid, port, base URL, profile, health summary, log path and
  redacted database URL

jarvis-down:
  stops only the daemon it owns through the PID file plus runtime metadata and
  process-command ownership checks
  leaves Jarvis runtime database volume intact

jarvis-reset:
  is explicitly destructive
  requires explicit confirmation, for example CONFIRM=YES
  stops daemon and removes Jarvis runtime database volume only when requested directly
```

Acceptance:

```text
one command can bring up the Jarvis runtime DB, migrations, daemon and health check;
one command opens the new interactive CLI against that daemon;
status and logs are discoverable without ps/lsof/manual Terminal inspection;
the Jarvis runtime database is separate from test database lifecycle;
startup fails clearly if prompt_toolkit or configured local models are missing;
startup never silently falls back to the old plain reader because dependencies
are absent;
normal startup does not install packages, pull models or destroy data;
destructive reset requires explicit confirmation;
status/runtime metadata does not persist raw database credentials;
architecture tests pin the startup script as operational glue, not runtime
business logic.
```

Out of scope:

```text
launchd user services
production deployment
Dockerizing the full daemon
cloud model fallback
automatic model downloads in jarvis-up
voice implementation
```

### PM-08k — Agentic loop-first request handling

Goal:

```text
Use the PM-08k.0 research outcome to reject runtime LLM route classifiers,
Hybrid Request Resolver and broad deterministic intent routing as production
defaults. Make the bounded agent loop the central path for every
natural-language typed request and future voice transcript. Deterministic code
remains for controls, policy and safety, not request understanding.
```

Inputs:

```text
docs/38_pm08k_classifier_contract_simplification.md
docs/39_pm08k_agentic_loop_refactor_plan.md
docs/adr/ADR-035_automatic_loop_strategy_selection.md
docs/34_post_mvp_roadmap.md
docs/37_post_mvp_tdd_slices_plan.md
tests/fixtures/intent_routing/tool_intent_corpus.json
tests/fixtures/intent_routing/pm08i_classifier_model_comparison.json
```

Rationale:

```text
PM-08e/PM-08h proved the classifier boundary and corpus gate, but local
evaluation and review showed that any pre-agent semantic classifier duplicates
the agent's reasoning, adds latency and creates brittle edge cases. Voice should
not inherit a second request-understanding layer because spoken turns make ASR
noise, filler words and near-misses more common.
```

PM-08k.0 research gate:

```text
review industry patterns for tool use, routers, planners, fallback and
abstention;
compare mandatory LLM classifier, deterministic-first router, non-LLM semantic
router, main-model/agent-loop tool choice and hybrid request-resolver
architectures;
record a decision matrix covering latency, safety, local-first operation,
schema adherence, false live-state positives and implementation complexity;
reject mandatory front-gate LLM classifier as the default;
reject Hybrid Request Resolver as the target direction;
accept agentic-loop-first request handling as the target direction.
```

Tests first:

```text
test_pm08k_docs_start_with_industry_research_gate
test_pm08k_research_gate_compares_mandatory_classifier_hybrid_router_and_agent_loop
test_pm08k_research_gate_records_source_links_and_architecture_decision
test_default_request_path_uses_agent_loop_without_route_classifier
test_voice_transcript_uses_same_agent_loop_request_path
test_slash_commands_remain_client_controls_not_backend_routing
test_tool_proposal_requires_allowlisted_tool_name
test_tool_proposal_arguments_validate_before_execution
test_policy_denial_skips_tool_adapter_execution
test_unsupported_calendar_event_does_not_guess_date
test_mixed_natural_language_calculator_request_does_not_truncate_expression
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
PM-08k.0 industry research gate does not exist;
runtime still wires model-route adjudicator before the agent loop;
deterministic natural-language guards still select semantic tool routes;
direct calculator path can truncate mixed natural-language arithmetic;
unsupported event/date questions can bypass unresolved/clarification handling;
PM-09 dependency chain does not mention PM-08k.
```

Implementation:

```text
research:
  maintain PM-08k.0 research note and architecture decision matrix
  keep ADR-035 and this slice plan aligned with the rejected mandatory-classifier
    default, rejected Hybrid Request Resolver direction and accepted
    agentic-loop-first direction

PM-08k.1 Agentic loop default contract:
  make natural-language typed input and future voice transcripts enter the
    bounded agent loop by default
  remove runtime model-route adjudicator, route threshold behavior and
    route-schema parser from the production request path
  remove deterministic natural-language route guards from the default path

PM-08k.2 Control and safety determinism:
  keep deterministic code for slash/control commands, cancellation, approvals,
    policy, permissions, sensitivity, budgets, allowlists, schemas, redaction
    and non-TTY/plain fallback
  forbid deterministic code from acting as a hidden semantic router for normal
    language

PM-08k.3 ReAct/tool loop hardening:
  supply bounded allowed tools to the agent loop
  validate tool proposals through schemas, PolicyPort and ToolGatewayPort
  ensure malformed calculator expressions, unsupported calendar events and
    denied tool calls fail closed or clarify instead of guessing

PM-08k.4 Voice-readiness acceptance gate:
  prove typed input and voice transcripts use the same request lifecycle and
    agent loop
  prove there is no separate voice router and no runtime classifier call before
    the loop
```

Acceptance:

```text
PM-08k.0 research gate is complete;
mandatory front-gate LLM classifier is rejected as the default;
Hybrid Request Resolver is rejected as the target direction;
agentic-loop-first request handling is accepted as the target direction;
natural-language typed input enters the bounded agent loop by default;
voice transcripts are documented to use the same lifecycle and loop;
runtime model-route adjudication and route-threshold tuning are removed;
deterministic code is limited to control/safety/policy responsibilities;
tool proposals cannot bypass PolicyPort, ToolGatewayPort, schemas, sensitivity
ceilings, budgets or approvals;
unsupported/risky tool attempts fail closed or ask clarification through the
agent loop;
PM-09 cannot start until PM-08k is implemented or explicitly rejected by an
updated architecture decision.
```

Follow-up:

```text
remove rejected routing modules follow-up:
  delete or quarantine RequestResolver, route registry, model-route parser,
    threshold config and classifier calibration runtime code once the
    agentic-loop-first implementation is green.
```

Out of scope:

```text
voice audio handling
cloud classifier calls
fine tuning or training pipelines
broad benchmark infrastructure
new tools
direct execution bypasses
CLI-owned routing logic
```

### Agentic request pipeline

PM-08k target pipeline:

```text
explicit chat mode
  -> bounded agent loop with tools disabled or unavailable

explicit tools mode
  -> bounded agent loop with tools required/allowed if policy permits

auto mode
  -> bounded agent loop

tool proposal
  -> allowlist/schema validation
  -> PolicyPort
  -> ToolGatewayPort
  -> typed observation back to the same loop

clear tool proposal while tools are disabled
  -> explicit unavailable-tools observation/result
```

The runtime must not run a natural-language classifier before this loop. The
agent decides whether a tool is useful; Jarvis decides whether the proposal is
allowed and how it executes.

### Historical PM-08a-PM-08h selection model

The following model documents the PM-08a through PM-08h selector/classifier
work. PM-08k supersedes it for production natural-language handling. Keep these
objects only where needed for migration, compatibility or evaluation; do not use
them as the default request path before the bounded agent loop.

PM-08 model objects:

```text
LoopSelectionMode:
  auto
  chat
  tools
  invalid_override, internal audit-only mode for rejected malformed overrides

IntentFamily:
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

SelectionDecisionStatus:
  selected
  fallback_chat
  rejected_by_policy
  tools_unavailable
  invalid_override
  classifier_unavailable

SelectionFallbackPreference:
  chat
  fail_unavailable
  ask_clarification
```

PM-08 routes only implemented intent families:

```text
ordinary_chat
project_docs_question
project_inspection
system_diagnostics
safe_builtin_tool
unknown
```

Future families may exist in the enum, but must stay disabled until their tools
and policy slices exist.

`LoopSelectionRequest` includes:

```text
request_id
conversation_id
user_id
requested_mode
user_input
current_message_sensitivity
active_project_namespace
working_directory
permission_mode
available_capabilities
available_tools_summary
runtime_budget_summary
model_profile_override optional
metadata
```

`CapabilityCandidate` includes:

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

`IntentClassification` includes:

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

`LoopSelectionDecision` includes:

```text
requested_mode
selected_loop_strategy
selected_model_profile optional, filled after selected_loop_strategy is known
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

Rules:

```text
requested_mode is persisted separately from selected_loop_strategy;
raw user_input is transient and must not be copied into selection events;
evidence_codes must be stable non-sensitive labels, not prompt snippets;
candidate tool_names are references only, not execution instructions;
selector may execute a loop only for selected or fallback_chat decisions.
```

### Historical confidence model

The confidence bands below are selector/classifier history. PM-08k production
behavior should not depend on route-confidence thresholds before the agent loop.

Confidence is normalized:

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
  -> tool_react_loop if policy/config allow

high confidence chat or project_docs_question
  -> memory_augmented_answer

medium confidence
  -> conservative fallback to memory_augmented_answer

low confidence
  -> memory_augmented_answer with classifier_low_confidence reason

answer_without_tools_would_be_misleading=true and tools unavailable
  -> tools_unavailable or rejected_by_policy, not fake chat
```

### Capability and tool metadata

Every auto-routable capability/tool should expose:

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

This metadata may guide historical classifier implementations and may also help
present available tools to the bounded agent loop. It is not authorization.

Future code sandbox metadata shape:

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
negative_examples:
  explain this code
  write an example but do not run it
default_selection_policy:
  developer_local: approval_required or allow only for read-only sandbox
  locked_down: approval_required or deny
  automation: deny by default
```

Adding code sandbox later should add capability metadata, policy and agent-loop
tool proposal tests. It should not require ad hoc code branches in a selector or
pre-loop classifier.

### IntentClassifierPort behavior

Classifier input:

```text
user request text
requested mode
active project namespace
available intent families
available capabilities and tool descriptions
permission mode summary
```

Classifier output:

```text
intent_family
requires_live_state
candidate_capabilities
confidence
answer_without_tools_would_be_misleading
reason_code
```

Initial intent families:

```text
ordinary_chat
project_docs_question
project_inspection
system_diagnostics
safe_builtin_tool
unknown
```

Future intent families may include:

```text
code_execution
external_integration
planner_task
background_workflow
```

### RAG rule

Project Docs RAG is not a tool-loop trigger.

The expected path remains:

```text
memory_augmented_answer
  -> ContextAssembler
      -> ContentRetrievalPort
```

This keeps documentation questions cheap, deterministic and covered by golden
context tests.

### Policy behavior

Policy remains authoritative:

```text
selector proposes candidate capabilities
PolicyPort decides allow / deny / approval_required
ToolGateway enforces policy again for actual tool calls
```

The selector must not grant access. A selected `tool_react_loop` only means the
runtime may attempt tool-capable execution under existing policy and budgets.

### Event and audit behavior

PM-08 adds redacted events:

```text
request.loop_selection.started
request.loop_selection.completed
request.loop_selection.failed
```

Completed event payload:

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

### Acceptance criteria

PM-08 is complete only when:

```text
tests were added before production code in each sub-slice;
PM-08a red phase failed for missing selector/domain behavior;
PM-08b red phase failed for missing API auto/request metadata behavior;
PM-08c red phase failed for missing CLI mode behavior;
PM-08d red phase failed for missing CLI tool/RAG/approval readiness behavior;
PM-08e red phase failed for missing model-backed classifier behavior;
PM-08f red phase failed for missing typed tool-observation/direct-answer
  behavior;
PM-08g red phase failed for missing registry-backed direct planning behavior;
PM-08h red phase failed for missing pre-voice corpus gate behavior;
PM-08i red phase failed for missing prompt_toolkit shell/status/completion
  behavior;
PM-08j red phase failed for missing canonical Jarvis runtime startup behavior;
PM-08k red phase failed for missing agentic-loop-first request contract and
  removal/quarantine of production classifier-first routing;
missing loop_strategy means auto request-plan policy, not selector-era direct
  memory loop;
ordinary chat, project-docs questions and live project/system inspection enter
  the bounded agent loop by default;
RAG remains ContextAssembler behavior and is not a separate route trigger;
safe tool use happens only through bounded agent loop proposals and ToolGateway;
tools-disabled tool intent does not silently hallucinate;
direct diagnostics answers use typed payloads and parse_status, not raw stdout
  parsing in the loop;
CLI defaults to auto and exposes debug override;
interactive CLI has /mode auto|chat|tools;
interactive CLI can drive approval-required tool flows;
interactive CLI can cancel active requests and remain usable;
CLI renders tool proposals/results in user-facing language, not raw JSON-first;
interactive CLI has a Codex-like status line, phase-based activity indicator,
dynamic slash command palette and deterministic --plain/non-TTY fallback;
canonical Jarvis runtime startup can bring up DB, migrations, daemon, health
check and CLI entrypoint without relying on manual Terminal orchestration;
natural-language typed input and future voice transcripts enter the same
bounded agent loop by default;
runtime model-route adjudication, classifier thresholds and broad deterministic
natural-language routing are removed from the production request path;
model-origin tool proposals are validated by schema, policy, approval and
ToolGateway before execution;
loop-selection events are redacted;
architecture guardrails pass;
no real LLM, real shell or host diagnostics calls are required for CI.
```

### Out of scope

Do not implement:

```text
planner-executor routing
multi-agent handoff
automatic external integration routing
background scheduler routing
remembered per-user mode preferences
cloud model fallback
provider-native tool execution bypassing ToolGateway
RAG as a tool-loop requirement
```

## 12. Slice PM-08l — Agent loop architecture hardening gate

### Goal

Close the final architecture-review findings that can make PM-09 start with
false confidence. PM-08l is not a voice implementation slice; it hardens the
post-PM-08k bounded agent loop architecture and then proves that voice can sit on
top of the same request path.

Detailed plan:

```text
docs/40_pm08l_agent_loop_architecture_hardening_plan.md
```

Target architecture:

```text
simple natural-language input or future voice transcript
  -> one bounded agent loop
  -> explicit step state machine
  -> optional tool action
  -> typed observation
  -> recovery, clarification or final answer

future compound task
  -> plan-and-execute shell
  -> each executable step uses the same bounded agent loop
```

### Tests first

```text
test_agent_loop_state_machine_records_expected_step_order
test_agent_loop_final_answer_uses_single_finalizer
test_agent_loop_auto_mode_plain_chat_survives_bad_non_tool_proposal
test_agent_loop_auto_mode_non_tool_question_survives_tool_observation_failure
test_agent_loop_auto_mode_arithmetic_survives_tool_observation_failure
test_agent_loop_cpu_usage_failure_returns_typed_recovery_or_actionable_failure
test_agent_loop_required_tools_mode_does_not_fallback_before_observation
test_agent_loop_denied_tool_observation_uses_recovery_policy
test_agent_loop_unavailable_tool_observation_preserves_typed_reason
test_agent_loop_cancel_or_approval_denial_has_controlled_terminal_state
test_agent_loop_tools_mode_budget_and_unavailable_matrix
test_agent_loop_stream_reconnect_replays_terminal_state
test_agent_loop_context_manifest_records_final_observation_refs
test_agent_loop_preserves_future_plan_step_metadata_without_planner_bypass
test_transcript_like_api_turn_uses_agent_loop_lifecycle
test_transcript_like_tool_turn_uses_toolgateway
test_default_runtime_can_execute_default_agent_loop_without_tool_registry
test_runtime_app_validates_request_plan_tools_match_gateway_registry
test_pm09_docs_gate_on_full_pm08_sequence_through_pm08l
```

### Implementation

```text
contract/docs:
  align PM-08l with the agent-loop architecture hardening plan
  replace old generic budget-exhaustion wording with the precise PM-08l
    budget/finalization matrix

runtime loop:
  keep ToolReactLoop as the public implementation vehicle until behavior is green
  extract explicit AgentLoopState and AgentLoopStep internals
  extract LoopEventRecorder so lifecycle events do not remain scattered through
    the loop body
  extract FinalAnswerStep so tools-disabled, final_answer proposal, safe budget
    exhaustion and malformed non-tool fallback all use one finalization path
  extract ToolObservationRecoveryPolicy and LoopFailurePolicy for denied,
    unavailable, failed, approval and malformed-output outcomes

auto/tools modes:
  auto mode must allow ordinary final answers without requiring a tool
    observation
  explicit malformed tool_call proposals still fail closed
  tools mode must not silently fallback before a valid tool observation when a
    tool observation is required
  live regressions such as "где раки зимуют?" and "Двадцать два в третьей
    степени." must not surface as generic `tool_observation_failed` when the
    request can be answered, clarified or safely finalized without the failed
    tool path

streaming/events:
  keep CLI/activity phases tied to real lifecycle events
  emit one terminal completion or failure
  support durable terminal replay after buffer cleanup or daemon restart through
    persisted request status, event log and conversation messages
  document any remaining lack of true token streaming as a PM-09 input if not
    fixed in PM-08l

e2e/API:
  add DB-backed transcript-like requests through the existing
    /v1/conversations/{id}/messages lifecycle
  cover one no-tool final-answer turn and one tool turn through ToolGateway

runtime composition:
  keep RuntimeTurnCommand, AgentRuntime default registry and legacy persisted
    request fallback aligned with the bounded agent loop
  direct runtime construction must not fail merely because no concrete tool
    registry was injected
  validate that request-plan allowed tool names are registered in the actual
    ToolGateway surface
  preserve compatibility with a future plan-and-execute shell where scoped plan
    steps execute through the same bounded agent loop, PolicyPort and
    ToolGatewayPort path

verification:
  each PM-08l milestone requires a verification gate and two read-only review
    agents from scratch after tests are green
  PM-09 entry requires the final DB-enabled gates, not only bare pytest:
    make test-unit
    make test-contract
    make test-golden
    make test-integration
    make test-e2e
    make test-architecture
```

### Acceptance

```text
PM-08l complete
spoken-transcript-like API turns enter the same request lifecycle and bounded
  agent loop as typed turns;
transcript-like tool turns execute only through ToolGateway;
AgentRuntime direct construction is aligned with the default agent loop;
request-plan tool availability cannot drift silently from ToolGateway registry;
ToolReactLoop no longer owns every state transition, event, proposal, final
  answer and failure policy in one large orchestration method;
final answers use one shared finalization path;
auto/chat/tools behavior is covered by unit, contract and e2e tests;
denied, unavailable, failed, approval and malformed-output outcomes have typed
  recovery, clarification or controlled-failure semantics;
lifecycle streaming emits stable phases and a single terminal event;
bounded agent loop can become the executor for future plan-and-execute scoped
  steps without adding a planner-specific router or ToolGateway bypass;
PM-09 cannot start until the DB-enabled preflight gates are green.
```

### Out of scope

```text
voice audio capture
STT/TTS providers
wake-word implementation
renaming loop_selection event types
storage/content application-service extraction
full planner-executor or plan-and-execute implementation
new production tools
provider-native tool calling migration
LangGraph or durable workflow checkpointing
semantic classifier or deterministic natural-language route resolver
```

## 13. Slice PM-09 — Voice gateway foundation

### Goal

Add the first voice assistant path after PM-08d proves the text CLI/API surface
can use chat, RAG, tools, approvals and cancellation, after PM-08f hardens tool
answers around typed observations rather than loop-level stdout parsing, after
PM-08g/PM-08h record routing/corpus evidence for spoken-transcript-like cases,
and after PM-08i makes the interactive CLI shell usable enough to dogfood the
same surface before voice. PM-08j must then make that Jarvis surface
operationally repeatable through canonical startup/status/log/shutdown commands.
PM-08k must then replace classifier-first routing with agentic-loop-first
request handling before spoken turns rely on the same path. PM-08l must then
prove that path with DB-backed transcript-like API/e2e turns and startup
invariants before any voice code is added.

Voice is a client/channel over the existing runtime, not a separate agent
runtime. A spoken turn must become the same kind of user turn that CLI/API uses:

```text
audio input
  -> SpeechToTextPort
  -> existing conversation/message/request lifecycle
  -> PM-08k bounded agent loop
  -> existing runtime/SSE stream
  -> TextToSpeechPort
  -> audio output
```

PM-09 starts with push-to-talk/local-session semantics. Wake word, always-on
listening, realtime cloud models and barge-in are deferred.

### Inputs

Required docs:

```text
docs/adr/ADR-035_automatic_loop_strategy_selection.md
docs/adr/ADR-029_capability_and_permission_model.md
docs/10_api_and_streaming.md
docs/17_data_sensitivity_and_privacy_policy.md
docs/22_api_shape_and_request_lifecycle.md
docs/26_testing_strategy.md
docs/37_post_mvp_tdd_slices_plan.md
```

Before implementation, write or promote a dedicated Voice Gateway ADR covering:

```text
push-to-talk vs wake word sequence
audio capture/playback boundaries
SpeechToTextPort
TextToSpeechPort
modular speech provider profiles
transcript sensitivity
audio retention policy
interrupt/cancel behavior
local-first provider policy
external speech API policy
```

### Dependencies

PM-09 depends on all PM-08 sub-slices:

```text
PM-08a complete
PM-08b complete
PM-08c complete
PM-08d complete
PM-08e complete
PM-08f complete
PM-08g complete
PM-08h complete
PM-08i complete
PM-08j complete
PM-08k complete
PM-08l complete
```

In practice this means the text CLI already has a working normal chat surface
for shared chat/RAG/tools handling, approval control, cancellation, typed tool
observations, transcript-like corpus evidence and Codex-like interactive shell
UX, plus a canonical local Jarvis startup path and an agentic-loop-first request
contract, before voice is layered on top.

Voice must use the PM-08k/PM-08l request path:

```text
spoken request -> transcript -> bounded agent loop -> request-plan policy
  -> policy/tool gates
```

The voice channel must not implement separate intent routing, a separate
classifier or transcript-specific deterministic natural-language router.

### Tests first

Unit tests:

```text
test_voice_turn_requires_transcript_text
test_voice_turn_uses_auto_mode_by_default
test_voice_turn_records_transcript_sensitivity
test_voice_config_disables_audio_storage_by_default
test_voice_session_can_cancel_active_request
test_fake_stt_returns_transcript_without_real_audio_device
test_fake_tts_returns_audio_ref_without_real_speaker
test_speech_provider_profile_supports_local_and_external_api_kinds
test_external_speech_provider_requires_explicit_policy_allow
test_external_speech_provider_uses_secret_reference_not_raw_secret
```

Contract tests:

```text
test_speech_to_text_port_transcribes_audio_input
test_text_to_speech_port_synthesizes_assistant_text
test_voice_gateway_submits_transcript_through_existing_request_lifecycle
test_voice_gateway_streams_runtime_events_without_private_audio_payloads
test_voice_gateway_does_not_store_raw_audio_when_retention_disabled
test_voice_gateway_uses_provider_registry_not_concrete_provider
```

Architecture tests:

```text
test_voice_gateway_does_not_import_concrete_model_providers
test_voice_gateway_does_not_import_external_speech_api_clients
test_voice_gateway_does_not_bypass_api_runtime_or_agent_runtime
test_stt_tts_adapters_do_not_import_conversation_storage
test_voice_channel_uses_agent_loop_request_plan_contract_not_custom_router
```

E2E smoke with fake STT/TTS/model providers:

```text
test_fake_voice_turn_transcribes_routes_answers_and_synthesizes
test_fake_voice_turn_can_trigger_toolgateway_from_transcript
test_fake_voice_cancel_stops_active_request
```

### Expected red phase

The first test run should fail because:

```text
SpeechToTextPort does not exist
TextToSpeechPort does not exist
VoiceGateway does not exist
voice turn submission path does not exist
fake STT/TTS adapters do not exist
```

Bad red failures:

```text
test requires microphone or speaker hardware
test requires real STT/TTS provider
test requires cloud model or cloud speech service
test stores raw audio by default
test bypasses PM-08k/PM-08l bounded agent loop request-plan policy
```

### Implementation

Minimal production changes:

```text
domain/voice:
  VoiceTurnRequest
  VoiceTurnResult
  AudioInputRef
  AudioOutputRef
  VoiceSessionState

ports/voice:
  SpeechToTextPort
  TextToSpeechPort

voice:
  VoiceGateway
  SpeechProviderProfile
  SpeechProviderRegistry
  FakeSpeechToTextAdapter
  FakeTextToSpeechAdapter

cli/api:
  thin voice entrypoint or local voice command after ADR approval
```

The first implementation may expose voice through a local CLI command or a
small local API endpoint, but the core contract is the voice gateway and fake
ports. Provider-specific local STT/TTS adapters can be added after the contract
is green.

Speech providers must be modular from the start:

```text
local_process
local_library
external_api
fake
```

The voice gateway must depend on provider-neutral profiles and the
`SpeechToTextPort` / `TextToSpeechPort` contracts only. It must not branch on a
specific local engine, command-line binary, SDK or external API client.

External API speech providers are a future adapter path, not the PM-09 default.
They must be disabled unless explicitly configured and allowed by policy. Secret
values must be referenced by secret id or environment key and must not be copied
into prompts, events, logs, transcripts or memory.

### Policy and privacy

Defaults:

```text
cloud speech providers: disabled
external speech API providers: disabled unless explicitly configured and allowed
audio storage: disabled
transcript storage: same conversation/message rules as typed input
transcript sensitivity: personal by default unless configured otherwise
wake word / always listening: disabled
```

Voice must not store raw audio unless an explicit future ADR/config option
enables it.

### Acceptance criteria

PM-09 is complete only when:

```text
tests were added before production code;
red phase failed for missing voice ports/gateway;
fake STT/TTS e2e voice turn works without hardware;
voice submits transcript through existing runtime lifecycle;
PM-08k/PM-08l bounded agent loop request-plan policy modes are used for spoken
  turns;
voice gateway is provider-neutral and can select local/fake/external-api
provider profiles without importing concrete clients;
raw audio is not stored by default;
cloud speech/model providers remain disabled;
architecture guardrails pass.
```

### Out of scope

Do not implement:

```text
wake word
always-on listening
cloud realtime model path
external speech API adapter implementation
barge-in/interruption beyond request cancel
speaker diarization
multi-user voice identity
audio archive/search
LangGraph voice workflow
```

## 13. Follow-up — Graph runtime adapter and LangGraph adoption gate

### Goal

Evaluate whether LangGraph should become the graph runtime adapter for complex
future workflows.

This is a follow-up, not part of the immediate PM-08 -> PM-09 voice path. It
should be reopened before planner-executor, durable code sandbox workflows,
sleep/reflection execution or long-running background workflows.

The follow-up must answer:

```text
Can LangGraph run behind Jarvis ports without taking over API, CLI, storage,
policy or ToolGateway boundaries?
Can LangGraph checkpoints/interrupts map cleanly to Jarvis request status,
approval records, SSE and EventLog semantics?
Should planner-executor, code sandbox workflows or sleep/reflection use
LangGraph, custom loops, or another durable workflow runtime?
```

### Inputs

Required docs:

```text
docs/adr/ADR-006_langgraph_as_phase_1_runtime_substrate.md
docs/adr/ADR-031_agent_loop_strategy_architecture.md
docs/adr/ADR-035_automatic_loop_strategy_selection.md
docs/35_post_mvp_adr_backlog.md
docs/06_agent_runtime_and_loop_architecture.md
docs/26_testing_strategy.md
```

External references to inspect before implementation:

```text
LangGraph overview
LangGraph persistence/checkpointing
LangGraph interrupts / human-in-the-loop
LangGraph workflows and agents / routing / conditional edges
```

### Scope

This follow-up may add an optional LangGraph dependency only if the adapter
spike cannot be evaluated with a local abstraction/fake first.

Preferred implementation order:

```text
1. define Jarvis GraphRuntimePort / GraphWorkflowAdapter contract;
2. add fake graph runtime adapter for CI;
3. add an optional LangGraph adapter spike behind that contract;
4. compare adapter behavior against existing custom loops;
5. document adopt/defer decision.
```

Do not switch production request execution to LangGraph in this follow-up.

### Tests first

Unit tests:

```text
test_graph_runtime_request_contains_request_and_correlation_ids
test_graph_runtime_state_contains_only_domain_safe_fields
test_graph_runtime_decision_can_report_adopt_or_defer
test_graph_interrupt_maps_to_waiting_approval_status
test_graph_resume_requires_same_request_or_thread_identity
test_graph_adapter_does_not_store_raw_prompt_in_checkpoint_metadata
```

Contract tests:

```text
test_fake_graph_runtime_executes_single_node_workflow
test_fake_graph_runtime_emits_jarvis_events
test_graph_runtime_uses_context_model_policy_tool_ports
test_graph_runtime_maps_interrupt_to_approval_record
test_graph_runtime_resume_maps_approval_decision_back_to_workflow
test_graph_runtime_failure_maps_to_request_failure_status
```

Architecture tests:

```text
test_graph_adapter_does_not_import_api_or_cli
test_graph_adapter_does_not_import_storage_adapters_directly
test_graph_nodes_use_ports_not_concrete_tools
test_custom_loop_strategies_do_not_require_langgraph
test_langgraph_dependency_is_isolated_to_graph_adapter_package
```

E2E smoke with fake graph adapter:

```text
test_fake_graph_workflow_can_pause_for_approval_and_resume
test_fake_graph_workflow_streams_jarvis_runtime_events
test_existing_memory_augmented_answer_still_runs_without_graph_runtime
test_existing_tool_react_loop_still_runs_without_graph_runtime
```

### Expected red phase

The first test run should fail because:

```text
GraphRuntimePort does not exist
GraphWorkflowAdapter does not exist
GraphWorkflowDecision does not exist
graph interrupt/resume mapping does not exist
graph adapter package does not exist
```

Bad red failures:

```text
test requires LangGraph installation before the fake adapter contract exists
test bypasses Jarvis EventLogPort or ApprovalStorePort
test makes real model/tool calls
test rewires production request execution to LangGraph
test stores raw prompt text in graph checkpoint metadata
```

### Adapter boundary

Graph-backed workflows may use:

```text
ContextAssemblerPort
ModelRouterPort
ToolGatewayPort
PolicyPort
EventLogPort
ConversationStorePort
ApprovalStorePort
domain schemas
```

Graph-backed workflows must not import:

```text
FastAPI route modules
CLI modules
SQLAlchemy models
concrete storage adapters directly
provider clients directly
shell or diagnostics adapters directly
```

### LangGraph evaluation criteria

Adopt LangGraph for future complex workflows only if the follow-up demonstrates:

```text
checkpoint identity maps cleanly to Jarvis request/correlation identity;
interrupt/resume maps cleanly to approval records and waiting_approval status;
streaming can be translated into Jarvis SSE/runtime events;
graph state can be redacted and kept free of raw prompt/secret leakage;
graph nodes can call only Jarvis ports;
custom loop strategies continue to run without LangGraph;
CI can use fake graph runtime without a real LLM or host-specific tools;
dependency footprint is acceptable for local-first runtime.
```

Defer LangGraph if:

```text
checkpoint persistence duplicates Jarvis storage too heavily;
interrupt/resume semantics fight the existing approval model;
graph state redaction is too hard to guarantee;
adapter code becomes more complex than the workflows it replaces;
dependency footprint or operational behavior is not acceptable.
```

### Output

The follow-up must end with a written decision update:

```text
adopt LangGraph for planner/code-sandbox/sleep workflows;
or defer LangGraph and continue custom loop/workflow runtime;
or keep LangGraph only as an optional adapter for selected workflows.
```

If adopted, write or promote the corresponding ADR before implementing
planner-executor or durable code sandbox workflows.

### Acceptance criteria

The follow-up is complete only when:

```text
tests were added before production code;
red phase failed for missing graph adapter boundary;
fake graph runtime contract is green;
architecture guardrails isolate LangGraph behind adapter package;
approval interrupt/resume mapping is tested with fake graph runtime;
existing custom loops still pass without LangGraph;
adopt/defer decision is documented;
no production behavior switch occurred without a follow-up ADR/slice.
```

### Out of scope

Do not implement:

```text
planner-executor product workflow
code sandbox execution
sleep/reflection execution
LangGraph production migration
LangGraph checkpoint tables as system of record
multi-agent orchestration
LangSmith/cloud deployment
```
