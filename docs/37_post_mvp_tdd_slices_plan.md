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
```

PM-01 through PM-07b are accepted in detail here. Later slices should be
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
memory_augmented_answer remains the default.
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
memory_augmented_answer is selected by default;
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
budget exhaustion fails safely;
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
test_allows_rg_inside_workspace
test_allows_sed_n_with_bounded_range
test_allows_head_and_tail_with_bounded_line_count
test_allows_wc_inside_workspace
test_allows_git_status_short
test_allows_git_diff_read_only
test_allows_git_show_read_only
test_denies_shell_metacharacters
test_denies_path_outside_workspace
test_denies_symlink_escape_outside_workspace
test_denies_secret_like_paths
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
test_allows_macos_top_snapshot
test_allows_macos_vm_stat
test_allows_macos_sysctl_selected_keys
test_allows_linux_top_batch_snapshot
test_allows_linux_free
test_allows_linux_lscpu
test_allows_linux_lshw
test_allows_network_diagnostics_selected_flags
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
test_system_diagnostics_tool_redacts_network_sensitive_output
test_sensor_backend_unavailable_returns_unavailable_observation
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
Linux: top -b -n 1, free, lscpu, lshw, ss/netstat, ip addr, lsof, nvidia-smi
```

Temperature and sensor diagnostics:

```text
macOS: powermetrics --samplers smc -n 1 if available without sudo
Linux: sensors
Linux: read-only /sys/class/thermal/thermal_zone*/temp adapter
Linux GPU: nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
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
test_markdown_chunker_splits_by_headings
test_markdown_chunker_splits_oversized_sections
test_chunk_preserves_heading_path
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
test_reingestion_marks_old_chunks_stale
test_deleted_source_marks_source_and_chunks_deleted_or_stale
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
  deterministic markdown heading chunker
  citation builder
  content_hash change detection
  stale/deleted source handling

events:
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
secret-like paths are denied;
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
