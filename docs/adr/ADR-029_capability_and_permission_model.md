# ADR-029 — Capability and Permission Model

## Status

Accepted.

## Context

The MVP core daemon is local-first, durable and auditable. Current post-MVP
Alpha has added bounded tool use, approval-gated actions and project-docs
retrieval, while MCP, planner-executor, voice, Telegram and other external
integrations remain future scope.

Post-MVP capabilities can read local data, call tools, execute shell commands,
access external services, retrieve content and later perform side effects.
Those capabilities must not be wired directly into `AgentRuntime`.

The existing Phase 1 policy model already handles:

- model request policy;
- memory write policy;
- context inclusion policy;
- local-first cloud denial;
- secret exclusion from memory, prompt context and raw logs.

That policy model is sufficient for MVP, but insufficient for post-MVP tools
and integrations. Before adding ToolGateway, shell, MCP, RAG or voice, the
system needs a capability and permission model that can answer:

```text
What is this action trying to do?
What data sensitivity does it touch?
Can it read, write, call network, or create external side effects?
Does it require user approval?
What should be audited?
```

## Decision

Introduce a post-MVP capability and permission model as an extension of
`PolicyPort`, not as a replacement for the existing sensitivity model.

The model has five layers:

```text
Capability
  what kind of system power is being requested

Risk class
  what category of impact the action can have

Permission scope
  where and under which constraints the action is allowed

Approval requirement
  whether user approval is required before execution

Audit record
  durable event log record of decision and execution
```

## Capability taxonomy

Capabilities are stable domain-level identifiers. They are not adapter class
names and not provider-specific tool names.

Initial capability families:

```text
model.local
model.cloud

memory.read
memory.write
memory.lifecycle

context.inspect

content.retrieve
content.ingest
content.index

tool.safe
tool.shell.read
tool.shell.write
tool.shell.network
tool.shell.destructive

tool.filesystem.read
tool.filesystem.write

integration.mcp
integration.search
integration.telegram
integration.spotify
integration.github
integration.calendar
integration.mail

task.schedule
task.background
task.sleep_reflection

voice.input
voice.output
voice.realtime

approval.request
approval.grant
approval.deny
```

Rules:

- unknown capabilities are denied by default;
- new capability families require config and tests;
- risky capability families require an ADR or ADR update before implementation;
- adapters may expose concrete tools, but policy decisions use capability
  identifiers.

## Risk classes

Each action is classified before execution.

Initial risk classes:

```text
safe
read_only
writes_local
network
external_side_effect
secrets
destructive
autonomous
cloud
```

Meaning:

- `safe`: deterministic local operation with no sensitive data and no side
  effect, for example calculator or datetime;
- `read_only`: reads local/project data without mutation;
- `writes_local`: mutates local files, local database state or local assistant
  state;
- `network`: contacts a network resource;
- `external_side_effect`: sends messages, creates issues, changes external
  services or otherwise acts outside the local machine;
- `secrets`: touches credentials, tokens, private keys or secret-bearing
  material;
- `destructive`: deletes, overwrites, purges or irreversibly changes state;
- `autonomous`: runs outside the immediate user-turn interaction;
- `cloud`: sends data to cloud model/provider/service.

An action may have multiple risk classes. Policy evaluates the highest-risk
combination, not only the first class.

## Permission scopes

Permissions are scoped. The system must avoid broad global grants.

Initial scope dimensions:

```text
user_id
conversation_id
request_id
task_id
project_namespace
working_directory
integration_id
tool_name
capability
sensitivity_ceiling
expires_at
max_uses
```

Alpha default:

- no persistent remembered approvals at first;
- approvals apply to one request or one task step unless explicitly configured
  otherwise;
- broad remembered permissions are deferred.

## Permission modes

Permission decisions should not be controlled by one global allow/deny switch.
Post-MVP Jarvis needs explicit permission modes that tune defaults for a usage
context while keeping the same capability, risk, scope and sensitivity model.

Mode is an input to policy evaluation:

```text
capability + risk_classes + scope + sensitivity + permission_mode
  -> allow | deny | approval_required
```

Modes do not bypass hard security rules. The following remain denied unless a
later ADR changes policy:

```text
secret access
automatic cloud fallback
destructive shell
persistent remembered approvals
irreversible external side effects
```

### Initial modes

The first implementation should support only these modes:

```text
locked_down
developer_local
automation
```

`locked_down`:

- safe tools are allowed;
- project content retrieval requires approval or is denied by configuration;
- read-only shell requires approval;
- writes, network, external side effects, cloud and destructive actions are
  denied by default.

Use when handling sensitive data, testing a new model/profile or debugging
policy behavior.

`developer_local`:

- safe tools are allowed;
- project docs retrieval is allowed;
- context inspection is allowed for manifests/refs only;
- read-only shell is allowed only inside allowlisted working directories;
- shell writes require approval;
- network shell, destructive shell and cloud are denied by default;
- external side effects require approval.

This is the recommended Alpha default for local development.

`automation`:

- safe tools are allowed;
- read-only project checks are allowed in configured scopes;
- scheduled/background work is allowed only for explicitly configured workflows;
- writes require approval or are denied by workflow policy;
- direct autonomous memory writes are denied;
- memory candidates are allowed;
- destructive actions, cloud and secret access are denied.

Use for reminders, maintenance checks and future sleep/reflection workflows.

### Planned modes

These modes are reserved for later design and should not be required in the
first implementation:

```text
review
trusted_project
break_glass
```

`review` is a cautious interactive mode where read-only shell also requires
approval.

`trusted_project` may allow selected project-local write operations such as
formatters, tests or generated artifacts inside a specific workspace.

`break_glass` is a manually enabled, time-limited emergency mode with stricter
audit and explicit confirmation requirements. It must not become a default
runtime mode.

## PolicyPort extensions

Post-MVP `PolicyPort` should grow explicit methods for capability decisions.

Potential shape:

```python
class PolicyPort(Protocol):
    async def evaluate_model_request(self, request: ModelPolicyRequest) -> PolicyDecision: ...
    async def evaluate_memory_write(self, request: MemoryWritePolicyRequest) -> PolicyDecision: ...
    async def evaluate_context_inclusion(self, request: ContextPolicyRequest) -> PolicyDecision: ...

    async def evaluate_capability_request(
        self,
        request: CapabilityPolicyRequest,
    ) -> PolicyDecision: ...

    async def evaluate_tool_call(
        self,
        request: ToolCallPolicyRequest,
    ) -> PolicyDecision: ...

    async def evaluate_content_access(
        self,
        request: ContentAccessPolicyRequest,
    ) -> PolicyDecision: ...

    async def evaluate_approval(
        self,
        request: ApprovalPolicyRequest,
    ) -> PolicyDecision: ...
```

The exact domain object names may change during implementation, but the
boundary must remain:

```text
runtime/tool/content/voice adapters ask PolicyPort before doing the action
PolicyPort returns allow, deny or approval_required
EventLog records the decision
execution happens only after allow or granted approval
```

## Policy decision outcomes

Policy decisions use a closed outcome set:

```text
allow
deny
approval_required
```

Required fields:

```text
decision_id
outcome
capability
risk_classes
subject
reason
sensitivity
scope
created_at
expires_at optional
redacted_payload
```

`reason` must be stable enough for tests and audit. Examples:

```text
unknown_capability
cloud_disabled
secret_access_denied
approval_required_for_write
destructive_action_denied
outside_allowed_workspace
network_denied_by_default
allowed_safe_tool
```

## Default policy for Alpha

The recommended Alpha default mode is:

```text
developer_local
```

Default deny:

```text
unknown capabilities
cloud
secret access
destructive actions
network shell commands
autonomous background work
filesystem writes
```

Allowed without approval:

```text
model.local within existing sensitivity rules
memory.read through existing MemoryReadPort policy
context.inspect for non-secret manifest data
tool.safe
tool.shell.read in allowlisted working directories
content.retrieve for project docs with project sensitivity
```

Approval required:

```text
tool.shell.write
tool.filesystem.write
integration side effects
other external side effects
task.schedule
task.background
memory lifecycle actions proposed by autonomous workflows
```

Denied until later ADR:

```text
model.cloud
tool.shell.destructive
voice.realtime cloud path
persistent remembered approvals
secret manager access
```

## Approval model baseline

Approvals are separate domain decisions, not free-form CLI prompts.

Baseline lifecycle:

```text
approval.required
approval.granted
approval.denied
approval.expired
approval.cancelled
```

Alpha behavior:

- approvals are explicit;
- approvals are scoped to one action, request or task step;
- approval payloads are redacted;
- approval grants expire;
- denied or expired approvals prevent execution;
- cancelled approvals prevent execution;
- approval decisions are recorded in EventLog.

Transport details are implemented by PM-05. Alpha starts with HTTP grant/deny
endpoints, CLI prompts and SSE/runtime events. WebSocket remains a later
control-channel decision.

## Audit events

Capability and permission decisions must be auditable.

New event types should include:

```text
policy.capability.decision.recorded
approval.required
approval.granted
approval.denied
approval.expired
approval.cancelled
```

Tool-specific lifecycle events are decided in the ToolGateway ADR, but they must
reference the related policy decision:

```text
tool.call.requested
tool.call.approved
tool.call.denied
tool.call.started
tool.call.completed
tool.call.failed
tool.observation.recorded
```

Audit rules:

- do not store raw secrets;
- do not store full raw prompts;
- store redacted inputs, hashes, refs and stable decision reasons;
- record denied decisions as well as allowed decisions.

## Configuration model

Initial configuration should be explicit and conservative.

Potential shape:

```yaml
permissions:
  mode: developer_local
  modes:
    locked_down:
      tool.safe: allow
      content.retrieve: approval_required
      context.inspect: allow
      tool.shell.read: approval_required
      tool.shell.write: deny
      tool.shell.network: deny
      tool.shell.destructive: deny
      model.cloud: deny

    developer_local:
      tool.safe: allow
      content.retrieve: allow
      context.inspect: allow
      tool.shell.read: allow
      tool.shell.write: approval_required
      tool.shell.network: deny
      tool.shell.destructive: deny
      integration.telegram: approval_required
      model.cloud: deny

    automation:
      tool.safe: allow
      content.retrieve: allow
      context.inspect: allow
      tool.shell.read: allow
      tool.shell.write: approval_required
      task.schedule: approval_required
      task.background: approval_required
      memory.write: deny
      model.cloud: deny

capabilities:
  tool.shell.read:
    allowed_roots:
      - "."
    max_output_bytes: 20000
    timeout_seconds: 10
```

Config is not a policy DSL. The initial `ConfigPolicyEngine` should remain
small and explicit.

## Boundary rules

Agent Runtime:

- may request capability decisions;
- must not execute tools directly;
- must not call subprocess, MCP, search, Telegram, Spotify, filesystem adapters
  or voice providers directly.

ToolGateway:

- must ask PolicyPort before execution;
- must record tool lifecycle events;
- must enforce output limits and timeout limits.

Content Retrieval:

- must ask PolicyPort before source access;
- must not store document chunks in Memory tables;
- must label content hits with sensitivity.

Shell adapter:

- must classify commands before execution;
- must deny outside allowlisted roots;
- must enforce timeout/output caps;
- must redact environment values.

Voice gateway:

- must label transcripts with sensitivity;
- must not persist audio by default;
- must keep STT/TTS providers behind provider-neutral ports and profiles;
- must disable external speech API providers unless explicit configuration and
  policy allow them;
- must reference external speech API secrets by secret id or environment key;
- must not use cloud realtime path without a later ADR.

## Rationale

This keeps the local-first assistant extensible without turning future features
into one-off runtime shortcuts.

The model is deliberately simple:

- capabilities say what power is requested;
- risk classes say why the action may be dangerous;
- scopes prevent broad permissions;
- approvals provide user control;
- audit preserves traceability.

This is enough for the next Alpha slices and can later evolve into a richer
permission system if needed.

## Consequences

Positive:

- tools can be added incrementally;
- shell access has a safety model before implementation;
- policy and audit remain central;
- future integrations share one permission path;
- tests can validate denied, allowed and approval-required behavior.

Trade-offs:

- early implementation is slower than direct adapter calls;
- policy domain objects must be maintained;
- approvals add UX complexity;
- broad remembered permissions are deferred.

## Alternatives considered

### Direct tool-specific checks

Rejected for the general architecture. Direct checks inside each adapter would
duplicate policy logic and make audit inconsistent.

### Full policy DSL

Deferred. A DSL is unnecessary for Alpha and would add complexity before the
capability set is proven.

### OS-level sandbox first

Deferred as the primary abstraction. OS sandboxing may be useful for shell
execution, but it does not replace domain policy, approvals and audit.

### Allow shell first and backfill policy

Rejected. Shell is high-risk and should not precede policy.

## Testing requirements

Required tests before implementation:

```text
unit:
  capability taxonomy validates
  permission mode validates
  developer_local is the Alpha default
  locked_down requires approval for read-only shell
  automation denies direct autonomous memory writes
  unknown capability denied
  safe capability allowed
  cloud denied by default
  secret access denied
  shell write requires approval
  destructive action denied
  approval_required outcome includes scoped metadata

contract:
  PolicyPort capability decisions
  ConfigPolicyEngine implements capability policy

architecture:
  AgentRuntime does not import tool/shell/MCP/integration adapters
  ToolGateway adapters cannot bypass PolicyPort
  shell adapter cannot execute before classification

e2e:
  denied capability creates audit event
  approval_required returns user-visible pending state
```

Granted, denied, expired and cancelled one-shot approval execution flows are
implemented by PM-05 through API, CLI and ToolGateway validation. Remembered
approvals remain deferred.

Real external services, real shell side effects and real cloud model calls are
not required for CI.

## Rollout plan

1. Add capability domain objects and config schema.
2. Add permission mode config with `locked_down`, `developer_local` and
   `automation`.
3. Extend `PolicyPort` and `ConfigPolicyEngine`.
4. Add audit events for capability decisions.
5. Add architecture tests for future adapter boundaries.
6. Add ToolGateway ADR and fake gateway.
7. Implement safe built-in tools.
8. Add read-only shell only after shell-specific ADR.

## Deferred

- persistent remembered approvals;
- multi-user role-based access control;
- OS/container sandbox design;
- secret manager integration;
- cloud model enablement;
- WebSocket/control channel;
- external integration-specific policies;
- full policy DSL;
- enterprise-grade audit export.
