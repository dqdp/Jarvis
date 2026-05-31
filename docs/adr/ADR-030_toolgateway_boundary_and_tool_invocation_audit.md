# ADR-030 — ToolGateway Boundary and Tool Invocation Audit

## Status

Accepted.

## Context

ADR-029 accepts a capability and permission model for post-MVP Alpha. The next
step is a controlled tool execution boundary.

Future Jarvis capabilities will need tools:

- safe built-in tools such as datetime, calculator and daemon status;
- project-local shell inspection;
- filesystem operations;
- MCP tools;
- search;
- Telegram, Spotify, GitHub and other integrations;
- future planner-executor and bounded tool loops.

Without a dedicated boundary, tools would likely be called directly from
`AgentRuntime`, CLI handlers, model-router code or ad hoc integration modules.
That would break the ports/adapters architecture and make policy, audit,
output limits and approvals inconsistent.

Tool execution must therefore go through a single domain boundary:

```text
AgentRuntime / loop strategy / API / CLI
  -> ToolGatewayPort
      -> PolicyPort
      -> concrete tool adapter
      -> EventLogPort
```

This ADR defines the first ToolGateway design. It does not introduce a
tool-capable agent loop, planner-executor, shell sandbox or MCP gateway
implementation by itself. ReAct is a future loop strategy, not a tool.

## Decision

Introduce `ToolGatewayPort` as the only tool execution boundary.

`ToolGatewayPort` owns:

- tool registry lookup;
- tool spec exposure;
- request validation;
- capability/policy checks;
- approval state checks when required;
- bounded execution;
- timeout and output limits;
- tool invocation audit;
- normalized tool observations.

Concrete adapters own only the actual tool implementation. They must not bypass
policy, approval or audit.

## Non-goals

This ADR does not implement:

```text
tool-capable/ReAct loop strategy
planner-executor
MCP gateway
shell sandbox
external integrations
approval transport
artifact storage backend
provider-native tool calling
```

Those require separate ADRs or implementation slices.

## ToolGatewayPort contract

Potential shape:

```python
class ToolGatewayPort(Protocol):
    async def list_tools(self, query: ToolListQuery) -> list[ToolSpec]: ...

    async def get_tool(self, tool_name: str) -> ToolSpec | None: ...

    async def invoke(self, request: ToolCallRequest) -> ToolObservation: ...
```

The exact Python names may change during implementation, but the contract must
keep these properties:

- callers use domain schemas, not adapter-specific types;
- `invoke` always evaluates policy before execution;
- denied calls return a `denied` `ToolObservation` and emit audit;
- approval-required calls return an `approval_required` `ToolObservation` until
  approval is granted;
- approval-required calls do not execute until approval is granted;
- tool outputs are bounded and redacted before returning to callers.

## ToolSpec

`ToolSpec` describes what a tool can do. It is not the tool implementation.

Required fields:

```text
name
display_name
description
capability
risk_classes
input_schema
output_schema optional
default_timeout_seconds
max_output_bytes
sensitivity_ceiling
requires_approval_by_default
adapter_name
enabled
metadata
```

Rules:

- `name` is stable and unique;
- `capability` uses ADR-029 capability identifiers;
- `risk_classes` use ADR-029 risk classes;
- disabled tools are visible only to diagnostics unless explicitly requested;
- tool specs must not contain secrets.

### Schema format

Alpha should use simple internal schemas backed by typed domain/Pydantic models.

JSON Schema may be derived later for:

- MCP;
- OpenAI/provider-native tool calling;
- external tool registries;
- UI forms.

Do not start with a full custom schema DSL.

## ToolCallRequest

Required fields:

```text
tool_name
arguments
request_id optional
conversation_id optional
correlation_id optional
step_id optional
user_id
project_namespace optional
working_directory optional
sensitivity
permission_mode
approval_id optional
idempotency_key optional
timeout_seconds optional
max_output_bytes optional
metadata
```

Rules:

- `arguments` are validated before adapter execution;
- caller-provided timeout/output limits may only tighten tool defaults;
- `permission_mode` is passed to PolicyPort;
- `approval_id` is required only when policy has returned
  `approval_required`;
- `idempotency_key` is recommended for future side-effecting tools.

## ToolObservation

`ToolObservation` is the normalized result returned by ToolGateway.

Required fields:

```text
tool_call_id
tool_name
status
content
content_type
structured_content optional
structured_schema optional
structured_schema_version optional
parse_status optional
parse_warnings optional
sensitivity
truncated
output_bytes
started_at
completed_at
duration_ms
error optional
artifact_refs
metadata
```

Initial statuses:

```text
completed
denied
approval_required
failed
timeout
cancelled
```

Rules:

- observations are not conversation messages by default;
- observations may be included in context only through ContextAssembler policy;
- large outputs should be truncated and later moved to artifact refs;
- raw secrets must not appear in observations;
- failed and denied calls are observable and auditable.

Typed observations:

- `content` / `content_type` remain the bounded human/debug representation;
- `structured_content` is the provider-neutral machine-readable payload used by
  direct answer formatters and future UIs;
- `structured_schema` and `structured_schema_version` identify the payload
  contract;
- `parse_status` describes whether native output was normalized into that
  contract;
- `parse_warnings` carries stable non-secret labels for lossy or partial
  normalization;
- tool adapters own normalization from native command/API output into typed
  payloads;
- loop strategies must not grow command-specific stdout parsers for each tool;
- if typed payload generation fails, the observation may still carry bounded raw
  content for audit/debug and normal model/ReAct fallback, but direct answers
  must not invent parsed state from unrecognized raw output.

Typed-capable tools use these parse statuses:

```text
parsed
partial
unparsed
not_applicable
```

The typed contract must be propagated through the full tool pipeline:

```text
ToolInvocationResult
  -> ToolObservation
  -> ToolObservationRef
  -> context/direct formatter/event payload
```

Direct answer builders may produce deterministic answers only from
`structured_content` with `parse_status=parsed` or from the available subset of
`structured_content` with `parse_status=partial` and an explicit cautious
warning. `parse_status=unparsed` must route to bounded ReAct/model analysis when
allowed by policy and model budget, or return a clear unavailable/unparsed
result. Raw stdout/stderr is never the primary direct-answer contract.

Initial typed diagnostics payloads should cover:

```text
system.os_version v1
system.battery_charge v1
system.disk_free v1
system.vpn_status v1
system.process_name_search v1
system.cpu_overview v1
system.sensor_snapshot v1
```

Schema names are provider-neutral, for example
`structured_schema=system.os_version` and `structured_schema_version=1`.

## Tool lifecycle events

ToolGateway emits durable events.

Initial events:

```text
tool.call.requested
tool.call.approved
tool.call.denied
tool.call.started
tool.call.completed
tool.call.failed
tool.call.timeout
tool.call.cancelled
tool.observation.recorded
```

Required event linkage:

```text
request_id        user-turn request when applicable
correlation_id    long-running workflow when applicable
causation_id      prior event that caused this event
tool_call_id      stable invocation id
policy_decision_id
approval_id optional
step_id optional
```

Payload rules:

- store tool name, capability, risk classes and stable reason;
- store redacted arguments;
- store output metadata and short redacted preview when safe;
- store artifact refs for large outputs when artifact storage exists;
- never store raw secrets.

## Policy and approval flow

Tool invocation flow:

```text
validate tool exists and enabled
validate arguments
classify capability and risk
call PolicyPort
emit policy decision event
if deny:
  emit tool.call.denied
  return denied observation
if approval_required and no granted approval:
  emit approval.required
  return approval_required observation
if allow or granted approval:
  emit tool.call.started
  execute adapter with limits
  emit completed/failed/timeout/cancelled
  emit tool.observation.recorded
  return observation
```

ToolGateway must call PolicyPort itself. Callers must not pre-approve tools by
skipping gateway policy.

Approval transport is implemented by PM-05 with one-shot approval records,
redacted HTTP grant/deny endpoints, CLI prompts and SSE/runtime events.
Remembered approvals and WebSocket transport remain deferred.

## Initial tool set

The first implementation should include only fake and safe tools.

Fake tools:

```text
fake.echo
fake.fail
fake.timeout
```

Safe built-in tools:

```text
datetime.now
calculator.evaluate
daemon.status
```

Optional after the first pass:

```text
memory.lookup
conversation.lookup
context.inspect
```

Do not include shell, MCP, Telegram, Spotify, GitHub, network search or
filesystem write tools in the first ToolGateway slice.

Do not include ReAct in the first ToolGateway slice. ReAct is not a tool; it is
a future loop strategy that will use ToolGateway after the loop-strategy
boundary exists.

## Output limits

Every tool has output limits.

Initial defaults:

```text
default_timeout_seconds: 10
max_output_bytes: 20000
max_observation_preview_bytes: 2000
```

Rules:

- timeout failure returns a `timeout` observation;
- output above limit is truncated;
- truncated output sets `truncated=true`;
- adapters must not stream unbounded data into event payloads;
- future artifact storage may preserve full output when policy allows it.

## Registry model

Alpha may use an in-process registry assembled at app startup.

Registry sources:

```text
static built-in tool registrations
config-enabled tools
future MCP-imported specs
future integration adapters
```

Rules:

- disabled tools cannot execute;
- duplicate tool names fail startup validation;
- each registered tool declares capability and risk classes;
- each adapter has contract tests.

## Boundary rules

AgentRuntime:

- may call `ToolGatewayPort`;
- must not import concrete tool adapters;
- must not call subprocess, MCP clients, external service clients or filesystem
  adapters directly.

Tool adapters:

- implement one narrow operation or adapter family;
- do not call PolicyPort directly unless ToolGateway delegates classification
  explicitly;
- do not write EventLog directly except through ToolGateway-managed hooks;
- return structured results, not arbitrary printed output.

ContextAssembler:

- may include selected tool observations;
- must not execute tools;
- must apply sensitivity and budget policy before including observations.

ModelRouter:

- must not execute tools;
- may later convert provider-native tool-call syntax into domain tool-call
  proposals, but execution still goes through ToolGateway.

CLI/API:

- may request tool execution through ToolGateway;
- must not call concrete adapters directly.

## Rationale

ToolGateway gives the system one place to enforce:

- permissions;
- approvals;
- audit;
- bounded execution;
- output normalization;
- future integration consistency.

Starting with fake and safe tools validates the boundary without taking on shell
or external-service risk too early.

## Consequences

Positive:

- tools become testable and replaceable;
- future loop strategies such as ReAct and planner-executor can use one
  execution path;
- security policy is centralized;
- audit is consistent across tool types;
- adapters remain small.

Trade-offs:

- extra domain types are needed before useful tools;
- simple tools require more ceremony than direct function calls;
- approval and artifact storage need follow-up ADRs;
- provider-native tool calling cannot bypass this layer.

## Alternatives considered

### Direct tool calls from AgentRuntime

Rejected. This would couple runtime to adapters and bypass policy/audit.

### Tool execution inside ModelRouter

Rejected. ModelRouter owns provider calls and provider-specific request
conversion, not side-effect execution.

### MCP-first gateway

Deferred. MCP is important, but starting with MCP would mix registry,
transport, process management and permission concerns before the core tool
boundary is proven.

### JSON Schema-first tool registry

Deferred. Simple internal typed schemas are enough for Alpha. JSON Schema can be
derived later for MCP/provider-native tool calling.

## Testing requirements

Required tests before implementation:

```text
unit:
  tool spec validates required fields
  duplicate tool names fail registry validation
  disabled tool cannot execute
  arguments validate before execution
  output truncation sets truncated=true

contract:
  ToolGatewayPort lists tools
  ToolGatewayPort gets tool by name
  ToolGatewayPort invokes fake success tool
  ToolGatewayPort returns denied observation when policy denies
  ToolGatewayPort returns approval_required without execution
  ToolGatewayPort records timeout observation

architecture:
  AgentRuntime does not import concrete tool adapters
  CLI/API do not import concrete tool adapters
  ModelRouter does not execute tools
  ContextAssembler does not execute tools

e2e:
  safe tool call emits policy and tool lifecycle events
  denied tool call emits denied event and no adapter execution
  approval-required call does not execute adapter
```

Real shell commands, real MCP servers and real external services are not
required for CI.

## Rollout status

1. Add tool domain schemas.
2. Add `ToolGatewayPort`.
3. Add fake tool adapter and registry.
4. Add ConfigPolicyEngine capability checks for tools from ADR-029.
5. Add event types and audit payloads.
6. Add safe built-in tools.
7. Add CLI/API diagnostic command to list tools.
8. Loop-strategy architecture is covered by ADR-031.
9. Approval transport is implemented by PM-05 and documented by ADR-032.
10. Shell sandbox policy is covered by ADR-033.
11. Bounded tool-loop implementation is covered by ADR-031 and the PM-03
    slices in `docs/37_post_mvp_tdd_slices_plan.md`.

## Deferred

- MCP server registry;
- provider-native tool calling;
- external integrations;
- artifact storage;
- streaming tool output;
- tool result caching;
- tool idempotency persistence for side effects;
- planner-executor integration.
