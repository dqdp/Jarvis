# 36 — Post-MVP Plan Review

## Status

Planning review for post-MVP Alpha.

Date: 2026-05-29

Reviewed documents:

```text
docs/13_phase_2_extension_points.md
docs/15_post_mvp_context_management.md
docs/20_post_mvp_rag_content_retrieval.md
docs/24_post_mvp_agent_loop_followups.md
docs/32_known_limitations.md
docs/34_post_mvp_roadmap.md
docs/35_post_mvp_adr_backlog.md
docs/37_post_mvp_tdd_slices_plan.md
docs/adr/ADR-029_capability_and_permission_model.md
docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md
docs/adr/ADR-031_agent_loop_strategy_architecture.md
```

## 1. Review verdict

The plans are directionally consistent:

- MVP remains a local-first core daemon;
- tools, MCP, RAG, planner-executor, scheduler, voice and integrations are
  post-MVP;
- RAG remains separate from Memory;
- tool-capable/ReAct loops wait for both `ToolGatewayPort` and an explicit
  `LoopStrategy` boundary;
- advanced context work remains behind `ContextAssemblerPort`;
- cloud fallback remains disabled until a future ADR changes policy.

The main risk is not a contradiction in the documents. The main risk is
implementation order. If we add shell, MCP, voice or RAG before permissions,
ToolGateway, loop strategy boundaries and audit are ready, the architecture
will likely degrade into direct adapters called from runtime code.

## 2. Strong points

### Clear MVP boundary

The docs consistently state that the MVP is complete and that advanced
capabilities are post-MVP. This reduces scope creep.

### Good port structure

The existing architecture already has the right extension points:

```text
ContextAssemblerPort
ModelRouterPort
MemoryReadPort
MemoryWritePort
PolicyPort
future ToolGatewayPort
future SchedulerPort
future ContentRetrievalPort
```

### Correct RAG boundary

The plan separates stable interpreted memory from source-content retrieval.
This is important. Document chunks must not become memory records.

### Correct loop boundary direction

The plan now treats ReAct as a future loop strategy, not as a tool. This
prevents uncontrolled tool loops and keeps the MVP `memory_augmented_answer`
workflow stable.

### Testing discipline is already strong

The project already has unit, contract, integration, golden, architecture and
e2e test layers. Post-MVP features can extend this discipline instead of
inventing a new process.

## 3. Weak points and gaps

### Permissions implementation is still pending

ADR-029 now accepts the capability taxonomy, risk classes and permission modes.
The implementation still needs PM-01 before tools or integrations start.

Required fix:

```text
Slice PM-01 Capability and policy extension
```

### Loop strategy boundary needs implementation

ToolGateway defines how tools execute, but not how an agent decides to use
them. The base agent loop must become an explicit strategy before adding a
tool-capable loop.

Required fix:

```text
Slice PM-03 LoopStrategy abstraction
```

### Approval baseline needs implementation

Risky tools need approval. The Alpha baseline is now one-shot approvals with
redacted HTTP grant/deny endpoints, CLI prompts and SSE/runtime events.
WebSocket and remembered approvals are deferred.

Required fix:

```text
Slice PM-05 Approval model and CLI/API flow
```

### Tool observation storage is not fully designed

The docs correctly say observations are not conversation messages by default.
They do not yet define where large outputs live or how they enter context.

Required fix:

```text
ADR-030 ToolGateway boundary
ADR-044 Artifact storage
```

### Shell sandbox needs an explicit threat model

Shell is the highest-risk early feature. We need command classification,
working-directory policy, output caps, timeouts and approval rules before any
runtime shell execution.

Required implementation:

```text
PM-06a Project read-only shell tool
PM-06b Read-only system diagnostics tools
```

### RAG ingestion lifecycle is not yet concrete

The RAG boundary is good, but implementation still needs source registry,
chunking, citation invalidation and refresh behavior.

Required implementation:

```text
PM-07a Project Docs ingestion and citation index
PM-07b Project Docs retrieval and ContextAssembler integration
```

### Scheduler choice should remain deferred

NATS/JetStream is mentioned as a possible future option. It should not be added
until DB-backed scheduling is proven insufficient.

Required fix:

```text
ADR-037 Scheduler and durable background workflows
```

## 4. Contradiction check

No blocking contradictions found.

Potential confusion points to watch:

- `MCP gateway` appears both as a tool adapter and future integration. Treat it
  as a ToolGateway adapter, not a runtime dependency.
- `Telegram` appears as both integration and interaction channel. Treat the
  first version as an integration adapter; a full client/channel model can come
  later.
- `cloud_reasoning` exists in config but remains disabled. Do not interpret its
  presence as permission to add fallback.
- `sleep/reflection` is accepted as a bounded workflow concept, but direct
  autonomous memory mutation remains disallowed until a future policy ADR.
- `voice` may need realtime transport, but push-to-talk can start without
  committing to WebSocket or cloud realtime models.

## 5. Recommended next implementation slices

### Slice PM-01 — Capability and policy extension

Tests first:

- capability enum/config validates;
- permission mode config validates;
- `developer_local` is the Alpha default;
- `locked_down` requires approval for read-only shell;
- `automation` denies direct autonomous memory writes;
- policy denies unknown capabilities;
- policy allows configured safe local capabilities;
- policy requires approval for risky classes;
- denied decisions emit audit events.

Implementation:

- capability domain objects;
- policy request/decision extensions;
- ConfigPolicyEngine rules.

### Slice PM-02 — ToolGatewayPort and fake gateway

Tests first:

- tool spec validates;
- fake tool executes through gateway;
- gateway calls policy before execution;
- tool lifecycle events are emitted;
- observation output is bounded.

Implementation:

- `ToolGatewayPort`;
- fake adapter;
- safe built-in tools.

### Slice PM-03 — LoopStrategy abstraction

Tests first:

- strategy registry selects `memory_augmented_answer` by default;
- unknown strategy is rejected;
- `memory_augmented_answer` keeps `max_tool_calls=0`;
- existing MVP event chain remains compatible;
- fake tool-capable strategy requires `ToolGatewayPort`.

Implementation:

- loop strategy domain objects;
- strategy registry;
- current deterministic workflow extracted as `MemoryAugmentedAnswerLoop`;
- loop-level events without changing user-visible MVP behavior.

### Slice PM-04 — Safe-tool loop v1

Tests first:

- fake model proposes safe tool call;
- fake/safe tool returns observation;
- final answer includes summarized result;
- max tool calls and max steps stop execution;
- malformed tool request fails safely.

Implementation:

- `tool_react_loop` or equivalent first tool-capable strategy;
- action parsing;
- tool observation context;
- only fake and safe tools, no shell/MCP/external integrations.

### Slice PM-05 — Approval model and CLI/API flow

Tests first:

- CLI displays approval request;
- approval-required action does not execute before grant;
- denied approval prevents execution;
- granted approval allows retry execution with matching approval id;
- expired approval prevents execution;
- interrupted approval cancels tool call.

Implementation:

- approval domain/API;
- CLI approval flow;
- ToolGateway approval validation and retry flow;
- redacted approval lifecycle events.

### Slice PM-06a — Project read-only shell tool

Tests first:

- read-only commands allowed only in allowlisted roots;
- write/destructive/network commands denied;
- timeout and output caps work;
- env values are redacted.

Implementation:

- shell adapter;
- command classifier;
- bounded subprocess execution.

### Slice PM-06b — Read-only system diagnostics tools

Tests first:

- process/resource/hardware diagnostics are allowed by command family;
- network diagnostics are bounded and redacted;
- temperature sensor snapshots are allowed without privilege escalation;
- missing temperature sensor backends are non-fatal;
- interactive diagnostics are denied;
- `du` is restricted to allowlisted workspace roots;
- platform-specific diagnostics are classified deterministically.

Implementation:

- diagnostics command classifier;
- platform-specific allowlists;
- sensor diagnostics adapter;
- bounded subprocess execution through the same shell adapter boundary.

### Slice PM-07a — Project Docs ingestion and citation index

Tests first:

- docs source registry;
- markdown chunking;
- secret-like docs paths are not ingested;
- source changes mark old chunks stale or refresh them.

Implementation:

- storage tables for content sources/chunks;
- markdown ingestion;
- citation index.

### Slice PM-07b — Project Docs retrieval and ContextAssembler integration

Tests first:

- retrieval returns citations;
- retrieval returns `ContentHit`, not `MemoryHit`;
- stale/deleted chunks are excluded;
- ContextAssembler includes content hits separately from memories;
- ContextManifest records content hit refs.

Implementation:

- `ContentRetrievalPort`;
- content embeddings;
- docs retrieval adapter.

## 6. Main questions for discussion

### Question 1 — What is the first valuable tool?

Options:

```text
safe built-in tools
read-only shell
project docs RAG
MCP gateway
Telegram
```

Recommendation:

```text
safe built-in tools -> LoopStrategy abstraction -> safe-tool loop
-> project read-only shell -> system diagnostics
```

Reason: it validates ToolGateway, policy, audit and the new loop boundary before
external integrations.

### Question 2 — How strict should approvals be in Alpha?

Agreed baseline:

```text
use permission modes
implement locked_down, developer_local and automation first
default Alpha mode: developer_local
defer remembered approvals
```

`developer_local` defaults:

```text
read-only safe tools: no approval
project docs RAG: no approval
context inspect: no approval for manifests/refs only
read-only shell in allowlisted cwd: no approval
write shell: approval required
network tools: approval required
external side effects: approval required
cloud/destructive/secrets: deny by default
```

`locked_down` is available when read-only shell should require approval too.
`automation` is available for future scheduled/background workflows, with direct
autonomous memory writes denied.

### Question 3 — Do we start RAG before or after tool loop?

Recommendation:

```text
after ToolGateway foundation and LoopStrategy abstraction, before large external integrations
```

Reason: project docs RAG is high-value and low-risk, but it still benefits from
ContextAssembler V2 and explicit content permissions.

### Question 4 — Do we need WebSocket now?

Recommendation:

```text
not yet
```

Use SSE for runtime events and HTTP for approvals first. Add WebSocket only when
approval latency, bidirectional task control or voice/realtime requirements make
it necessary.

### Question 5 — What is the first scheduler implementation?

Recommendation:

```text
DB-backed scheduler first
```

Reason: fewer moving parts. NATS/JetStream should wait until we have enough
background workflow pressure to justify it.

### Question 6 — How much shell access should the assistant get?

Recommendation:

```text
split PM-06 into project read-only shell and read-only system diagnostics
```

Project shell starts read-only, allowlisted cwd, no network, bounded output.
System diagnostics allow read-only process/resource/hardware/network probes with
redaction and output caps. Write commands and network clients remain out of
scope for PM-06.

### Question 7 — How do we treat Telegram?

Recommendation:

```text
first as an integration adapter, later as a full client/channel
```

This avoids mixing notification/tool semantics with the main conversation
transport too early.

### Question 8 — When can cloud models be enabled?

Recommendation:

```text
only after ADR-041
```

Cloud fallback should remain disabled. Explicit cloud calls need sensitivity
rules, redaction, user approval and audit before implementation.

## 7. Review conclusion

The roadmap is ready for discussion and can drive the next implementation goal.

Recommended immediate next step:

```text
Implement Slice PM-01.
```

Do not start with shell, MCP, RAG, voice or Telegram before the capability and
permission foundation is in place.
