# 36 — Post-MVP Plan Review

## Status

Planning review for post-MVP Alpha and implementation status after PM-01
through PM-07b.

Date: 2026-05-30

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

The plans remain directionally consistent after the first Alpha implementation
wave:

- MVP remains a local-first core daemon;
- tools, read-only shell, read-only diagnostics, approvals and project docs RAG
  are implemented as post-MVP Alpha capabilities, not as original MVP scope;
- MCP, planner-executor, scheduler, voice, remembered approvals and external
  integrations remain future work;
- RAG remains separate from Memory;
- tool-capable loops use `ToolGatewayPort` and explicit `LoopStrategy`
  selection;
- advanced context work remains behind `ContextAssemblerPort`;
- cloud fallback remains disabled until a future ADR changes policy.

The original implementation-order risk was addressed for PM-01..PM-07b:
permissions, ToolGateway, loop strategy boundaries, approvals, shell/diagnostics
and project-docs retrieval were added through ports/adapters and tests.

The remaining risk is the next wave. MCP, voice, planner-executor, scheduler
and external integrations must not bypass the policy, approval, audit and
context boundaries now in place.

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
ToolGatewayPort
ContentRetrievalPort
future SchedulerPort
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

## 3. Remaining weak points and gaps

### Durable workflow execution is still missing

Request execution is still in-process. Approval records and
`waiting_approval` request status are durable, but the loop itself is not
checkpointed as a resumable workflow.

Required follow-up:

```text
ADR-037 Scheduler and durable background workflows
workflow checkpoint/resume slice before planner-executor or long automations
```

### Tool observation storage is still narrow

Tool observations are not conversation messages by default, which is correct.
The current implementation keeps observations suitable for bounded context, but
large artifacts still need a dedicated artifact store.

Required follow-up:

```text
ADR-044 Artifact storage
large-output artifact references before broad MCP/file/integration tools
```

### Shell remains read-only Alpha scope

PM-06a/PM-06b implemented read-only project shell and system diagnostics with
classification, allowlists, redaction, output caps and timeouts. Write, network
and destructive shell actions remain intentionally denied or deferred.

Required follow-up:

```text
separate write-shell ADR/slice with stronger sandboxing and approval rules
```

### Project docs RAG is narrow by design

PM-07a/PM-07b implemented project documentation ingestion, retrieval and
ContextAssembler integration. General file RAG, source-code indexing, PDFs,
web/email/Telegram ingestion and MCP resource indexing remain out of scope.

Required follow-up:

```text
source-specific RAG slices with explicit permissions and citation lifecycle
```

### Scheduler choice should remain deferred

NATS/JetStream is still only a possible future option. It should not be added
until DB-backed scheduling is proven insufficient.

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

## 5. Implemented Alpha slices

### Slice PM-01 — Capability and policy extension

Implemented with tests for:

- capability enum/config validates;
- permission mode config validates;
- `developer_local` is the Alpha default;
- `locked_down` requires approval for read-only shell;
- `automation` denies direct autonomous memory writes;
- policy denies unknown capabilities;
- policy allows configured safe local capabilities;
- policy requires approval for risky classes;
- denied decisions emit audit events;
- `tools_enabled` acts as a real kill switch for tool capabilities.

Implemented:

- capability domain objects;
- policy request/decision extensions;
- ConfigPolicyEngine rules.

### Slice PM-02 — ToolGatewayPort and fake gateway

Implemented with tests for:

- tool spec validates;
- fake tool executes through gateway;
- gateway calls policy before execution;
- tool lifecycle events are emitted;
- observation output is bounded.

Implemented:

- `ToolGatewayPort`;
- fake adapter;
- safe built-in tools.

### Slice PM-03 — LoopStrategy abstraction

Implemented with tests for:

- strategy registry selects `memory_augmented_answer` by default;
- unknown strategy is rejected;
- `memory_augmented_answer` keeps `max_tool_calls=0`;
- existing MVP event chain remains compatible;
- fake tool-capable strategy requires `ToolGatewayPort`.

Implemented:

- loop strategy domain objects;
- strategy registry;
- current deterministic workflow extracted as `MemoryAugmentedAnswerLoop`;
- loop-level events without changing user-visible MVP behavior.

### Slice PM-04 — Safe-tool loop v1

Implemented with tests for:

- fake model proposes safe tool call;
- fake/safe tool returns observation;
- final answer includes summarized result;
- max tool calls and max steps stop execution;
- malformed tool request fails safely.

Implemented:

- `tool_react_loop` or equivalent first tool-capable strategy;
- action parsing;
- tool observation context;
- only fake and safe tools, no shell/MCP/external integrations.

### Slice PM-05 — Approval model and CLI/API flow

Implemented with tests for:

- CLI displays approval request;
- approval-required action does not execute before grant;
- denied approval prevents execution;
- granted approval allows retry execution with matching approval id;
- expired approval prevents execution;
- interrupted approval cancels tool call.
- approval pauses expose durable `waiting_approval` request status.

Implemented:

- approval domain/API;
- CLI approval flow;
- ToolGateway approval validation and retry flow;
- redacted approval lifecycle events.

### Slice PM-06a — Project read-only shell tool

Implemented with tests for:

- read-only commands allowed only in allowlisted roots;
- write/destructive/network commands denied;
- timeout and output caps work;
- env values are redacted.

Implemented:

- shell adapter;
- command classifier;
- bounded subprocess execution.

### Slice PM-06b — Read-only system diagnostics tools

Implemented with tests for:

- process/resource/hardware diagnostics are allowed by command family;
- network diagnostics are bounded and redacted;
- temperature sensor snapshots are allowed without privilege escalation;
- missing temperature sensor backends are non-fatal;
- interactive diagnostics are denied;
- `du` is restricted to allowlisted workspace roots;
- platform-specific diagnostics are classified deterministically.

Implemented:

- diagnostics command classifier;
- platform-specific allowlists;
- sensor diagnostics adapter;
- bounded subprocess execution through the same shell adapter boundary.

### Slice PM-07a — Project Docs ingestion and citation index

Implemented with tests for:

- docs source registry;
- markdown chunking;
- secret-like docs paths are not ingested;
- source changes mark old chunks stale or refresh them.

Implemented:

- storage tables for content sources/chunks;
- markdown ingestion;
- citation index.

### Slice PM-07b — Project Docs retrieval and ContextAssembler integration

Implemented with tests for:

- retrieval returns citations;
- retrieval returns `ContentHit`, not `MemoryHit`;
- stale/deleted chunks are excluded;
- ContextAssembler includes content hits separately from memories;
- ContextManifest records content hit refs.
- retrieval failures emit `content.retrieval.failed` without storing raw query
  text.

Implemented:

- `ContentRetrievalPort`;
- content embeddings;
- docs retrieval adapter.

### Operational hardening added after PM-07b

Implemented:

- server-side `loop_strategy` and `model_profile` authorization;
- event payload sanitization before storage;
- storage CHECK constraints and no-secret constraints;
- append-only protection for events;
- inference readiness in `/v1/health`;
- `make run-ollama`, `make local-smoke` and `make content-ingest`;
- CLI/API content ingest, reindex, list and status operations;
- same-process SSE live buffers cleaned up after terminal drain, with durable
  replay from EventLog for completed requests.

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

Status: the safe-tool loop, project read-only shell, system diagnostics and
project docs RAG path are implemented. The next tool decision should focus on
MCP or a specific external integration only after artifact storage and workflow
durability are clarified.

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

Status: the first narrow project docs RAG path is implemented after
ToolGateway and LoopStrategy. Broader RAG remains source-specific future work.

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

PM-01 through PM-07b are implemented and the original architecture sequencing
concerns are no longer blockers for the first Alpha wave.

Recommended next direction:

```text
stabilize the implemented Alpha surface
then choose between durable workflow execution, artifact storage, MCP or voice
```

Do not start planner-executor, broad MCP, voice, Telegram/Spotify or cloud
reasoning before the durable workflow, artifact, permission and audit questions
for that specific feature are documented and tested.
