# 35 — Post-MVP ADR Backlog

## Status

Proposed backlog for post-MVP architecture decisions.

Date: 2026-05-29

This document lists the ADRs that should be written before the corresponding
implementation work starts. It is intentionally a backlog, not an accepted set
of decisions.

## 1. How to use this backlog

Create a real ADR when a topic moves from planning to implementation.

Each ADR should include:

```text
Context
Decision
Consequences
Alternatives considered
Boundary rules
Testing requirements
Migration or rollout plan
```

If a change only affects implementation order, update the slice plan. If it
changes architecture, write or update an ADR first.

## 2. High-priority ADRs

### ADR-029 — Capability and permission model

Status:

```text
Promoted to docs/adr/ADR-029_capability_and_permission_model.md.
Current ADR status: Accepted.
```

Needed before:

```text
tools
shell
MCP
external integrations
voice
cloud enablement
```

Decision scope:

- capability taxonomy;
- permission modes;
- risk classes;
- default deny rules;
- approval requirement model;
- interaction with existing `PolicyPort`;
- audit events for policy decisions.

Resolved baseline:

- Alpha uses explicit permission modes and local configuration first;
- approval grants are explicit, expiring and scoped to one action/request/task
  step;
- broad remembered permissions are deferred.

Testing:

- unit tests for allow/deny decisions;
- contract tests for `PolicyPort`;
- architecture tests that tool adapters cannot bypass policy.

### ADR-030 — ToolGateway boundary and tool invocation audit

Status:

```text
Promoted to docs/adr/ADR-030_toolgateway_boundary_and_tool_invocation_audit.md.
Current ADR status: Accepted.
```

Needed before:

```text
safe tools
CLI tools
MCP
shell
integrations
tool-capable loops
```

Decision scope:

- `ToolGatewayPort` contract;
- `ToolSpec` and tool input/output schema;
- tool lifecycle events;
- observation storage rules;
- max output/truncation policy;
- fake tool adapter requirements.

Resolved baseline:

- PM-02 starts with a small internal `ToolSpec` contract;
- PM-02 does not add artifact storage for large outputs;
- only bounded, redacted observations may enter context automatically.

Testing:

- contract tests for gateway behavior;
- fake tool e2e tests;
- audit-event tests for every tool state transition.

### ADR-031 — Agent loop strategy architecture

Status:

```text
Promoted to docs/adr/ADR-031_agent_loop_strategy_architecture.md.
Current ADR status: Accepted.
```

Needed before:

```text
tool-capable loops
bounded ReAct
planner-executor
sleep/reflection workflows
maintenance workflows
```

Decision scope:

- loop strategy boundary;
- strategy registry;
- preserving `memory_augmented_answer`;
- tool-capable loop dependencies;
- loop and step events;
- per-strategy budgets and stopping conditions.

Resolved baseline:

- PM-03 extracts the current loop behind `memory_augmented_answer` without
  behavior changes;
- strategy selection starts as config/registry driven;
- PM-03 adds loop-level events, while step-level events wait for PM-04.

Testing:

- current MVP loop still passes unchanged;
- strategy registry selects `memory_augmented_answer` by default;
- tool-capable strategy requires ToolGateway;
- architecture tests prevent loop strategies from importing concrete adapters.

### ADR-032 — Approval extensions and control channel

Needed before:

```text
remembered approvals
approval rules UI
multi-client approval routing
external side effects
planner-executor
long-running tasks
```

Alpha baseline:

```text
PM-05 implements one-shot approvals.
Grant/deny happens through HTTP endpoints and CLI prompts.
Runtime events can use existing SSE/event streams.
WebSocket is deferred.
Remembered approvals are deferred.
```

Decision scope:

- remembered approval scope and expiry rules;
- cross-client approval routing;
- approval rules UI;
- WebSocket/control-channel upgrade triggers;
- concurrency semantics beyond a single local user.

Open questions:

- Which approval grants may be remembered after Alpha?
- Which workflows need push-style control instead of HTTP plus SSE?
- How should multi-client approval conflicts be represented?

Testing:

- remembered-approval scope tests;
- cross-client approval routing tests;
- concurrency tests for duplicate approvals;
- WebSocket/control-channel tests if that channel is introduced.

### ADR-033 — Shell sandbox and local command policy

Status:

```text
Promoted to docs/adr/ADR-033_shell_sandbox_and_local_command_policy.md.
Current ADR status: Accepted.
```

Needed before:

```text
CLI/shell tool execution
agent-driven repo inspection
local automation
```

Decision scope:

- allowed working directories;
- allowed command families;
- read/write/network/destructive classification;
- environment redaction;
- timeout and output caps;
- audit format;
- approval requirements.

Resolved baseline:

- PM-06 is split into `PM-06a Project read-only shell tool` and `PM-06b
  Read-only system diagnostics tools`;
- execution is argv-only through `ToolGatewayPort`, with no shell string;
- initial project commands are read-only inspection commands;
- initial diagnostics commands are read-only process/resource/hardware/network
  probes plus temperature sensor snapshots with bounded and redacted output;
- sensor diagnostics never request sudo and never mutate fan or power settings;
- cwd and file path access are restricted to allowlisted roots unless the
  diagnostics command family explicitly permits a system read.

Testing:

- unit tests for command classification;
- contract tests for shell adapter;
- fake shell adapter for CI;
- regression tests for denied destructive commands.

### ADR-034 — Content Retrieval subsystem and Project Docs RAG

Status:

```text
Promoted to docs/adr/ADR-034_content_retrieval_subsystem_and_project_docs_rag.md.
Current ADR status: Accepted.
```

Needed before:

```text
RAG
document ingestion
citations
content retrieval context sections
```

Decision scope:

- `ContentRetrievalPort`;
- source registry;
- chunk model;
- citation model;
- ingestion lifecycle;
- refresh/re-index rules;
- permission and sensitivity handling;
- ContextAssembler integration.

Resolved baseline:

- initial corpus is `README.md`, `docs/*.md` and `docs/adr/*.md`;
- document chunks are stored only in Content Retrieval tables, not Memory;
- initial storage may use PostgreSQL arrays behind `ContentRetrievalPort`;
- pgvector is an adapter optimization, not required for PM-07;
- markdown chunking is deterministic by heading, with bounded oversized splits;
- citations use source path plus line range when available;
- changed/deleted sources mark old chunks stale or deleted.

Testing:

- ingestion integration tests;
- PM-07a source/chunk/citation index tests;
- retrieval contract tests;
- PM-07b retrieval/context integration tests;
- golden context tests with `ContentHit`;
- citation formatting tests.

### ADR-035 — Bounded tool loop strategy

Needed before:

```text
ReAct-style tool use
model-proposed actions
multi-step tool calls
```

Decision scope:

- loop name and contract;
- action parser;
- budget model;
- allowed capabilities;
- failure and retry semantics;
- tool observation handling;
- event chain.

Open questions:

- Should model tool calls use structured output, text protocol or provider tool
  calling?
- Should the first implementation allow only one tool call per step?
- How should repeated bad tool proposals be handled?

Testing:

- fake model proposes tool action;
- fake tool returns observation;
- budget exhaustion stops loop;
- malformed action fails safely.

## 3. Medium-priority ADRs

### ADR-036 — Planner-executor task model

Needed before long-running multi-step tasks.

Decision scope:

- task tables;
- plan and plan-step schema;
- agent-step events;
- pause/resume/cancel;
- correlation model;
- user clarification flow.

### ADR-037 — Scheduler and durable background workflows

Needed before reminders, proactive checks and sleep/reflection execution.

Decision scope:

- DB scheduler vs NATS/JetStream;
- worker process model;
- retry policy;
- idempotency;
- restart semantics.

### ADR-038 — Sleep/reflection workflow and memory candidates

Needed before autonomous consolidation.

Decision scope:

- event selection window;
- sleep report format;
- memory candidate generation;
- candidate approval;
- archive/supersede proposals;
- direct memory mutation restrictions.

### ADR-039 — External integration adapter model

Needed before Telegram, Spotify, GitHub, calendar, mail and similar adapters.

Decision scope:

- integration secrets;
- adapter lifecycle;
- fake servers;
- rate limits;
- side-effect policy;
- audit model.

### ADR-040 — MCP gateway

Needed before MCP server/tool support.

Decision scope:

- MCP server registry;
- capability mapping;
- tool schema import;
- resource access policy;
- process isolation;
- audit and approval integration.

## 4. Lower-priority ADRs

### ADR-041 — Voice gateway

Needed before voice implementation.

Decision scope:

- push-to-talk vs wake word sequence;
- VAD/STT/TTS providers;
- audio storage policy;
- transcript sensitivity;
- interruption/barge-in;
- realtime model policy.

### ADR-042 — Cloud model enablement and fallback policy

Needed before enabling any cloud model or fallback.

Decision scope:

- explicit enablement;
- sensitivity redaction;
- user approval;
- model profile selection;
- audit;
- failure and fallback semantics.

Default remains:

```text
cloud disabled
no automatic fallback
secret never sent to cloud
```

### ADR-043 — WebSocket/control channel

Needed if SSE plus HTTP is insufficient for approvals, live task control or
voice/realtime sessions.

Decision scope:

- transport contract;
- authentication/local access;
- backpressure;
- reconnect semantics;
- relation to existing SSE events.

### ADR-044 — Artifact storage

Needed when tool/RAG outputs become too large for event payloads.

Decision scope:

- artifact references;
- retention;
- sensitivity;
- filesystem vs database;
- relation to ContextManifest and tool observations.

## 5. ADR dependency order

Recommended order:

```text
ADR-029 Capability and permission model (accepted)
ADR-030 ToolGateway boundary (accepted)
ADR-031 Agent loop strategy architecture (accepted)
ADR-032 Approval extensions/control channel (deferred after PM-05 baseline)
ADR-033 Shell sandbox (accepted)
ADR-034 Content Retrieval / Project Docs RAG
ADR-035 Bounded tool loop
ADR-037 Scheduler
ADR-038 Sleep/reflection
ADR-036 Planner-executor
ADR-039 External integrations
ADR-040 MCP gateway
ADR-041 Voice gateway
ADR-042 Cloud enablement
ADR-043 WebSocket/control channel
ADR-044 Artifact storage
```

Rationale:

- permissions must precede side effects;
- ToolGateway must precede tool-capable loop strategies;
- loop-strategy architecture must precede ReAct/planner loops;
- shell needs approval before write capability;
- RAG can proceed after context V2, but must remain separate from memory;
- voice should wait until streaming, interrupt, policy and context semantics are
  stronger.

## 6. ADRs not needed yet

Do not write detailed ADRs yet for:

```text
fine tuning
multi-user SaaS permissions
distributed deployment
mobile app
consumer web UI
training pipeline
```

They are outside the current product horizon.
