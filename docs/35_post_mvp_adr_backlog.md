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
- strategy registry selects `memory_augmented_answer` by default in the PM-03
  baseline;
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

### ADR-035 — Automatic loop strategy selection

Status:

```text
Promoted to docs/adr/ADR-035_automatic_loop_strategy_selection.md.
Current ADR status: Accepted.
```

Needed before:

```text
default CLI/API access to tools
default CLI/API access to Project Docs RAG
automatic routing between normal chat and tool-capable loops
future local model-backed intent classifier adapter
```

Decision scope:

- user-facing `auto`, `chat` and `tools` modes;
- backend `LoopStrategySelector`;
- `IntentClassifierPort`;
- domain objects for `LoopSelectionRequest`, `IntentClassification`,
  `CapabilityCandidate` and `LoopSelectionDecision`;
- confidence bands and fallback semantics;
- capability routing metadata for future tools such as code sandbox;
- fake classifier for CI;
- conservative deterministic runtime classifier as an initial adapter;
- optional local structured classifier adapter later;
- relation to policy and approvals;
- relation to RAG and ContextAssembler;
- CLI/API override semantics;
- redacted routing audit events.

Resolved baseline:

- `auto` is a routing mode, not a concrete loop;
- PM-08 must not make a deterministic-only selector the target architecture;
- PM-08 starts with `LoopStrategySelector` plus `IntentClassifierPort`;
- runtime may initially use a conservative deterministic classifier
  implementation, but the port must allow a local model-backed classifier later;
- ordinary chat and project-docs questions use `memory_augmented_answer`;
- live project/system inspection uses `tool_react_loop`;
- RAG is not a tool-loop trigger by itself;
- tools-disabled tool intent must not silently fallback to hallucinated chat;
- CLI does not own safety-critical routing.

Testing:

- unit tests for selector decisions and reason codes;
- contract tests for API default `auto` behavior;
- architecture tests preventing CLI/tool/storage adapter coupling;
- e2e fake-provider tests for ordinary chat, project-docs RAG and tool intent.
- implementation is split into PM-08a selector contract, PM-08b API lifecycle,
  PM-08c CLI mode controls and PM-08d CLI tool/RAG/approval readiness.

### Resolved by ADR-031/PM-04 — Bounded tool loop strategy

The original ADR-035 backlog item for a bounded tool loop is now covered by
`docs/adr/ADR-031_agent_loop_strategy_architecture.md` and implemented through
PM-04.

Open questions:

- Should the later ambiguous-intent classifier call the same structured local
  profile used by `tool_react_loop`, or a smaller dedicated classifier profile?
- Should explicit CLI `/mode tools` persist only in-session or in a local config
  file?
- Which future external integrations should be eligible for automatic routing?

Testing:

- keep bounded tool-loop tests under ADR-031/PM-04;
- add PM-08 auto-selection tests under ADR-035.

## 3. Next-priority ADRs

### ADR-042 — Voice gateway

Needed before voice implementation.

Current priority:

```text
write/promote this ADR after PM-08d CLI tool/RAG/approval readiness, before
PM-09 voice gateway foundation implementation.
```

Decision scope:

- push-to-talk before wake word;
- voice as a client/channel over the existing runtime, not a separate agent;
- `SpeechToTextPort`;
- `TextToSpeechPort`;
- fake STT/TTS adapters for CI;
- modular STT/TTS provider profiles for local engines and future external API
  adapters;
- local-first STT/TTS provider policy;
- audio capture/playback boundaries;
- audio storage disabled by default;
- transcript sensitivity and retention;
- cancellation/interruption semantics;
- cloud realtime model policy.

Baseline:

- spoken turns submit transcripts through existing conversation/request
  lifecycle;
- PM-08 `auto` routing is used for spoken turns;
- the gateway depends only on `SpeechToTextPort` and `TextToSpeechPort`, never
  on a concrete local model, command-line binary or external API client;
- speech providers are selected by configuration and policy, not hard-coded in
  the voice gateway;
- local speech providers are the default path;
- external speech API providers remain disabled by default and require explicit
  configuration, policy allow/approval and secret references, never raw secrets
  in prompts, logs or memory;
- cloud speech/realtime providers remain disabled by default;
- wake word, always-listening and barge-in are deferred.

Testing:

- fake STT/TTS unit and contract tests;
- provider profile tests covering local and external-api provider kinds without
  making network calls;
- voice turn e2e through fake STT/TTS/model providers;
- architecture tests preventing voice from bypassing runtime or loop selection;
- architecture tests preventing provider adapters from leaking into the voice
  gateway contract;
- privacy tests proving raw audio is not stored by default.

## 4. Medium-priority ADRs

### ADR-036 — Graph runtime adapter and LangGraph adoption gate

Needed before:

```text
planner-executor
durable code sandbox workflows
sleep/reflection execution
long-running maintenance workflows
multi-agent/subgraph orchestration
```

Decision scope:

- whether LangGraph becomes the graph runtime adapter for complex workflows;
- graph state schema and mapping to Jarvis domain objects;
- checkpoint storage ownership;
- relation between LangGraph checkpoints and Jarvis PostgreSQL tables;
- mapping LangGraph interrupts to Jarvis approval flow;
- mapping LangGraph streaming to Jarvis SSE/runtime events;
- adapter boundaries around `ContextAssemblerPort`, `ModelRouterPort`,
  `ToolGatewayPort`, `PolicyPort`, `EventLogPort` and stores;
- dependency and deployment implications;
- rollback plan if LangGraph adds too much coupling.

Baseline:

- PM-08 remains framework-agnostic and does not require LangGraph;
- PM-09 voice foundation does not require LangGraph;
- this is a follow-up item, not the immediate post-PM-08 implementation path;
- LangGraph may be adopted only behind a graph runtime adapter;
- existing custom loop strategies must keep working during the adoption test;
- no graph-backed workflow may bypass policy, approvals or ToolGateway.

Open questions:

- Should LangGraph checkpoints use Jarvis PostgreSQL directly, a separate schema
  or a replaceable checkpointer adapter?
- Which Jarvis request status maps to a paused LangGraph interrupt?
- Do graph node events become first-class EventLog entries or derived telemetry?
- Should the first graph-backed prototype be planner-executor, code sandbox or
  approval interrupt/resume?

Testing:

- fake graph workflow contract tests;
- architecture tests that graph nodes import only ports/domain schemas;
- approval interrupt/resume mapping tests;
- SSE/event translation tests;
- rollback tests proving custom loops still execute without LangGraph.

### ADR-037 — Planner-executor task model

Needed before long-running multi-step tasks.

Decision scope:

- task tables;
- plan and plan-step schema;
- agent-step events;
- pause/resume/cancel;
- correlation model;
- user clarification flow.

### ADR-038 — Scheduler and durable background workflows

Needed before reminders, proactive checks and sleep/reflection execution.

Decision scope:

- DB scheduler vs NATS/JetStream;
- worker process model;
- retry policy;
- idempotency;
- restart semantics.

### ADR-039 — Sleep/reflection workflow and memory candidates

Needed before autonomous consolidation.

Decision scope:

- event selection window;
- sleep report format;
- memory candidate generation;
- candidate approval;
- archive/supersede proposals;
- direct memory mutation restrictions.

### ADR-040 — External integration adapter model

Needed before Telegram, Spotify, GitHub, calendar, mail and similar adapters.

Decision scope:

- integration secrets;
- adapter lifecycle;
- fake servers;
- rate limits;
- side-effect policy;
- audit model.

### ADR-041 — MCP gateway

Needed before MCP server/tool support.

Decision scope:

- MCP server registry;
- capability mapping;
- tool schema import;
- resource access policy;
- process isolation;
- audit and approval integration.

## 5. Lower-priority ADRs

### ADR-043 — Cloud model enablement and fallback policy

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

### ADR-044 — WebSocket/control channel

Needed if SSE plus HTTP is insufficient for approvals, live task control or
voice/realtime sessions.

Decision scope:

- transport contract;
- authentication/local access;
- backpressure;
- reconnect semantics;
- relation to existing SSE events.

### ADR-045 — Artifact storage

Needed when tool/RAG outputs become too large for event payloads.

Decision scope:

- artifact references;
- retention;
- sensitivity;
- filesystem vs database;
- relation to ContextManifest and tool observations.

## 6. ADR dependency order

Recommended order:

```text
ADR-029 Capability and permission model (accepted)
ADR-030 ToolGateway boundary (accepted)
ADR-031 Agent loop strategy architecture (accepted)
ADR-032 Approval extensions/control channel (deferred after PM-05 baseline)
ADR-033 Shell sandbox (accepted)
ADR-034 Content Retrieval / Project Docs RAG (accepted)
ADR-035 Automatic loop strategy selection (accepted)
ADR-042 Voice gateway
ADR-036 Graph runtime adapter and LangGraph adoption gate
ADR-037 Planner-executor
ADR-038 Scheduler
ADR-039 Sleep/reflection
ADR-040 External integrations
ADR-041 MCP gateway
ADR-043 Cloud enablement
ADR-044 WebSocket/control channel
ADR-045 Artifact storage
```

Rationale:

- permissions must precede side effects;
- ToolGateway must precede tool-capable loop strategies;
- loop-strategy architecture must precede ReAct/planner loops;
- shell needs approval before write capability;
- RAG can proceed after context V2, but must remain separate from memory;
- voice should wait until PM-08d auto-routing and CLI readiness so spoken turns
  can use the same chat/RAG/tool/approval/cancel path as typed turns;
- graph runtime evaluation is deferred until planner-executor, durable code
  sandbox, sleep/reflection or long-running workflow pressure justifies it.

## 7. ADRs not needed yet

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
