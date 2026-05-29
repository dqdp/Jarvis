# 34 — Post-MVP Roadmap

## Status

Proposed Alpha planning baseline.

Date: 2026-05-29

This roadmap starts after `mvp-0.1`. It does not change the accepted MVP
scope. New capabilities must be introduced through explicit ports, policy
checks, audit events and TDD slices.

## 1. Purpose

The MVP proved the local-first core daemon:

- durable conversations, messages, events and memory;
- deterministic context assembly;
- local model routing;
- SSE streaming;
- CLI dogfood path;
- architecture tests for ports/adapters boundaries.

Post-MVP work should now add the missing assistant capabilities without turning
`AgentRuntime` into a direct tool runner, RAG engine, voice pipeline or
integration hub.

## 2. Roadmap principles

Every post-MVP capability must satisfy the same rules:

```text
port first
policy before side effects
audit every decision and execution
fake providers/adapters for CI
bounded runtime budgets
no raw prompt or secret leakage
no cloud fallback without a dedicated ADR
```

New runtime loops must declare:

```text
max_steps
max_model_calls
max_tool_calls
max_wall_time_seconds
allowed_capabilities
policy_hooks
approval_rules
stopping_conditions
failure_semantics
emitted_events
```

## 3. Recommended implementation order

### Phase A — Roadmap and governance

Goal:

```text
Make post-MVP scope explicit before adding code.
```

Work:

- maintain this roadmap;
- maintain `docs/35_post_mvp_adr_backlog.md`;
- maintain `docs/36_post_mvp_plan_review.md`;
- maintain `docs/37_post_mvp_tdd_slices_plan.md`;
- update ADRs before architecture-changing implementation;
- update the TDD slice plan before changing implementation order.

Acceptance:

- README indexes all new documents;
- architecture documentation tests stay green;
- unresolved architecture questions are tracked explicitly.

### Phase B — Capability and permissions foundation

Goal:

```text
Create the policy foundation required before tools, integrations, RAG or voice.
```

Work:

- introduce a capability taxonomy;
- introduce permission modes, starting with `locked_down`, `developer_local`
  and `automation`;
- extend `PolicyPort` for tool calls, content source access, shell actions,
  external integrations and approval requirements;
- define risk classes for actions;
- define approval request/decision domain objects;
- emit audit events for allow, deny and approval decisions.

Initial capability examples:

```text
model.local
model.cloud
memory.read
memory.write
content.retrieve
tool.safe
tool.shell.read
tool.shell.write
tool.shell.network
integration.telegram
integration.spotify
voice.input
voice.output
```

Acceptance:

- policy can decide whether a requested action is allowed before the action
  exists;
- mode-specific defaults are testable and conservative;
- denied actions are auditable;
- `AgentRuntime` remains independent of concrete tools and integrations.

### Phase C — Context management V2

Goal:

```text
Prepare context assembly for tools, RAG, planner loops and voice without
changing the AgentRuntime contract.
```

Work:

- evolve provider-neutral context parts;
- add explicit references for tool observations and content hits;
- add context inspection CLI/API;
- add rolling summaries as current-context artifacts;
- add compression strategy hooks;
- extend `ContextManifest` with dropped reasons, budget use, source sensitivity
  and context source refs.

Acceptance:

- golden tests cover context shape, trimming and dropped refs;
- raw full prompt logging remains disabled by default;
- ModelRouter remains the only provider-specific conversion layer.

### Phase D — ToolGateway foundation

Goal:

```text
Add controlled tool execution without changing the agent loop yet.
```

Work:

- introduce `ToolGatewayPort`;
- define `ToolSpec`, `ToolCallRequest`, `ToolObservation` and tool invocation
  audit records;
- implement fake tool gateway for CI;
- implement safe built-in tools first;
- record tool lifecycle events.

Initial safe tools:

```text
datetime.now
calculator.evaluate
daemon.status
```

Optional after the first pass:

```text
conversation.lookup
memory.lookup
context.inspect
```

Acceptance:

- runtime can execute a specific approved tool call through the gateway;
- tool observations are not persisted as normal conversation messages by
  default;
- all tool calls are policy checked and audited.

### Phase E — LoopStrategy architecture

Goal:

```text
Separate the base MVP loop from future tool-capable and planner loops.
```

Work:

- extract the current deterministic workflow into a named
  `memory_augmented_answer` loop strategy without changing behavior;
- add a loop strategy registry;
- keep `memory_augmented_answer` as the default strategy;
- define loop budgets, loop results and step-level event concepts;
- make `ToolGatewayPort` an optional dependency for tool-capable strategies,
  not for the base loop.

Acceptance:

- MVP user-turn behavior remains unchanged;
- `memory_augmented_answer` still has `max_tool_calls=0`;
- a fake tool-capable strategy can be tested without real shell/MCP/tools;
- AgentRuntime selects a strategy instead of embedding every loop algorithm.

### Phase F — Safe tool loop v1

Goal:

```text
Add the first bounded tool-capable loop using only fake and safe tools.
```

Work:

- add `tool_react_loop` as a separate loop strategy;
- parse model-proposed actions into provider-neutral tool proposals;
- execute tools through `ToolGatewayPort`;
- return observations to loop context;
- enforce max steps, model calls, tool calls and consecutive failures;
- stop deterministically on malformed actions or budget exhaustion.

Acceptance:

- e2e fake model can request a fake/safe tool and produce a final answer;
- no shell, MCP or external integrations are required;
- repeated tool failures stop the loop deterministically.

### Phase G — CLI tools and shell sandbox

Goal:

```text
Expose useful local CLI/shell capabilities safely.
```

Work:

- implement shell adapter behind `ToolGatewayPort`;
- split PM-06 into project read-only shell and read-only system diagnostics;
- allowlist working roots;
- deny destructive/network/write operations by default;
- enforce timeouts and max output bytes;
- redact environment values;
- return structured output with truncation metadata;
- reserve approvals for future write or risky commands.

Initial read-only command family:

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

`git status` and `git ls-files` are read-only only with an explicit safe file
pathspec after `--`; whole-repository and directory index listings stay denied
to avoid leaking stale secret-like filenames from git metadata.
Direct reads of `.git` metadata through generic file readers are also denied,
and shell observations are at least `project` sensitivity.

Initial read-only diagnostics family:

```text
ps
pgrep
uptime
df
du inside allowlisted workspace roots
macOS: top -l 1, vm_stat, sysctl selected keys, netstat, ifconfig, lsof
Linux: top -b -n 1, free, lscpu, lshw, ss/netstat, ip addr, lsof, nvidia-smi
temperature sensors:
  macOS powermetrics without sudo when available
  Linux sensors
  Linux /sys/class/thermal read-only adapter
  GPU temperature through nvidia-smi query mode
```

Acceptance:

- shell execution is impossible without policy classification;
- command output is bounded;
- interactive diagnostics are denied by default;
- temperature diagnostics never request sudo or change fan/power settings;
- network diagnostics output is redacted and bounded;
- test suite uses fake shell adapter and does not depend on host state.

### Phase H — Content Retrieval / Project Docs RAG

Goal:

```text
Add RAG as a separate Content Retrieval subsystem, not as Memory.
```

Work:

- split delivery into `PM-07a Project Docs ingestion and citation index` and
  `PM-07b Project Docs retrieval and ContextAssembler integration`;
- add source registry, chunks and citations first;
- ingest project docs, ADRs and README as the first corpus in PM-07a;
- keep document chunks out of Memory tables;
- use deterministic markdown heading chunking with line-range citations;
- mark old chunks stale when source content changes;
- introduce `ContentRetrievalPort`, content embeddings and retrieval in
  PM-07b;
- integrate content hits into ContextAssembler as a separate section in PM-07b;
- emit content ingestion and retrieval events.

Acceptance:

- memory tables do not store document chunks;
- RAG answers cite content hits;
- ContextManifest records content hit refs separately from memory refs;
- source refresh and re-indexing are explicit workflows.

### Phase I — External integrations

Goal:

```text
Add real integrations as tool adapters, not runtime dependencies.
```

Candidate order:

```text
filesystem
MCP gateway
web/search
Telegram
GitHub
calendar/mail
Spotify
home automation
```

Acceptance:

- each integration has a fake adapter for CI;
- each integration has policy rules and secret handling;
- external side effects require approval unless explicitly configured safe.

### Phase J — Planner-executor

Goal:

```text
Support multi-step tasks that outgrow a single request/response loop.
```

Work:

- introduce task and plan domain objects;
- add tables for tasks, task runs, plans, plan steps and agent steps;
- support pause/resume/cancel;
- support user clarification and approval gates;
- use `correlation_id` for long-running workflows.

Acceptance:

- `assistant_requests` does not become the universal long-running task table;
- plan execution is auditable at step level;
- planner budgets are enforced.

### Phase K — Scheduler, proactive tasks and sleep/reflection

Goal:

```text
Add durable background workflows without free-running autonomy.
```

Work:

- introduce `SchedulerPort`;
- start with a DB-backed scheduler unless NATS/JetStream is justified;
- add reminders and periodic checks;
- implement bounded sleep/reflection workflow;
- write memory candidates instead of direct autonomous memory writes.

Acceptance:

- scheduled work survives daemon restart;
- sleep/reflection produces reviewable reports and candidates;
- direct memory mutation remains policy controlled.

### Phase L — Advanced memory

Goal:

```text
Move from manual memory to controlled memory evolution.
```

Work:

- memory candidate review;
- archive/supersede suggestions;
- consolidation jobs;
- hard purge/privacy workflow;
- reranking and better retrieval scoring.

Acceptance:

- model-generated memory changes are reviewable;
- lifecycle decisions are auditable;
- `secret` remains excluded from long-term memory.

### Phase M — Voice

Goal:

```text
Add voice interaction using the same runtime, policy and audit model.
```

Recommended order:

```text
push-to-talk
local VAD
local STT
local TTS
interrupt/barge-in
wake word
optional realtime model path after a dedicated ADR
```

Acceptance:

- voice turns use the same conversation/runtime pipeline as CLI/API;
- transcripts have sensitivity policy;
- audio is not stored unless explicitly configured;
- cloud realtime models remain disabled unless a future ADR changes policy.

## 4. Dependency map

```text
Capability/permissions
  -> ToolGateway
      -> LoopStrategy architecture
      -> safe tool loop
      -> CLI shell tools
      -> MCP/integrations
      -> bounded ReAct loop details
          -> planner-executor

Context management V2
  -> ContentRetrieval/RAG
  -> tool-aware context
  -> planner-aware context
  -> voice/multimodal context

Scheduler/EventPublisher
  -> proactive tasks
  -> sleep/reflection
  -> long-running maintenance workflows
```

## 5. Near-term Alpha recommendation

The next implementation goal should not start with RAG or voice. It should be:

```text
Capability and permissions foundation
Permission modes for locked_down, developer_local and automation
ToolGatewayPort with fake and safe tools
LoopStrategy architecture
safe tool loop with fake/safe tools
approval model and CLI/API control flow
project read-only shell tools
read-only system diagnostics tools
Project Docs RAG
```

This order gives useful capability quickly while preserving the local-first
security model and ports/adapters boundaries.

## 6. Explicit non-goals until the relevant phase

Do not implement these opportunistically:

```text
direct subprocess calls from AgentRuntime
direct MCP calls from AgentRuntime
document chunks in memory tables
cloud fallback
unbounded ReAct loop
background self-dialogue
voice wake word before push-to-talk
external side effects without policy and audit
```
