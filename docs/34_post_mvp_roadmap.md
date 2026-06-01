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
- keep `memory_augmented_answer` as the default concrete strategy until the
  later `auto` selector is introduced;
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
- route malformed actions, repeated failures and budget exhaustion through explicit PM-08l failure/recovery/finalization policy.

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

### Phase I — PM-08 Automatic loop selection and CLI readiness

Goal:

```text
Make tools, RAG and ordinary chat available through one natural user-facing
chat surface without requiring users to choose internal loop strategies.
```

Work is split into sub-slices:

```text
PM-08a Loop selection domain and selector contract
PM-08b API/request lifecycle auto mode
PM-08c CLI auto mode and mode controls
PM-08d CLI tool/RAG/approval readiness surface
PM-08e Model-backed intent classifier adapter
PM-08f Typed tool observations and direct-answer hardening
PM-08g Historical capability routing registry cleanup
PM-08h Historical tool-intent corpus evidence
PM-08i Interactive CLI shell UX hardening
PM-08j Canonical Jarvis runtime startup
PM-08k Agentic loop-first request handling cleanup
PM-08l Agent loop architecture hardening gate
```

PM-08a through PM-08h describe the selector/classifier-era implementation path.
PM-08k supersedes that direction for production natural-language request
handling: the bounded agent loop is the default, and classifier/threshold
artifacts are historical, evaluation-only or quarantined follow-up material
unless an updated ADR explicitly changes the architecture. PM-08l then hardens
the agent loop internals and proves that the PM-08k path is ready for voice
through DB-backed transcript-like API/e2e turns and startup invariants.

PM-08a:

- add `auto` as the default user-facing routing mode;
- add backend `LoopStrategySelector`;
- add `IntentClassifierPort` so routing is not architecturally tied to keyword
  lists;
- add explicit selection/classification domain objects and confidence/fallback
  semantics;
- add capability routing metadata so future tools such as code sandbox can join
  auto-selection without bespoke selector branches;
- use fake classifier implementations in CI;
- start runtime with a local structured classifier when configured, and keep a
  conservative deterministic classifier as bootstrap/failure fallback;
- route ordinary chat to `memory_augmented_answer`;
- route project-docs questions to `memory_augmented_answer` with ContextAssembler
  RAG, not to a tool loop;
- route live project inspection and system diagnostics requests to
  `tool_react_loop`;
- keep tool execution behind `ToolGatewayPort`;
- keep policy and approval checks authoritative;

PM-08b:

- wire `auto` into API/request lifecycle;
- persist requested mode separately from selected concrete loop;
- resolve model profile after selected loop is known;
- emit redacted loop-selection events.

PM-08c:

- expose CLI/API overrides for `auto`, `chat` and `tools` for debugging;
- add interactive CLI `/mode auto|chat|tools`;
- make interactive CLI default to `auto`;
- keep routing rules on the backend, not in CLI.

PM-08d:

- prove ordinary CLI chat can reach RAG and safe/read-only tools through `auto`;
- render tool proposals, approvals and observations in user-facing language;
- support approve/deny/cancel from the normal interactive CLI flow;
- make request cancellation and Ctrl-C leave the CLI session usable;
- avoid raw JSON-first output for normal tool flow.

PM-08e:

- add a local structured model-backed `IntentClassifierPort` adapter for runtime
  auto-routing;
- keep fake classifiers for CI and deterministic classifier as failure fallback;
- keep `LoopStrategySelector` provider-agnostic and policy-authoritative;
- reject invalid classifier output rather than turning model text into raw tool
  commands.

PM-08f:

- keep typed tool observations as the durable answer contract after a tool has
  executed through the bounded agent loop and ToolGatewayPort path;
- propagate provider-neutral typed payload fields through
  `ToolInvocationResult -> ToolObservation -> ToolObservationRef -> formatter,
  context and events` while keeping bounded raw content for audit/debug and
  model fallback;
- use one typed contract:
  `structured_content`, `structured_schema`, `structured_schema_version`,
  `parse_status` and `parse_warnings`;
- move OS, battery, disk, VPN, memory and process-output interpretation into
  capability-specific adapters/normalizers with platform fixture tests;
- make user-visible answers consume typed fields only after policy/tool gates;
- answer from `parsed` payloads, answer cautiously from `partial` payloads, and
  route `unparsed` payloads to bounded model analysis or a clear unavailable
  result;
- when a tool returns raw or unrecognized output, either route the bounded
  observation through normal bounded-loop analysis or return a clear
  unavailable/unparsed result.

PM-08g:

- retain capability-routing registry lessons as selector-era historical evidence;
- keep tool-name validation lessons for PM-08l request-plan and ToolGateway
  registry checks;
- quarantine selector-era direct-routing metadata and duplicated allowlists as
  historical material unless a future ADR reintroduces them explicitly;
- do not treat PM-08g as a live direct-tool production path or PM-09 runtime gate.

PM-08h:

- turn the multilingual tool-intent corpus into pre-voice evaluation evidence;
- add exact CI baseline checks for critical tool names and direct-plan outcomes
  while those paths exist;
- add negative near-miss examples so conceptual questions do not become tools;
- cover datetime, calculator, daemon status, diagnostics, process search, VPN,
  disk, battery, CPU, memory, temperature, project inspection and ordinary chat;
- cover spoken-transcript-like variants before PM-09, including fillers, missing
  punctuation, wake-name prefixes, casing variants and mixed-language terms;
- keep misheard tool nouns as advisory evaluation cases until real STT output
  shows recurring errors worth promoting to hard-gate fixtures;
- keep CPU and memory direct-answer v1 aggregate-only, with per-core and
  per-process resource details deferred to later schemas;
- keep real local model evaluation out of CI; classifier model comparison is
  historical/evaluation evidence after PM-08k, not a PM-09 runtime gate;
- treat recorded PM-09 readiness evidence as spoken-transcript-like agent-loop
  cases, proving the transcript enters the same bounded loop and tool gateway
  path as typed input;
- if PM-08h-era local classifier evidence is missing, record that state
  explicitly without blocking the PM-08k agentic-loop-first direction.

PM-08i:

- add Codex-like inline terminal UX before voice starts;
- use `prompt_toolkit` for TTY prompts, completion menus, status toolbar,
  in-memory history and key bindings;
- keep non-TTY and `--plain` behavior deterministic and line-oriented;
- show user-facing mode, readiness, conversation, request phase, model summary
  and redacted cwd scope in a live status line;
- show request activity as real lifecycle phases rather than fake percentages;
- make slash command discovery dynamic while typing `/...`, including filtered
  descriptions and argument hints;
- keep chat, tool, approval, cancellation and error rendering readable and not
  raw JSON-first;
- keep routing and policy decisions on the backend, not in CLI code.

PM-08j:

- add one canonical local Jarvis runtime entrypoint before voice starts;
- stop relying on the test database target as the user-facing runtime path;
- define a persistent dogfood Postgres service separate from test DB teardown;
- add startup orchestration for dependency checks, DB readiness, migrations,
  daemon launch, PID/log management and health polling;
- expose Make wrappers such as `jarvis-up`, `jarvis-cli`, `jarvis-status`,
  `jarvis-logs`, `jarvis-down` and explicit reset;
- fail loudly when required dependencies or local models are missing, instead
  of silently falling back to an older CLI/daemon shape;
- keep dependency installation and model pulling in an explicit bootstrap step,
  not in normal startup.

PM-08k:

- start with an industry-informed request-routing architecture review before
  changing production routing code;
- reject both a mandatory front-gate LLM classifier and the deterministic-first
  Hybrid Request Resolver as production defaults;
- make the bounded agent loop the central request-handling path for
  natural-language typed input and future voice transcripts;
- remove runtime model-route adjudication, route thresholds and route-schema
  parsing from the production path;
- keep deterministic code for controls and safety only: slash commands,
  cancellation, approvals, policy, permissions, sensitivity, budgets,
  allowlists, schemas, redaction and non-TTY/plain behavior;
- keep model-origin tool proposals behind PolicyPort, ToolGatewayPort,
  allowlists, schemas and typed observations;
- ensure unsupported/risky tool attempts fail closed or ask clarification
  through the agent loop rather than being guessed by a pre-router;
- require invalid model output to abstain or fall back instead of inventing
  tool/capability metadata.

Acceptance:

- a plain CLI chat request can automatically use RAG or safe tools when needed;
- users do not need to type `tool_react_loop`, route names or tool names for
  normal usage;
- tools-disabled tool intent does not silently hallucinate live state;
- RAG remains ContextAssembler behavior and does not become a tool-loop trigger;
- approval-required tool flow is usable from the CLI;
- cancel/interrupt does not break the interactive CLI session;
- fake model-router responses, fake providers and fake adapters cover the
  agent-loop behavior in CI.
- direct answers are not built from fragile regex parsing of raw command stdout
  inside `tool_react_loop`;
- new diagnostics commands add adapter/normalizer tests instead of bespoke loop
  branches.
- the interactive CLI has a status line, live activity phases, dynamic slash
  command palette, in-session history, stable cancel/interrupt behavior and
  deterministic `--plain`/non-TTY fallback before PM-09 starts.
- one canonical Jarvis command path can bring up DB, migrations, daemon and
  health checks, and can report status/logs or shut the daemon down without
  manual Terminal orchestration.
- production request handling no longer asks a separate classifier to emit
  capabilities, tool names, risk classes or policy/fallback decisions before the
  agent loop.
- PM-08k reports false live-state positives, unsupported-tool behavior,
  policy-denial behavior, transcript-path parity and clarification/unavailable
  outcomes before PM-09 starts.

### Phase J — Voice gateway foundation

Goal:

```text
Add push-to-talk voice interaction on top of the existing runtime after PM-08d
proves the text CLI/API surface can use auto-selected chat, RAG, tools,
approvals and cancellation, PM-08f hardens typed observations, PM-08g/PM-08h remain historical selector-era evidence, PM-08i hardens the interactive CLI shell
as the pre-voice dogfood surface, and PM-08j makes that surface operationally
repeatable through canonical startup commands. PM-08k then replaces
classifier-first routing with agentic-loop-first request handling so voice does
not inherit a separate semantic router or threshold-tuned classifier path.
PM-08l then runs the agent-loop architecture hardening gate: transcript-like API
turns, agent-loop decomposition, single finalization, tool-observation recovery,
tool-gateway parity and DB-enabled verification must be green before voice code
starts.
```

Work:

- treat voice as a client/channel over existing conversations and request
  lifecycle, not as a separate agent runtime;
- add `SpeechToTextPort` and `TextToSpeechPort`;
- add provider-neutral speech provider profiles and registry so the same
  gateway can later use local engines, local libraries or external API adapters;
- add fake STT/TTS adapters for CI;
- submit transcripts through the same API/runtime path as typed input;
- use the PM-08k agentic loop path for spoken turns after PM-08l readiness;
- stream assistant text through existing runtime events before TTS output;
- disable raw audio storage by default;
- keep external speech API and cloud realtime providers disabled until explicit
  configuration and policy allow them;
- start with push-to-talk/local-session semantics.

Acceptance:

- fake voice e2e works without microphone, speaker or real STT/TTS provider;
- voice turns use the same conversation/runtime pipeline as CLI/API;
- transcripts have sensitivity policy;
- raw audio is not stored unless explicitly configured;
- voice gateway is provider-neutral and does not depend on a concrete local
  model, local binary or external API client;
- external speech API providers are represented as a future adapter path but are
  not called by default or in CI;
- spoken requests enter the PM-08k/PM-08l bounded agent loop through the
  same request-plan policy surface as typed turns.

### Phase K — External integrations

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

### Phase L — Planner-executor

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

### Phase M — Scheduler, proactive tasks and sleep/reflection

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

### Phase N — Advanced memory

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

### Phase O — Graph runtime adapter follow-up

Goal:

```text
Evaluate LangGraph or another graph runtime only when planner-executor, durable
code sandbox workflows, sleep/reflection or long-running automation need it.
```

Work:

- define a Jarvis graph runtime adapter boundary;
- add fake graph runtime contract tests first;
- evaluate LangGraph persistence/checkpoints, interrupts and streaming behind
  the adapter boundary if needed;
- map graph interrupts to Jarvis approval records and `waiting_approval`;
- map graph streams to Jarvis SSE/runtime events;
- prove existing custom loop strategies still run without LangGraph;
- document adopt/defer decision before any graph-backed production workflow.

Acceptance:

- LangGraph, if used, is isolated behind a graph adapter package;
- graph nodes call Jarvis ports, not concrete adapters;
- checkpoint state does not silently replace Jarvis PostgreSQL conversation,
  request, approval, event or memory tables;
- fake graph adapter covers CI without real LLM/tool calls;
- the output is an explicit adopt/defer decision.

## 4. Dependency map

```text
Capability/permissions
  -> ToolGateway
      -> LoopStrategy architecture
      -> safe tool loop
      -> CLI shell tools
      -> automatic loop selection
      -> CLI tool/RAG/approval readiness surface
          -> typed tool observations and direct-answer hardening
          -> historical capability routing registry cleanup
          -> historical tool-intent corpus evidence
          -> agent-loop architecture hardening gate
      -> voice gateway foundation
      -> MCP/integrations
      -> bounded ReAct loop details
          -> planner-executor

Context management V2
  -> ContentRetrieval/RAG
  -> automatic loop selection for project-docs questions
  -> tool-aware context
  -> planner-aware context
  -> voice/multimodal context

Scheduler/EventPublisher
  -> proactive tasks
  -> sleep/reflection
  -> long-running maintenance workflows
```

## 5. Near-term Alpha recommendation

The next implementation goal should make the implemented RAG and tools
reachable from normal chat before adding broader integrations or voice.

Current near-term sequence:

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
PM-08a Loop selection domain and selector contract
PM-08b API/request lifecycle auto mode
PM-08c CLI auto mode and mode controls
PM-08d CLI tool/RAG/approval readiness surface
PM-08e Model-backed intent classifier adapter
PM-08f Typed tool observations and direct-answer hardening
PM-08g Historical capability routing registry cleanup
PM-08h Historical tool-intent corpus evidence
PM-08i Interactive CLI shell UX hardening
PM-08j Canonical Jarvis runtime startup
PM-08k Agentic loop-first request handling cleanup
PM-08l Agent loop architecture hardening gate
Voice gateway foundation
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
