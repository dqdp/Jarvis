# ADR-037 — Agentic Loop-First Request Handling

## Status

Accepted.

Date: 2026-06-01

## Context

PM-08a..PM-08k added automatic loop selection, safe/read-only tools, project
docs retrieval, a Codex-like CLI shell and canonical local startup. During
PM-08k we evaluated a mandatory LLM classifier, a deterministic-first hybrid
router, non-LLM semantic routing and a thinner model route schema.

That direction still left Jarvis with two places trying to understand the user:

```text
request router / classifier
main agent loop
```

The duplication is especially risky before voice. Spoken transcripts are noisy,
often contain filler words, partial phrases, corrections and ASR mistakes, and
make brittle lexical routing rules more visible. A separate route classifier
also adds latency and configuration complexity before the actual agent gets a
chance to reason over the request.

## Decision

Jarvis uses an agentic-loop-first default for natural-language user input.

```text
user text or voice transcript
  -> single bounded agent loop
  -> model decides whether to answer or propose a tool call
  -> PolicyPort / ToolGatewayPort validate and execute proposed tool calls
  -> tool observations are returned to the same loop
  -> final answer
```

There is no runtime LLM route classifier in the default user path. There is also
no deterministic intent router that tries to understand arbitrary natural
language before the agent loop.

Deterministic code remains responsible for control and safety, not for semantic
request understanding:

- slash commands and client controls such as `/cancel`, `/exit`, approvals and
  prompt handling;
- policy, permission, sensitivity, budget and approval gates;
- tool allowlists and schema validation;
- redaction, logging and event-shape constraints;
- non-TTY/plain CLI fallback behavior;
- explicit API/user overrides where they are operational controls, not hidden
  natural-language inference.

Safety guardrails may require evidence before finalization. This is not a
semantic route classifier and not a direct natural-language execution path. For
example, if a request appears to ask for a current live fact such as local time
and an allowed local tool can observe it, the loop must not accept an
unevidenced `final_answer` that asserts that fact. The model may still choose
the specific allowed tool through the bounded loop, and execution still goes
through PolicyPort and ToolGatewayPort.

The evidence guard uses broad live-state intent families rather than exact
deterministic-question allowlists. It must cover current time/date wording,
including "what time is it", "current time", "local time", `сколько времени`,
`который час`, `текущее время`, `в данный момент` and `сейчас`; local machine,
process and daemon state such as CPU, memory, load, battery, network/VPN/IP,
disk, hardware, process/service/PID and status wording; and current live values
combined with arithmetic, threshold or comparison wording. A match only means
that completed evidence is required before a live-state claim can be accepted.
For process-scoped claims, the completed observation must match the requested
process identity. Per-process CPU or memory claims require typed process-resource
evidence, not aggregate `tool.system.read.resources` evidence.

The guard algorithm is intentionally limited: build the candidate live-state
tool set from explicit metadata and allowed local tools; lightly normalize user
text; match broad live-state intent families; produce typed guard metadata with
the live-state family, evidence requirement and candidate tools; block
unevidenced live-state `final_answer` when a relevant local tool is allowed;
then finalize only after a matching completed observation. A future
unavailable/clarification contract is required before the no-allowed-tool path
can claim the same hard guard. Location or scope wording may prevent a narrow
deterministic finalizer, but it must not disable the evidence guard when a
matching local observation is available.

Deterministic finalization is narrower than the evidence guard. It is allowed
only as a small source-backed transformation of a completed typed observation,
such as formatting a completed `datetime.now` observation as `HH:MM`. It must
not expand into broad natural-language routing, countdown calculation,
calendar reasoning or direct tool execution before the loop.

The exception for "what tools are available now?" is also source-backed and
narrow: it answers from the current request's `ToolRequestPlan`, specifically
`allowed_tool_names` and matching safe summaries. It must not consult RAG or a
global tool registry, and it must not reveal disabled or hidden tools. Questions
about tool architecture, documentation or external ecosystems remain ordinary
model/RAG-capable requests, and compound requests remain inside the bounded loop.

The ReAct/tool loop is therefore the central runtime primitive for normal chat,
typed input and future voice transcripts.

PM-08l refines this into a bounded typed agent-loop contract. ReAct is a design
influence, not the runtime wire protocol: Jarvis must not depend on free-form
`Thought/Action/Observation` transcript parsing. Tool proposals, observations,
finalization, recovery and lifecycle streaming are typed runtime states behind
the existing ports/adapters boundaries.

The bounded typed loop is also the intended executor for future
plan-and-execute workflows. Simple requests should not pay a mandatory planning
latency cost, while future compound tasks may introduce a planner shell that
executes scoped plan steps through the same bounded loop, policy gates and
ToolGateway path.

## Consequences

- PM-08k implementation should remove runtime model-route adjudication, route
  threshold tuning and route-schema parsing from the production path.
- Thresholds such as `0.87` or `0.9` may remain only in historical/evaluation
  documents while comparing old classifier experiments. They are not runtime
  behavior for the agentic-loop-first default.
- Direct answer optimizations are not part of the default natural-language
  path. If reintroduced later, they require a separate ADR and must be limited
  to unambiguous, non-semantic operations with strong tests. Narrow
  source-backed deterministic finalization after a completed typed observation
  is part of the bounded loop contract and is not a direct execution route.
- Voice must submit transcripts through the same agent loop as typed input and
  must not add a separate voice-specific router.
- Tool choice belongs inside the bounded agent loop. Tool execution still
  belongs outside the model, behind PolicyPort and ToolGatewayPort.

## Non-Goals

- This ADR does not introduce unbounded autonomous agents.
- This ADR does not remove PolicyPort, ToolGatewayPort, approvals, budgets or
  typed tool observations.
- This ADR does not require a full-screen TUI, voice implementation or durable
  workflow checkpoints.
- This ADR does not implement a full planner-executor or plan-and-execute
  runtime.

## Acceptance Notes

PM-08k is complete only when the documentation, tests and runtime agree that:

- natural-language requests enter the bounded agent loop by default;
- no runtime LLM route classifier is required before the agent loop;
- deterministic request handling is restricted to control/safety/policy
  responsibilities;
- live-state claims require relevant completed evidence when an allowed local
  tool can observe that state;
- unsupported or risky tool proposals fail through PolicyPort/ToolGatewayPort
  or clarification behavior, not hidden pre-routing guesses;
- PM-09 voice uses the same request lifecycle and agent loop as text.
- PM-08l hardens the loop state machine, final-answer path, tool-observation
  recovery and stream/replay semantics before PM-09 starts.
