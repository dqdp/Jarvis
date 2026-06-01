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

The ReAct/tool loop is therefore the central runtime primitive for normal chat,
typed input and future voice transcripts.

## Consequences

- PM-08k implementation should remove runtime model-route adjudication, route
  threshold tuning and route-schema parsing from the production path.
- Thresholds such as `0.87` or `0.9` may remain only in historical/evaluation
  documents while comparing old classifier experiments. They are not runtime
  behavior for the agentic-loop-first default.
- Direct answer optimizations are not part of the default natural-language
  path. If reintroduced later, they require a separate ADR and must be limited
  to unambiguous, non-semantic operations with strong tests.
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

## Acceptance Notes

PM-08k is complete only when the documentation, tests and runtime agree that:

- natural-language requests enter the bounded agent loop by default;
- no runtime LLM route classifier is required before the agent loop;
- deterministic request handling is restricted to control/safety/policy
  responsibilities;
- unsupported or risky tool proposals fail through PolicyPort/ToolGatewayPort
  or clarification behavior, not hidden pre-routing guesses;
- PM-09 voice uses the same request lifecycle and agent loop as text.
