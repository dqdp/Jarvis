# ADR-019 — Model Profiles and ModelRouter Baseline

## Status

Accepted.

## Context

Phase 1 needs local LLM inference, structured outputs and embeddings, while preserving local-first privacy and avoiding direct dependency of `AgentRuntime` on a specific provider such as vLLM, Ollama or OpenAI.

The system must support future external LLM fallback without making cloud access accidental.

## Decision

Phase 1 defines required model profiles:

```text
local_main
local_structured
local_embedding
```

Phase 1 defines disabled future profile:

```text
cloud_reasoning
```

`local_fast` is not required for MVP.

`ModelRouter` remains an internal module/package inside the modular monolith.

The local inference node remains an external OpenAI-compatible process.

Phase 1 provider adapter is:

```text
local_openai_compatible
```

Embeddings are accessed through a narrow `EmbeddingPort`; its default implementation delegates to `ModelRouter.embed()` with `local_embedding`.

Structured output uses JSON schema prompting plus local validation. Provider-native structured output is allowed only as an optimization.

Retry baseline:

```text
chat retry = 0
structured validation retry = 1
embedding retry = 1
```

Fallback baseline:

```text
no automatic profile fallback
no automatic cloud fallback
```

Every model call must pass `PolicyPort`, create `model_invocation` audit record and emit model events.

## Rationale

This preserves fast MVP implementation while preventing backend lock-in.

`local_openai_compatible` avoids tying the runtime to vLLM specifically.

A separate `EmbeddingPort` keeps the memory subsystem independent from chat-oriented model routing while allowing centralized audit and policy enforcement.

Disabling automatic fallback protects local-first semantics.

## Consequences

Positive:

- provider-neutral runtime;
- local-first policy enforced through ModelRouter/PolicyPort;
- embeddings are replaceable;
- structured outputs are validated by our system;
- model invocations are auditable.

Trade-offs:

- one additional `EmbeddingPort`;
- no automatic fallback in MVP;
- local inference outage causes explicit request failure.

## Deferred

- `local_fast`;
- model pools;
- load-based routing;
- cost optimization;
- automatic cloud fallback;
- advanced structured repair;
- multi-node model scheduling;
- provider-specific tool calling.
