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

The local inference backend remains external to `AgentRuntime` and is accessed
only through provider adapters behind `ModelRouter`.

Phase 1 provider adapters are:

```text
local_openai_compatible
local_embedding
ollama
```

`ollama` is the native Ollama adapter. It is accepted for local dogfood because
some local Qwen builds expose usable output through Ollama native `/api/chat`
while their OpenAI-compatible endpoint can separate reasoning from final
content in a way that produces empty assistant text for this MVP adapter.

Embeddings are accessed through a narrow `EmbeddingPort`; its default
implementation delegates to `ModelRouter.embed()` with the `local_embedding`
profile and provider key.

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

`local_openai_compatible` and the native Ollama adapter avoid tying the runtime
to a specific model server while preserving the same `ModelRouterPort` contract.

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
