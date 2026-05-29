# 09 — Model Router and Inference

## 1. Decision

`ModelRouter` is an internal module/package in Phase 1.

Local inference is external to `AgentRuntime` and accessed only through
provider adapters behind `ModelRouter`.

The runtime never calls vLLM, Ollama, OpenAI, NIM or provider-specific clients directly.

Detailed profile and routing rules are defined in:

```text
18_model_profiles_and_model_router.md
ADR-019_model_profiles_and_model_router_baseline.md
```

---

## 2. Responsibilities

`ModelRouter`:

- hides concrete providers;
- routes by explicit model profile;
- supports streaming chat;
- supports structured output through local validation;
- supports embeddings through `local_embedding`;
- consults `PolicyPort` before every model call;
- logs every model invocation;
- emits model-related events;
- never silently sends data to external LLM;
- does not store raw full prompts by default.

---

## 3. Required Phase 1 profiles

Required:

```text
local_main
local_structured
local_embedding
```

Future disabled:

```text
cloud_reasoning
```

Not required for MVP:

```text
local_fast
```

---

## 4. Provider adapters

Phase 1 provider adapters:

```text
local_openai_compatible
local_embedding
ollama
```

The `local_openai_compatible` adapter may point to vLLM, Ollama
OpenAI-compatible mode, NIM or another local OpenAI-compatible serving
endpoint. The `local_embedding` adapter is the narrow embedding-provider key
used by the `local_embedding` profile.

The native Ollama adapter is accepted for local dogfood when the
OpenAI-compatible endpoint does not expose final assistant content in the shape
required by the MVP adapter.

These adapters avoid a vLLM-specific runtime dependency in Phase 1.

Future adapters:

- OpenAIProviderAdapter;
- LlamaCppProviderAdapter;
- NIMProviderAdapter;
- multi-node router adapter.

---

## 5. Embeddings

Phase 1 introduces a narrow `EmbeddingPort`.

Default implementation:

```text
EmbeddingPort -> ModelRouter.embed(profile=local_embedding)
```

The memory subsystem must not depend on a concrete embedding provider.

---

## 6. Structured output

Structured outputs use:

```text
JSON schema prompt + local validation
```

Provider-native structured output may be used only as an optimization.

Invalid structured output may be retried once.

---

## 7. Policy

Phase 1 rules:

```text
local model:
  allowed for public/project/personal/infra
  forbidden for secret

cloud model:
  denied by default for all sensitivity classes
```

`cloud_reasoning` may exist in configuration, but is disabled and denied by policy.

---

## 8. Retry and fallback

Phase 1 retry baseline:

```text
chat retry: 0
structured validation retry: 1
embedding retry: 1
```

Phase 1 fallback baseline:

```text
no automatic profile fallback
no automatic cloud fallback
```

If local inference is unavailable, request processing fails explicitly and emits error events.

---

## 9. Extraction path

`ModelRouter` can later become a standalone service if:

- multiple assistant runtimes need it;
- multiple inference nodes exist;
- routing by latency/load/cost is needed;
- model pool management is needed;
- model admin UI is required.

This extraction must not change `AgentRuntime`, which depends only on `ModelRouterPort`.


## Configuration relation

Model profiles are config-driven.

`local_main`, `local_structured`, `local_embedding` and disabled `cloud_reasoning` must be validated at startup.

Model endpoints, model names, timeouts and max tokens must not be hardcoded.
