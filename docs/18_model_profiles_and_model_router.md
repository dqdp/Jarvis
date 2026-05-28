# 18 — Model Profiles and ModelRouter Details

## 1. Назначение

`ModelRouter` является стабильной границей между agent runtime и конкретными model providers.

Он не должен быть тонким клиентом к vLLM, Ollama или OpenAI. Его задача — нормализовать:

- model profiles;
- provider adapters;
- policy checks;
- structured output;
- embeddings;
- streaming;
- timeouts and retries;
- model invocation audit;
- provider-neutral request/response contracts.

Phase 1 должен работать с одним локальным backend, но не должен зашивать конкретный backend в `AgentRuntime`.

---

## 2. Архитектурная позиция

```text
AgentRuntime
  -> ContextAssemblerPort
  -> ModelRouterPort
      -> PolicyPort
      -> ModelProfileRegistry
      -> ProviderAdapterRegistry
      -> ModelInvocationLogger
      -> ProviderAdapter
          -> local OpenAI-compatible inference node
```

Phase 1:

- `ModelRouter` — internal module/package внутри modular monolith;
- local inference node — отдельный процесс;
- provider adapter — `local_openai_compatible`;
- cloud provider adapter может быть описан, но выключен.

---

## 3. Required Phase 1 model profiles

Phase 1 required profiles:

```text
local_main
local_structured
local_embedding
```

Disabled future profile:

```text
cloud_reasoning
```

Out of MVP:

```text
local_fast
model pool
load-based routing
cost optimizer
automatic cloud fallback
multi-node model scheduler
```

---

## 4. `local_main`

Основной чатовый профиль.

Используется для:

- обычных ответов ассистента;
- deterministic memory-augmented workflow;
- technical/project discussions;
- general assistant responses.

Properties:

```yaml
local_main:
  purpose: chat
  provider: local_openai_compatible
  enabled: true
  cloud: false
  supports_streaming: true
  supports_structured_output: false
  timeout_seconds: 120
  max_input_tokens: 12000
  max_output_tokens: 2048
  temperature: 0.3
```

Rules:

- policy check required;
- allowed for `public`, `project`, `personal`, `infra`;
- denied for `secret`;
- no automatic retry for streaming chat;
- every call creates `model_invocation`.

---

## 5. `local_structured`

Профиль для JSON/structured output.

Используется для будущих задач:

- memory candidate extraction;
- classification;
- intent classification;
- context metadata extraction;
- internal structured decisions.

Properties:

```yaml
local_structured:
  purpose: structured
  provider: local_openai_compatible
  enabled: true
  cloud: false
  supports_streaming: false
  supports_structured_output: true
  timeout_seconds: 60
  max_input_tokens: 8000
  max_output_tokens: 1024
  temperature: 0.0
  structured_output:
    mode: json_schema_prompt
    validation: local
    validation_retry: 1
```

Rules:

- local only;
- no cloud fallback;
- output must be parsed and validated locally;
- provider-specific structured output may be used only as optimization;
- invalid JSON/schema output may be retried once.

---

## 6. `local_embedding`

Профиль для embeddings.

Используется для:

- memory write embedding generation;
- retrieval query embedding;
- future document/RAG embeddings.

Properties:

```yaml
local_embedding:
  purpose: embedding
  provider: local_embedding
  enabled: true
  cloud: false
  timeout_seconds: 30
  batch_size: 32
  dimension: 1024
  retry: 1
```

Rules:

- embeddings are local only in Phase 1;
- embedding calls are audited as `model_invocations`;
- embedding backend is not hardcoded in memory subsystem;
- embedding failures are explicit errors, not silent fallbacks.

---

## 7. `cloud_reasoning`

Future external LLM profile.

Phase 1:

```yaml
cloud_reasoning:
  purpose: reasoning
  provider: openai
  enabled: false
  cloud: true
```

Rules:

- exists only as disabled future profile;
- `PolicyPort` denies all cloud calls in Phase 1;
- no automatic fallback to cloud;
- future enabling requires explicit ADR and privacy/redaction policy.

---

## 8. `local_fast`

`local_fast` is not required for MVP.

Rationale:

- deterministic Phase 1 loop does not require model-based routing;
- `local_structured` can cover early internal structured tasks;
- additional model profile increases operational complexity;
- can be introduced post-MVP for routing/classification/cheap extraction.

---

## 9. ModelRouterPort

Canonical interface:

```python
class ModelRouterPort(Protocol):
    async def chat(self, request: ChatModelRequest) -> ChatModelResponse: ...
    async def stream_chat(self, request: ChatModelRequest) -> AsyncIterator[ModelStreamEvent]: ...
    async def structured(self, request: StructuredModelRequest) -> StructuredModelResponse: ...
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
```

MVP implementation may implement `structured()` using `chat()` plus local JSON parsing/validation.

---

## 10. Provider-neutral canonical messages

`ModelRouter` accepts provider-neutral message/content structures.

`AgentRuntime` and `ContextAssembler` must not produce provider-specific OpenAI/vLLM/Ollama request dictionaries.

Provider-specific translation belongs inside provider adapters.

Phase 1 may use text-only content, but schemas must not prevent post-MVP typed content parts.

---

## 11. EmbeddingPort

Phase 1 introduces a narrow `EmbeddingPort`.

```text
Memory subsystem
  -> EmbeddingPort
      -> default implementation delegates to ModelRouter.embed(local_embedding)
```

Reason:

- Memory subsystem should not depend on chat-oriented ModelRouter details;
- embedding capability remains auditable and policy-aware;
- future embedding backend can be replaced without changing memory subsystem.

---

## 12. Policy integration

Before every model call, `ModelRouter` must consult `PolicyPort`.

Policy input includes:

- profile;
- provider;
- cloud flag;
- purpose;
- sensitivity;
- request_id;
- conversation_id where applicable.

Phase 1 policy rules:

```text
local model + non-secret sensitivity -> allow
secret sensitivity -> deny
cloud provider -> deny
```

Policy decisions emit `policy.decision.recorded`.

Denied model requests may emit `model.request.denied`.

---

## 13. Audit and events

Every model call creates a `model_invocations` row and emits model events.

Normal flow:

```text
policy.decision.recorded
model.request.created
model.response.received
```

Failure flow:

```text
policy.decision.recorded
model.request.created
model.request.failed
request.processing.failed
```

Denied flow:

```text
policy.decision.recorded
model.request.denied
request.processing.failed
```

Do not persist token-by-token streaming events in event log.

---

## 14. Retry and fallback baseline

Phase 1 retry policy:

```text
chat retry: 0
structured validation retry: 1
embedding retry: 1
```

Phase 1 fallback policy:

```text
no automatic profile fallback
no automatic cloud fallback
local inference unavailable -> controlled request failure
```

Rationale:

- local-first must not be violated accidentally;
- streaming chat retries are hard to make safe;
- MVP should fail explicitly rather than silently change model/provider.

---

## 15. MVP vs deferred

MVP includes:

- `local_main`;
- `local_structured`;
- `local_embedding`;
- disabled `cloud_reasoning`;
- `local_openai_compatible` provider adapter;
- `EmbeddingPort` thin wrapper;
- model invocation audit;
- policy checks;
- basic timeout/retry policy.

Deferred:

- `local_fast`;
- model pool;
- load-based routing;
- cost optimizer;
- automatic fallback;
- model benchmarking;
- provider-specific tool calling;
- advanced structured repair;
- multi-node model scheduling.


## 16. Retrieval baseline relation

The `local_embedding` profile is used by `EmbeddingPort`.

Phase 1 embedding usage is limited to explicit long-term memory records.

Document/RAG embeddings are deferred to a future Content Retrieval subsystem.


## Configuration relation

All model profiles are defined in configuration and validated at startup.

`cloud_reasoning.enabled` must be false by default.

Secrets such as API keys are referenced by environment variable name, not stored in YAML.
