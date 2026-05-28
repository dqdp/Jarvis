# 17. Data Sensitivity and Privacy Policy — Phase 1

## 1. Назначение

Этот документ фиксирует минимальную модель чувствительности данных для Phase 1.

Цель Phase 1 — не построить полноценную privacy/security platform, а заложить простую и обязательную классификацию данных, чтобы система не могла случайно:

- сохранить секреты в long-term memory;
- включить секреты в prompt context;
- отправить приватные данные во внешнюю LLM;
- записать raw prompt или raw secret в event log / application logs;
- смешать публичные, персональные, проектные, инфраструктурные и секретные данные.

Главный принцип:

> Phase 1 uses a minimal sensitivity model as an architectural safety baseline. Advanced privacy enforcement is deferred.

---

## 2. Sensitivity classes

Phase 1 использует закрытый набор sensitivity labels:

```text
public
project
personal
infra
secret
```

Новые классы чувствительности не добавляются автоматически. Добавление нового класса требует обновления документации или ADR.

---

## 3. Meaning of sensitivity classes

### 3.1 `public`

Публичные или несекретные данные.

Примеры:

- официальная документация библиотек;
- публичные статьи;
- публичные технические факты;
- общедоступные ссылки.

### 3.2 `project`

Проектные данные пользователя.

Примеры:

- архитектура личного ассистента;
- ADR;
- проектная документация;
- планы реализации;
- engineering decisions.

### 3.3 `personal`

Личные данные пользователя общего характера.

Примеры:

- preferences;
- working style;
- личные заметки;
- обычные диалоги;
- персональные, но не критичные сведения.

### 3.4 `infra`

Инфраструктурные данные.

Примеры:

- сведения о серверах;
- сетевые настройки;
- hostname/IP/paths;
- local inference node details;
- сведения о локальных сервисах;
- SSH/Linux/system context.

`infra` считается более чувствительным, чем `personal`, потому что утечка инфраструктурного контекста может помочь получить доступ к системам.

### 3.5 `secret`

Секреты и credentials.

Примеры:

- API keys;
- access tokens;
- passwords;
- SSH private keys;
- seed phrases;
- cookies/session tokens;
- private credentials.

`secret` имеет специальные hard rules: не хранить в memory, не включать в prompt, не отправлять во внешнюю LLM, не логировать raw.

---

## 4. Sensitivity ordering

Для Phase 1 используется следующий ordering:

```text
public < project < personal < infra < secret
```

Если artifact собирается из нескольких источников, его sensitivity равна максимальному sensitivity included sources.

Пример:

```text
context = project memory + personal preference + infra node fact
context.sensitivity = infra
```

---

## 5. Where sensitivity is stored

Sensitivity label должен присутствовать минимум в:

```text
events.sensitivity
messages.sensitivity
memories.sensitivity
model_invocations.sensitivity
context_manifest.sensitivity_summary
```

### 5.1 Events

Каждое событие имеет поле `sensitivity` в `EventEnvelope`.

### 5.2 Messages

Каждое user/assistant message имеет sensitivity label.

Default:

```text
user.message      → personal
assistant.message → personal
```

### 5.3 Memories

Каждый `MemoryRecord` имеет sensitivity label.

Default определяется namespace registry.

### 5.4 Model invocations

Каждый model invocation должен фиксировать input/output sensitivity.

MVP может хранить одно поле:

```text
model_invocations.sensitivity
```

Post-MVP может разделить:

```text
input_sensitivity
output_sensitivity
```

### 5.5 ContextManifest

`ContextManifest` хранит:

```text
sensitivity_summary
max_sensitivity
sources_by_sensitivity
```

Full raw prompt не хранится по умолчанию.

---

## 6. Default assignment rules

Phase 1 не использует LLM-based sensitivity classifier.

Sensitivity назначается через:

1. defaults by namespace;
2. defaults by source/event type;
3. manual override через API/config;
4. future simple guards/detectors, не входящие в MVP.

### 6.1 Namespace defaults

```text
user.preferences              → personal
user.working_style             → personal
project.personal_assistant     → project
system.runtime_rules           → project
environment.inference_node     → infra
```

### 6.2 Event/source defaults

```text
user.message.created           → personal by default
assistant.message.created      → personal by default
context.assembled              → max sensitivity of included sources
memory.retrieved               → max sensitivity of retrieved memories
model.request.created          → max sensitivity of context
model.response.received        → same or derived from request
policy.decision.recorded       → project/internal by default
```

---

## 7. Long-term memory rules

### 7.1 Allowed

`MemoryWritePort` may create memories with sensitivity:

```text
public
project
personal
infra
```

subject to namespace/type validation.

### 7.2 Forbidden

`secret` cannot be stored as a `MemoryRecord`.

If user requests storing a secret, Phase 1 behavior:

1. reject memory write;
2. do not store raw secret in memory;
3. emit redacted event;
4. return user-visible explanation that secrets should not be stored in memory;
5. defer secure secret manager integration.

---

## 8. Context assembly rules

`ContextAssembler` must apply sensitivity constraints.

Phase 1 rules:

```text
public/project/personal/infra → can be included in local model context
secret                        → excluded from context
```

Hard rule:

> `secret` is never included in prompt context, even for local models.

Reasons:

- local model can echo secrets;
- debug/trace paths may leak context;
- future cloud routing could accidentally expose context;
- raw prompt is not trusted as a secret container.

`ContextManifest` records selected/dropped sources and sensitivity summary.

---

## 9. Model routing rules

`ModelRouter` must consult `PolicyPort` for model provider access.

### 9.1 Local models

Phase 1:

```text
allowed: public, project, personal, infra
forbidden: secret
```

### 9.2 Cloud models

Phase 1:

```text
deny all by default
```

The disabled cloud profile may exist in configuration, but all cloud calls are denied unless a future ADR explicitly enables them.

Post-MVP possible policy:

```text
public    → may allow
project   → explicit policy required
personal  → explicit policy required
infra     → deny by default
secret    → always deny
```

This post-MVP policy is not part of Phase 1.

---

## 10. Event log rules

Event log is historical truth about system actions, but it must not become a secret dump.

Phase 1 event log rules:

```text
- every event has sensitivity;
- raw full prompts are not stored;
- raw secrets are not stored;
- events use refs/hashes/manifests where possible;
- secret payloads must be redacted;
- raw message content lives in messages table, not duplicated by default in events.
```

Example redacted event payload:

```json
{
  "message_id": "...",
  "content_hash": "sha256:...",
  "redacted": true,
  "redaction_reason": "contains_secret"
}
```

---

## 11. Operational logging rules

Application logs must not contain raw private content by default.

Allowed logs:

```text
request_id
conversation_id
event_id
model_profile
latency
token counts
memory ids
context section names
error codes
```

Forbidden by default:

```text
raw user messages
raw full prompts
raw secrets
raw credentials
large raw model outputs
```

Model outputs are stored in `messages` as domain data, not duplicated in operational logs.

---

## 12. PolicyPort integration

Phase 1 uses minimal `PolicyPort`.

Required decisions:

```text
evaluate_model_request
evaluate_memory_write
evaluate_context_inclusion
```

`evaluate_context_inclusion` is minimal in Phase 1: it denies `secret` sources
and allows non-secret sources for local context.

Deferred:

```text
evaluate_tool_call
evaluate_cloud_redaction
evaluate_secret_access
```

Minimal policy implementation:

```text
ConfigPolicyEngine
  allow_local_model: true
  allow_cloud_model: false
  allow_tools: false
  allow_secret_in_memory: false
  allow_secret_in_context: false
```

---

## 13. MVP scope

Included in MVP:

- sensitivity enum;
- sensitivity fields in core tables;
- default assignment by namespace/source;
- PolicyPort denies cloud by default;
- MemoryWritePort rejects `secret`;
- ContextAssembler excludes `secret`;
- raw prompt logging disabled;
- event payloads use refs/hashes/manifests by default.

---

## 14. Explicitly deferred

Not included in MVP:

- LLM-based sensitivity classifier;
- full PII/secret detector;
- policy DSL;
- privacy dashboard;
- encryption per sensitivity class;
- secret manager integration;
- cloud redaction pipeline;
- tool credential injection;
- fine-grained per-tool privacy policies;
- automated data retention/purge workflows.

---

## 15. Acceptance rules

Phase 1 implementation must satisfy:

1. `secret` cannot be stored as memory.
2. `secret` cannot be included in prompt context.
3. cloud model calls are denied by default.
4. full raw prompts are not stored by default.
5. every event has a sensitivity label.
6. every memory has a sensitivity label.
7. ContextManifest records max sensitivity.
8. model invocations record input/context sensitivity.
9. operational logs do not contain raw prompts or secrets.

---

## 16. Summary decision

Phase 1 introduces a minimal sensitivity model:

```text
public
project
personal
infra
secret
```

`secret` is never stored as long-term memory, never included in model context, never sent to cloud, and never logged raw.

Cloud model access is denied by default for all sensitivity classes.

Advanced privacy enforcement is deferred.


## Configuration relation

Sensitivity defaults, denied sensitivity classes, raw prompt logging and raw secret logging are configured through PolicyConfig/PrivacyConfig.

Policy decisions are made through `PolicyPort`, not through scattered direct config checks.
