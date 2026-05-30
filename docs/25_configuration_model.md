# 25 — Configuration Model

## 1. Purpose

This document defines the Phase 1 configuration model.

The goal is to keep MVP configuration simple while ensuring that important architectural defaults are not hardcoded.

---

## 2. Core decision

Phase 1 uses:

```text
YAML config + environment overrides
```

Config is loaded and validated at startup.

Invalid configuration causes startup failure.

No hot reload is required in MVP.

---

## 3. Config precedence

Config source priority:

```text
defaults < local YAML < environment variables
```

Recommended files:

```text
config/default.yaml
config/local.yaml
config/test.yaml
```

Secrets are not stored in YAML.

Environment overrides use one canonical MVP convention:

```text
prefix: JARVIS_
nested keys: double underscore
example: JARVIS_API__PORT=8081
example: JARVIS_MODEL_PROFILES__LOCAL_MAIN__ENDPOINT=http://127.0.0.1:8000/v1
```

Override values are parsed into the target `Settings` field type during
startup validation. Environment values may provide secret values or secret
environment variable names, but resolved secrets must not be written back to
YAML, raw logs, event payloads or prompts.

---

## 4. Secrets

Secrets must never be stored in committed YAML config.

Forbidden in YAML:

```text
API keys
OpenAI keys
Telegram bot tokens
SSH keys
database passwords in repo
private credentials
session cookies
```

Allowed secret sources:

```text
environment variables
.env.local ignored by git
Docker secrets
future secret manager
```

MVP may use `.env.local`, but it must not be committed.

---

## 5. Startup validation

A `ConfigLoader` must:

```text
load YAML
apply env overrides
validate schema
resolve references
fail fast if invalid
```

Examples of startup failures:

```text
local_main profile missing
local_embedding profile missing
model profile references missing provider
invalid sensitivity class
invalid namespace allowed_types
cloud profile or cloud policy enabled in Phase 1 without a future ADR
secret logging enabled
```

---

## 6. Config sections

Phase 1 config should cover:

```text
app
database
api
model_profiles
memory
context_assembly
runtime_budgets
policy
privacy
observability
```

These sections may be kept in one file for MVP.

---

## 7. Example config

```yaml
app:
  environment: local
  instance_id: personal-assistant-local
  default_user_id: local_user
  default_language: ru

database:
  url_env: DATABASE_URL
  pool_size: 10
  echo_sql: false

api:
  host: 127.0.0.1
  port: 8080
  request_timeout_seconds: 180
  sse_heartbeat_seconds: 15

model_profiles:
  local_main:
    purpose: chat
    provider: local_openai_compatible
    enabled: true
    cloud: false
    model: qwen3-32b-instruct
    endpoint: http://inference-node:8000/v1
    timeout_seconds: 120
    max_input_tokens: 12000
    max_output_tokens: 2048
    temperature: 0.3
    supports_streaming: true

  local_structured:
    purpose: structured
    provider: local_openai_compatible
    enabled: true
    cloud: false
    model: qwen3-14b-instruct
    endpoint: http://inference-node:8000/v1
    timeout_seconds: 60
    max_input_tokens: 8000
    max_output_tokens: 1024
    temperature: 0.0
    structured_output:
      mode: json_schema_prompt
      validation: local
      validation_retry: 1

  local_embedding:
    purpose: embedding
    provider: local_embedding
    enabled: true
    cloud: false
    model: qwen3-embedding-0.6b
    endpoint: http://inference-node:8001
    dimension: 1024
    timeout_seconds: 30
    batch_size: 32
    retry: 1

  cloud_reasoning:
    purpose: reasoning
    provider: openai
    enabled: false
    cloud: true
    model: gpt-placeholder
    api_key_env: OPENAI_API_KEY

memory:
  allowed_types: [fact, preference, procedure, summary]

  namespaces:
    user.preferences:
      sensitivity: personal
      allowed_types: [preference]
      default_retrieval: true

    user.working_style:
      sensitivity: personal
      allowed_types: [preference, procedure, summary]
      default_retrieval: true

    project.personal_assistant:
      sensitivity: project
      allowed_types: [fact, procedure, summary]
      default_retrieval: when_active_project

    system.runtime_rules:
      sensitivity: project
      allowed_types: [fact, procedure]
      default_retrieval: always

    environment.inference_node:
      sensitivity: infra
      allowed_types: [fact, procedure, summary]
      default_retrieval: when_relevant

  retrieval:
    max_hits_total: 8
    max_hits_per_namespace: 4
    min_score: null
    include_statuses: [active]
    exclude_sensitivity: [secret]

context_assembly:
  full_prompt_logging: false

  sections:
    order:
      - system_identity
      - runtime_rules
      - user_preferences
      - working_style
      - project_or_environment_memory
      - recent_conversation
      - current_user_message
      - output_contract

  conversation_window:
    max_messages: 12
    max_tokens: 3000
    include_roles: [user, assistant]
    exclude_sensitivity: [secret]
    trimming_strategy: drop_oldest_first

  context_budget:
    max_input_tokens: 12000
    memory_tokens_max: 2500
    recent_conversation_tokens_max: 6000

runtime_budgets:
  memory_augmented_answer:
    max_model_calls: 1
    max_tool_calls: 0
    max_wall_time_seconds: 180
    max_context_assembly_seconds: 10
    max_memory_retrieval_seconds: 5
    max_model_call_seconds: 120
    max_output_tokens: 2048
    allow_cloud: false
    allow_tools: false
    allow_autonomous_memory_write: false

policy:
  cloud_models_enabled: false
  tools_enabled: true
  autonomous_memory_write_enabled: false

  model_access:
    local:
      allow_sensitivity: [public, project, personal, infra]
      deny_sensitivity: [secret]

    cloud:
      enabled: false
      allow_sensitivity: []
      deny_sensitivity: [public, project, personal, infra, secret]

  memory_write:
    deny_sensitivity: [secret]

  context_inclusion:
    deny_sensitivity: [secret]

privacy:
  sensitivity_order: [public, project, personal, infra, secret]
  raw_prompt_logging: false
  raw_secret_logging: false
  store_context_manifest: true

observability:
  log_level: info
  structured_logs: true
  log_raw_messages: false
  log_raw_prompts: false
  log_model_outputs: false
  metrics_enabled: true
```

---

## 8. Hardcoded vs configurable

Domain invariants may be represented as enums:

```text
core sensitivity values
core memory types
event envelope required fields
request status enum
MVP message roles
```

Must be configurable:

```text
model endpoint
model name
timeouts
runtime budgets
retrieval top-k
context window size
namespace registry
cloud enabled/disabled
raw prompt logging
```

---

## 9. Config vs Policy

Config stores values.

Policy makes decisions.

Example:

```text
policy.cloud_models_enabled=false
```

is config.

At runtime:

```text
PolicyPort.evaluate_model_request(...)
```

returns:

```text
deny: cloud_models_disabled_by_default
```

The code must not scatter raw config checks across runtime components.

MVP implementation:

```text
PolicyPort -> ConfigPolicyEngine
```

---

## 10. Config ownership

Logical sections:

```text
AppConfig
DatabaseConfig
ApiConfig
ModelConfig
MemoryConfig
ContextConfig
RuntimeBudgetConfig
PolicyConfig
PrivacyConfig
ObservabilityConfig
```

Implementation may expose a single typed `Settings` object.

---

## 11. No hot reload in MVP

Phase 1 loads configuration at startup.

Config changes require restart.

Rationale:

- simpler implementation;
- avoids race conditions;
- easier testing;
- stable runtime behavior.

---

## 12. Environment profiles

MVP profiles:

```text
local
test
```

No SaaS/prod complexity in MVP.

---

## 13. Config validation tests

Required tests:

```text
default config validates
test config validates
JARVIS_ env overrides apply through double-underscore nested keys
cloud_reasoning disabled by default
secret not allowed in memory_write
local_main profile exists
local_structured profile exists
local_embedding profile exists
memory namespaces valid
runtime budget max_model_calls=1 for memory_augmented_answer
raw_prompt_logging=false by default
```

These tests keep documentation decisions executable.

---

## 14. MVP vs deferred

MVP includes:

```text
YAML config
env overrides
strict startup validation
model profile registry
memory namespace registry
runtime budget config
policy config
privacy/logging flags
no secrets in YAML
no hot reload
local/test configs
config validation tests
```

Deferred:

```text
dynamic config reload
admin config UI
policy DSL
secret manager integration
multi-user config
tenant config
remote config service
feature flag service
runtime model profile switching UI
```
