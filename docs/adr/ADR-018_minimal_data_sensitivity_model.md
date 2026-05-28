# ADR-018: Minimal Data Sensitivity Model

## Status

Accepted.

## Context

The assistant is intended to run as a local-first, always-on personal system. It will eventually access personal information, project data, infrastructure details, external LLMs, tools, MCP integrations, Telegram, voice, search and Linux CLI.

Phase 1 does not implement those advanced integrations, but it must prevent early architectural mistakes that would leak sensitive data into memory, prompts, logs or future cloud calls.

## Decision

Phase 1 introduces a minimal closed sensitivity model:

```text
public
project
personal
infra
secret
```

All core data records must carry a sensitivity label:

- events;
- messages;
- memories;
- model invocations;
- context manifests.

Sensitivity defaults are assigned by namespace and source type. Phase 1 does not use an LLM-based sensitivity classifier.

`secret` has hard rules:

- never stored as long-term memory;
- never included in prompt context;
- never sent to cloud;
- never logged raw;
- only redacted refs/hashes/manifests may appear in events.

Cloud model access is denied by default for all sensitivity classes in Phase 1.

`PolicyPort` enforces at least:

- model request decisions;
- memory write decisions.

Advanced privacy features are deferred.

## Rationale

This provides a small but enforceable baseline:

- protects against accidental cloud leakage;
- prevents secrets from entering long-term memory;
- preserves local-first architecture;
- keeps MVP small;
- provides future hooks for cloud redaction, tool policies and secret manager integration.

The chosen classes are broad enough for Phase 1 and future extension, but small enough to avoid building a full privacy ontology.

## Consequences

Positive:

- all core artifacts are classifiable;
- ContextAssembler can exclude secrets deterministically;
- ModelRouter can deny cloud by default;
- MemoryWritePort can reject secret memories;
- event log avoids raw prompt/secret storage by default.

Negative:

- no automatic PII/secret detection in MVP;
- classification may be conservative or manually assigned;
- future integrations will require richer policy logic.

## Deferred

- LLM-based sensitivity classifier;
- full PII/secret detector;
- policy DSL;
- privacy dashboard;
- secret manager integration;
- cloud redaction pipeline;
- tool credential injection;
- fine-grained per-tool policies;
- automated retention/purge workflows.
