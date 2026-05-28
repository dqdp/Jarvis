# ADR-013 — Memory namespace model

## Status

Accepted.

## Context

The assistant needs namespaces to separate user preferences, working style, project memory, runtime rules and environment/inference facts. A single default namespace would mix unrelated memory. A deep namespace hierarchy would make MVP brittle.

## Decision

Phase 1 uses a minimal explicit namespace registry:

```text
user.preferences
user.working_style
project.personal_assistant
system.runtime_rules
environment.inference_node
```

Namespaces are not auto-created by LLM in Phase 1.

Namespace is not used as a deep taxonomy. Phase, component, ADR and tags are stored in metadata.

## Consequences

- Retrieval is namespace-aware.
- MemoryWritePort validates namespace/type compatibility.
- New namespaces require documentation update or ADR.
- The model avoids namespace explosion during MVP.
