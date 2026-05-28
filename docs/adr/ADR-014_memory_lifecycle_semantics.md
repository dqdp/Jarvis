# ADR-014 — Memory lifecycle semantics

## Status

Accepted.

## Context

Long-term memory can become outdated, incorrect, duplicated or replaced by newer decisions. The system needs lifecycle semantics without losing provenance.

## Decision

Memory records use three lifecycle statuses:

```text
active
archived
superseded
```

Default retrieval includes only active memories.

`update` is used for same-identity corrections or metadata changes.

`archive` removes a memory from active retrieval without direct replacement.

`supersede` creates or links a newer memory that semantically replaces older memory.

Hard delete is not a normal Phase 1 operation and is reserved for future explicit privacy/purge semantics.

All lifecycle changes emit event log records.

## Consequences

- Memory can evolve without corrupting historical truth.
- Event log remains the immutable record of what happened.
- Retrieval avoids archived/superseded noise by default.
- Later sleep/reflection workflows can use the same lifecycle model.
