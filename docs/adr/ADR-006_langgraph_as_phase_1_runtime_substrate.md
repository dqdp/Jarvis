# ADR-006: LangGraph as Phase 1 runtime substrate

Status: Accepted

## Context

Need graph execution, state and checkpointing without writing custom graph runtime.

## Decision

Use LangGraph as execution substrate. Store checkpoints in PostgreSQL for Phase 1.

## Consequences

Good fit for stateful workflows. LangGraph does not define our agent architecture.
