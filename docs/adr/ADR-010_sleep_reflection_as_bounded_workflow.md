# ADR-010: Sleep/reflection as bounded workflow

Status: Accepted

## Context

Self-reflection is valuable but unsafe if implemented as free-running self-dialogue.

## Decision

Sleep/reflection will be a bounded workflow with inputs, outputs, limits and no external side effects without approval.

## Consequences

Safer and more testable. Deferred beyond Phase 1.
