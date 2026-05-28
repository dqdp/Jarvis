# ADR-001: Phase 1 uses modular monolith with ports/adapters boundaries

Status: Accepted

## Context

Phase 1 must be fast to implement but must avoid throwaway architecture and subsystem lock-in.

## Decision

Use a modular monolith. All major subsystems are accessed through explicit ports. Concrete implementations are adapters.

## Consequences

Faster MVP than microservices; still supports later extraction. Requires discipline and contract tests.
