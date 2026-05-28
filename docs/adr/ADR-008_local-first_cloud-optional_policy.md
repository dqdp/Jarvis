# ADR-008: Local-first cloud-optional policy

Status: Accepted

## Context

Assistant operates in private home loop. Cloud LLM must not receive private data accidentally.

## Decision

Cloud profiles disabled by default. Minimal PolicyPort enforces local-first policy.

## Consequences

Safer default. External LLM integration requires explicit config/policy.
