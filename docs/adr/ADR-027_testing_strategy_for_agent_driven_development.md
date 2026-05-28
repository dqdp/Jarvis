# ADR-027 — Testing Strategy for Agent-Driven Development

## Status

Accepted.

## Context

The system will be implemented by coding agents and must follow TDD.

Architecture decisions such as ports/adapters, local-first policy, event log, context assembly, memory boundaries and ModelRouter policy checks are vulnerable to implementation shortcuts.

Tests must serve as executable architectural policy.

## Decision

Phase 1 development is TDD-first.

Required test layers:

```text
unit
contract
integration
golden
architecture
e2e
```

Contract tests are mandatory for replaceable ports.

Golden tests are mandatory for ContextAssembler.

Architecture tests are mandatory to enforce ports/adapters boundaries.

Fake model and embedding providers are mandatory.

Real LLM calls are not required for CI.

The E2E user-turn lifecycle test is mandatory.

Config validation tests are mandatory.

Testing strategy must be mapped to implementation slices, but the detailed TDD implementation slicing plan is a separate follow-up discussion.

No tool/MCP/RAG/ReAct tests are required in MVP.

## Rationale

TDD makes implementation by coding agents safer and more deterministic.

Contract tests preserve replaceability.

Architecture tests prevent boundary erosion.

Fake providers make tests deterministic and avoid dependency on real model quality or availability.

Separating TDD slice planning from testing strategy avoids mixing policy with delivery planning.

## Consequences

Positive:

- architecture decisions become executable;
- coding agents receive clear constraints;
- MVP behavior is testable without real LLM calls;
- backend replacement remains possible.

Trade-offs:

- additional upfront test work;
- architecture tests require maintenance;
- some behavior may need fake providers and fixtures before production code exists.

## Deferred

- detailed TDD implementation slices plan;
- performance/load testing;
- real provider live tests;
- tool/MCP/RAG/ReAct tests;
- multi-user/auth tests.
