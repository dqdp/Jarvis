# ADR-028 — Provisional TDD Implementation Slices Plan

## Status

Accepted as provisional baseline.

## Context

The project will be implemented by coding agents using TDD. A high-level slice plan is needed to guide implementation order, but the exact order may need adjustment after repository/dependency analysis.

The plan must not override accepted architecture decisions.

## Decision

`27_tdd_implementation_slices_plan.md` is accepted as the initial provisional implementation plan for Phase 1.

The plan may be revised after coding-agent analysis.

Implementation-order-only changes may update the slice plan document.

Architecture-changing revisions require ADR update.

Coding agents proposing changes must include:

```text
proposed slice plan change
reason
affected docs/ADR
risk
changed tests
whether architecture decision changes
```

Non-negotiable constraints:

```text
TDD-first
no ports/adapters bypass
no tools/RAG/ReAct in MVP
no cloud fallback
contract tests remain mandatory
architecture tests remain mandatory
```

## Rationale

The slice plan gives coding agents a safe starting path while preserving flexibility.

Real implementation may reveal practical ordering issues.

Separating implementation-order changes from architecture changes avoids unnecessary ADR churn while preserving governance.

## Consequences

Positive:

- agents have a concrete implementation path;
- TDD remains central;
- architecture decisions remain protected;
- practical ordering changes are allowed.

Trade-offs:

- slice plan requires maintenance;
- agents must classify changes carefully;
- some implementation work may cause document updates.

## Deferred

- final implementation plan after repository analysis;
- detailed agent task prompts per slice;
- CI pipeline definition;
- release checklist.
