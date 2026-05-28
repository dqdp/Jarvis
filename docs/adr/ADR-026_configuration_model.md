# ADR-026 — Configuration Model

## Status

Accepted.

## Context

Many Phase 1 decisions are configurable defaults, not hardcoded architecture:

- model profiles;
- provider endpoints;
- memory namespaces;
- retrieval limits;
- context window limits;
- runtime budgets;
- policy flags;
- privacy/logging flags.

The system needs a simple local-first configuration model without introducing a large config platform.

## Decision

Phase 1 uses YAML configuration plus environment overrides.

Secrets are never stored in YAML.

Config precedence:

```text
defaults < local YAML < environment variables
```

Config is loaded and strictly validated at startup.

Invalid config causes startup failure.

No hot reload in MVP.

Model profiles, memory namespace registry, runtime budgets and context/windowing policy are config-driven.

`PolicyPort` uses `ConfigPolicyEngine` in MVP.

Cloud is disabled by default in both config and policy.

Raw prompt logging is disabled by default.

Config validation tests are required.

MVP environment profiles:

```text
local
test
```

## Rationale

This keeps Phase 1 simple while preventing architectural decisions from being hardcoded.

Startup validation catches misconfiguration early.

Environment overrides allow local secrets and deployment-specific values without committing secrets to the repository.

## Consequences

Positive:

- simple local configuration;
- no secrets in committed config;
- model/profile/budget/window defaults remain flexible;
- policy decisions remain centralized;
- config decisions become testable.

Trade-offs:

- config changes require restart;
- no admin UI;
- no dynamic policy reload;
- no secret manager integration in MVP.

## Deferred

- dynamic config reload;
- admin configuration UI;
- policy DSL;
- secret manager integration;
- multi-user/tenant config;
- remote config service;
- feature flag service.
