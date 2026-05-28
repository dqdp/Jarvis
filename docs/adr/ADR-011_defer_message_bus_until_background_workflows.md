# ADR-011: Defer message bus until background workflows

Status: Accepted

## Context

Phase 1 request path is synchronous; queue/event bus would add complexity.

## Decision

Do not require Redis/NATS in Phase 1. Add future SchedulerPort/EventPublisherPort. Prefer NATS JetStream for Phase 2 if needed.

## Consequences

Faster MVP. Background jobs are deferred or in-process only.
