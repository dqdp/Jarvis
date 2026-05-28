# ADR-002: PostgreSQL is the primary system of record

Status: Accepted

## Context

Core daemon needs durable storage for conversations, messages, events, memory, model invocations and runtime state.

## Decision

Use PostgreSQL as primary system of record for Phase 1.

## Consequences

Simple local deployment, strong consistency and auditability. Must avoid leaking DB details into runtime.
