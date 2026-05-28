# ADR-005: Replaceable subsystems require contract tests

Status: Accepted

## Context

Ports are only useful if adapters can be verified against stable behavior.

## Decision

Define contract tests for Memory, EventLog, ConversationStore, ModelRouter and future ports.

## Consequences

Adapter replacement becomes safer. Adds test burden early.
