# ADR-004: Memory subsystem is accessed only through Memory ports

Status: Accepted

## Context

Memory must be replaceable and must not be tied to database schema or retrieval technology.

## Decision

Use MemoryReadPort and MemoryWritePort from Phase 1. Future MemoryCandidatePort and MemoryConsolidationPort are documented.

## Consequences

Runtime can later use a different memory backend. Some up-front interface design required.
