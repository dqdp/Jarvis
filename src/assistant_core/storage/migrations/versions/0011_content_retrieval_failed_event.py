from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0011_content_retrieval_failed"
down_revision = "0010_storage_hardening"
branch_labels = None
depends_on = None


EVENT_TYPES = (
    "user.message.created",
    "assistant.message.created",
    "request.processing.started",
    "request.processing.completed",
    "request.processing.failed",
    "request.processing.cancelled",
    "agent.loop.started",
    "agent.loop.completed",
    "agent.loop.failed",
    "agent.loop.cancelled",
    "agent.step.started",
    "agent.step.completed",
    "agent.step.failed",
    "context.assembly.started",
    "context.assembled",
    "context.assembly.failed",
    "context.assembly.truncated",
    "memory.retrieved",
    "memory.retrieval.failed",
    "memory.embedding.created",
    "memory.embedding.failed",
    "memory.created",
    "memory.updated",
    "memory.archived",
    "memory.superseded",
    "model.request.created",
    "model.response.received",
    "model.request.failed",
    "model.request.denied",
    "policy.decision.recorded",
    "policy.capability.decision.recorded",
    "tool.call.requested",
    "tool.call.approved",
    "tool.call.denied",
    "tool.call.started",
    "tool.call.completed",
    "tool.call.failed",
    "tool.call.timeout",
    "tool.call.cancelled",
    "tool.shell.classified",
    "tool.shell.denied",
    "tool.shell.started",
    "tool.shell.completed",
    "tool.shell.failed",
    "tool.shell.timeout",
    "tool.shell.output_truncated",
    "tool.system.diagnostics.classified",
    "tool.system.diagnostics.denied",
    "tool.system.diagnostics.started",
    "tool.system.diagnostics.completed",
    "tool.system.diagnostics.failed",
    "tool.system.diagnostics.timeout",
    "tool.system.diagnostics.output_truncated",
    "tool.system.diagnostics.unavailable",
    "content.source.discovered",
    "content.source.ingested",
    "content.source.updated",
    "content.source.deleted",
    "content.chunk.created",
    "content.chunk.stale",
    "content.embedding.created",
    "content.embedding.failed",
    "content.retrieved",
    "content.retrieval.failed",
    "tool.observation.recorded",
    "approval.required",
    "approval.granted",
    "approval.denied",
    "approval.expired",
    "approval.cancelled",
    "runtime.error",
)


def upgrade() -> None:
    op.drop_constraint("events_event_type_check", "events", type_="check")
    op.create_check_constraint(
        "events_event_type_check",
        "events",
        sa.text(f"event_type in {_sql_values(EVENT_TYPES)}"),
    )


def downgrade() -> None:
    previous = tuple(value for value in EVENT_TYPES if value != "content.retrieval.failed")
    op.drop_constraint("events_event_type_check", "events", type_="check")
    op.create_check_constraint(
        "events_event_type_check",
        "events",
        sa.text(f"event_type in {_sql_values(previous)}"),
    )


def _sql_values(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"
