from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0010_storage_hardening"
down_revision = "0009_content_embeddings"
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
ACTOR_TYPES = ("user", "assistant", "system", "model", "tool", "scheduler")
EVENT_VISIBILITIES = ("internal", "user_visible", "debug")
SENSITIVITIES = ("public", "project", "personal", "infra", "secret")
NON_SECRET_SENSITIVITIES = ("public", "project", "personal", "infra")
CONVERSATION_STATUSES = ("active", "archived")
MESSAGE_ROLES = ("user", "assistant", "system", "tool", "developer")
REQUEST_STATUSES = (
    "accepted",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
)
MODEL_INVOCATION_STATUSES = ("running", "completed", "failed", "cancelled")
MEMORY_TYPES = ("fact", "preference", "procedure", "summary")
MEMORY_STATUSES = ("active", "archived", "superseded")
MEMORY_INDEXING_STATUSES = ("indexed", "embedding_pending", "embedding_failed")
MEMORY_CANDIDATE_STATUSES = ("pending", "approved", "rejected", "merged", "expired")
APPROVAL_STATUSES = ("pending", "granted", "denied", "expired", "cancelled")
CONTENT_SOURCE_STATUSES = ("active", "stale", "deleted", "failed")
CONTENT_CHUNK_STATUSES = ("active", "stale", "deleted")
CONTENT_EMBEDDING_STATUSES = ("indexed", "failed")


def upgrade() -> None:
    _create_in_check("events_event_type_check", "events", "event_type", EVENT_TYPES)
    _create_in_check("events_actor_type_check", "events", "actor_type", ACTOR_TYPES)
    _create_in_check("events_sensitivity_check", "events", "sensitivity", SENSITIVITIES)
    _create_in_check("events_visibility_check", "events", "visibility", EVENT_VISIBILITIES)
    op.create_check_constraint(
        "events_event_version_check",
        "events",
        sa.text("event_version > 0"),
    )

    _create_in_check(
        "conversations_status_check",
        "conversations",
        "status",
        CONVERSATION_STATUSES,
    )
    _create_in_check("messages_role_check", "messages", "role", MESSAGE_ROLES)
    _create_in_check("messages_sensitivity_check", "messages", "sensitivity", SENSITIVITIES)
    _create_in_check(
        "assistant_requests_status_check",
        "assistant_requests",
        "status",
        REQUEST_STATUSES,
    )

    _create_in_check(
        "model_invocations_status_check",
        "model_invocations",
        "status",
        MODEL_INVOCATION_STATUSES,
    )
    _create_in_check(
        "model_invocations_sensitivity_check",
        "model_invocations",
        "sensitivity",
        SENSITIVITIES,
    )
    _create_in_check(
        "model_invocations_sensitivity_no_secret_check",
        "model_invocations",
        "sensitivity",
        NON_SECRET_SENSITIVITIES,
    )

    _create_in_check("memories_memory_type_check", "memories", "memory_type", MEMORY_TYPES)
    _create_in_check("memories_status_check", "memories", "status", MEMORY_STATUSES)
    _create_in_check(
        "memories_indexing_status_check",
        "memories",
        "indexing_status",
        MEMORY_INDEXING_STATUSES,
    )
    _create_in_check("memories_sensitivity_check", "memories", "sensitivity", SENSITIVITIES)
    _create_in_check(
        "memories_sensitivity_no_secret_check",
        "memories",
        "sensitivity",
        NON_SECRET_SENSITIVITIES,
    )
    _create_in_check(
        "memory_candidates_status_check",
        "memory_candidates",
        "status",
        MEMORY_CANDIDATE_STATUSES,
    )
    _create_in_check(
        "memory_candidates_sensitivity_check",
        "memory_candidates",
        "sensitivity",
        SENSITIVITIES,
    )
    _create_in_check(
        "memory_candidates_sensitivity_no_secret_check",
        "memory_candidates",
        "sensitivity",
        NON_SECRET_SENSITIVITIES,
    )

    _create_in_check("approvals_status_check", "approvals", "status", APPROVAL_STATUSES)
    op.create_check_constraint(
        "approvals_risk_classes_is_array_check",
        "approvals",
        sa.text("jsonb_typeof(risk_classes) = 'array'"),
    )

    _create_in_check(
        "content_sources_status_check",
        "content_sources",
        "status",
        CONTENT_SOURCE_STATUSES,
    )
    _create_in_check(
        "content_sources_sensitivity_check",
        "content_sources",
        "sensitivity",
        SENSITIVITIES,
    )
    _create_in_check(
        "content_sources_sensitivity_no_secret_check",
        "content_sources",
        "sensitivity",
        NON_SECRET_SENSITIVITIES,
    )
    _create_in_check(
        "content_chunks_status_check",
        "content_chunks",
        "status",
        CONTENT_CHUNK_STATUSES,
    )
    _create_in_check(
        "content_chunks_sensitivity_check",
        "content_chunks",
        "sensitivity",
        SENSITIVITIES,
    )
    _create_in_check(
        "content_chunks_sensitivity_no_secret_check",
        "content_chunks",
        "sensitivity",
        NON_SECRET_SENSITIVITIES,
    )
    _create_in_check(
        "content_embeddings_status_check",
        "content_embeddings",
        "status",
        CONTENT_EMBEDDING_STATUSES,
    )

    op.execute(
        """
        create or replace function events_append_only_guard()
        returns trigger
        language plpgsql
        as $$
        begin
            if TG_OP = 'TRUNCATE'
               and current_setting('jarvis.allow_events_truncate', true) = 'on'
               and right(lower(current_database()), 5) = '_test' then
                return null;
            end if;
            raise exception 'events table is append-only';
        end;
        $$;
        """,
    )
    op.execute(
        """
        create trigger events_append_only_trigger
        before update or delete on events
        for each row execute function events_append_only_guard();
        """,
    )
    op.execute(
        """
        create trigger events_append_only_truncate_trigger
        before truncate on events
        for each statement execute function events_append_only_guard();
        """,
    )


def downgrade() -> None:
    op.execute("drop trigger if exists events_append_only_truncate_trigger on events")
    op.execute("drop trigger if exists events_append_only_trigger on events")
    op.execute("drop function if exists events_append_only_guard()")

    for table_name, constraint_name in reversed(
        (
            ("content_embeddings", "content_embeddings_status_check"),
            ("content_chunks", "content_chunks_sensitivity_no_secret_check"),
            ("content_chunks", "content_chunks_sensitivity_check"),
            ("content_chunks", "content_chunks_status_check"),
            ("content_sources", "content_sources_sensitivity_no_secret_check"),
            ("content_sources", "content_sources_sensitivity_check"),
            ("content_sources", "content_sources_status_check"),
            ("approvals", "approvals_risk_classes_is_array_check"),
            ("approvals", "approvals_status_check"),
            ("memory_candidates", "memory_candidates_sensitivity_no_secret_check"),
            ("memory_candidates", "memory_candidates_sensitivity_check"),
            ("memory_candidates", "memory_candidates_status_check"),
            ("memories", "memories_sensitivity_no_secret_check"),
            ("memories", "memories_sensitivity_check"),
            ("memories", "memories_indexing_status_check"),
            ("memories", "memories_status_check"),
            ("memories", "memories_memory_type_check"),
            ("model_invocations", "model_invocations_sensitivity_no_secret_check"),
            ("model_invocations", "model_invocations_sensitivity_check"),
            ("model_invocations", "model_invocations_status_check"),
            ("assistant_requests", "assistant_requests_status_check"),
            ("messages", "messages_sensitivity_check"),
            ("messages", "messages_role_check"),
            ("conversations", "conversations_status_check"),
            ("events", "events_event_version_check"),
            ("events", "events_visibility_check"),
            ("events", "events_sensitivity_check"),
            ("events", "events_actor_type_check"),
            ("events", "events_event_type_check"),
        ),
    ):
        op.drop_constraint(constraint_name, table_name, type_="check")


def _create_in_check(
    constraint_name: str,
    table_name: str,
    column_name: str,
    allowed_values: tuple[str, ...],
) -> None:
    op.create_check_constraint(
        constraint_name,
        table_name,
        sa.text(f"{column_name} in {_sql_values(allowed_values)}"),
    )


def _sql_values(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"
