from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision = "0004_memory_write"
down_revision = "0003_model_invocations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid_array_default = sa.text("'{}'::uuid[]")

    op.create_table(
        "memories",
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("sensitivity", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("indexing_status", sa.Text(), nullable=False, server_default="embedding_pending"),
        sa.Column("source_event_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default=uuid_array_default),
        sa.Column("supersedes_memory_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default=uuid_array_default),
        sa.Column("superseded_by_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archive_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("memories_namespace_idx", "memories", ["namespace"])
    op.create_index("memories_type_idx", "memories", ["memory_type"])
    op.create_index("memories_status_idx", "memories", ["status"])
    op.create_index("memories_namespace_type_status_idx", "memories", ["namespace", "memory_type", "status"])
    op.create_index(
        "memories_retrieval_filter_idx",
        "memories",
        ["namespace", "memory_type", "status", "sensitivity", "indexing_status"],
    )

    op.create_table(
        "memory_candidates",
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("proposed_namespace", sa.Text(), nullable=False),
        sa.Column("proposed_memory_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sensitivity", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source_event_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default=uuid_array_default),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_reason", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("memory_candidates_status_idx", "memory_candidates", ["status"])


def downgrade() -> None:
    op.drop_index("memory_candidates_status_idx", table_name="memory_candidates")
    op.drop_table("memory_candidates")
    op.drop_index("memories_retrieval_filter_idx", table_name="memories")
    op.drop_index("memories_namespace_type_status_idx", table_name="memories")
    op.drop_index("memories_status_idx", table_name="memories")
    op.drop_index("memories_type_idx", table_name="memories")
    op.drop_index("memories_namespace_idx", table_name="memories")
    op.drop_table("memories")
