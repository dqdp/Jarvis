from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision = "0008_content_retrieval"
down_revision = "0007_approval_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "content_sources",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("sensitivity", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("source_id"),
        sa.UniqueConstraint("path"),
    )
    op.create_index("content_sources_status_idx", "content_sources", ["status"])
    op.create_index("content_sources_content_hash_idx", "content_sources", ["content_hash"])

    op.create_table(
        "content_chunks",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("heading_path", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=False),
        sa.Column("line_end", sa.Integer(), nullable=False),
        sa.Column("citation", sa.Text(), nullable=False),
        sa.Column("sensitivity", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(["source_id"], ["content_sources.source_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("content_chunks_source_status_idx", "content_chunks", ["source_id", "status"])
    op.create_index("content_chunks_content_hash_idx", "content_chunks", ["content_hash"])
    op.create_index("content_chunks_source_ordinal_idx", "content_chunks", ["source_id", "ordinal"])


def downgrade() -> None:
    op.drop_index("content_chunks_source_ordinal_idx", table_name="content_chunks")
    op.drop_index("content_chunks_content_hash_idx", table_name="content_chunks")
    op.drop_index("content_chunks_source_status_idx", table_name="content_chunks")
    op.drop_table("content_chunks")
    op.drop_index("content_sources_content_hash_idx", table_name="content_sources")
    op.drop_index("content_sources_status_idx", table_name="content_sources")
    op.drop_table("content_sources")
