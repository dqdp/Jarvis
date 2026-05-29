from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision = "0009_content_embeddings"
down_revision = "0008_content_retrieval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "content_embeddings",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding_profile", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(["chunk_id"], ["content_chunks.chunk_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", "embedding_profile"),
    )
    op.create_index("content_embeddings_content_hash_idx", "content_embeddings", ["content_hash"])
    op.create_index("content_embeddings_status_idx", "content_embeddings", ["status"])


def downgrade() -> None:
    op.drop_index("content_embeddings_status_idx", table_name="content_embeddings")
    op.drop_index("content_embeddings_content_hash_idx", table_name="content_embeddings")
    op.drop_table("content_embeddings")
