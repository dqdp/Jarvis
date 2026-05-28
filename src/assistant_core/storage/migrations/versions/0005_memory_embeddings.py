from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision = "0005_memory_embeddings"
down_revision = "0004_memory_write"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_embeddings",
        sa.Column(
            "memory_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memories.memory_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding_profile", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("memory_id", "embedding_profile"),
    )
    op.create_index("memory_embeddings_content_hash_idx", "memory_embeddings", ["content_hash"])


def downgrade() -> None:
    op.drop_index("memory_embeddings_content_hash_idx", table_name="memory_embeddings")
    op.drop_table("memory_embeddings")
