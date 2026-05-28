from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision = "0003_model_invocations"
down_revision = "0002_conversation_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_invocations",
        sa.Column("model_invocation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("sensitivity", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("input_token_estimate", sa.Integer(), nullable=True),
        sa.Column("input_tokens_reported", sa.Integer(), nullable=True),
        sa.Column("output_tokens_reported", sa.Integer(), nullable=True),
        sa.Column("streaming", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("context_manifest_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("model_invocations_request_idx", "model_invocations", ["request_id"])
    op.create_index("model_invocations_profile_started_idx", "model_invocations", ["profile", "started_at"])


def downgrade() -> None:
    op.drop_index("model_invocations_profile_started_idx", table_name="model_invocations")
    op.drop_index("model_invocations_request_idx", table_name="model_invocations")
    op.drop_table("model_invocations")
