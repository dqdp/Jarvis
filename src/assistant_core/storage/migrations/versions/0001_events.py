from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql
import sqlalchemy as sa


revision = "0001_events"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("event_seq", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("event_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("causation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("parent_event_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text(), nullable=True),
        sa.Column("source_component", sa.Text(), nullable=False),
        sa.Column("source_node", sa.Text(), nullable=True),
        sa.Column("sensitivity", sa.Text(), nullable=False, server_default="personal"),
        sa.Column("visibility", sa.Text(), nullable=False, server_default="internal"),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("events_conversation_seq_idx", "events", ["conversation_id", "event_seq"])
    op.create_index("events_request_seq_idx", "events", ["request_id", "event_seq"])
    op.create_index("events_correlation_seq_idx", "events", ["correlation_id", "event_seq"])
    op.create_index("events_type_idx", "events", ["event_type"])
    op.create_index("events_causation_idx", "events", ["causation_id"])


def downgrade() -> None:
    op.drop_index("events_causation_idx", table_name="events")
    op.drop_index("events_type_idx", table_name="events")
    op.drop_index("events_correlation_seq_idx", table_name="events")
    op.drop_index("events_request_seq_idx", table_name="events")
    op.drop_index("events_conversation_seq_idx", table_name="events")
    op.drop_table("events")
