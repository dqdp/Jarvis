from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0012_waiting_approval_status"
down_revision = "0011_content_retrieval_failed"
branch_labels = None
depends_on = None


REQUEST_STATUSES = (
    "accepted",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
)
PREVIOUS_REQUEST_STATUSES = (
    "accepted",
    "running",
    "completed",
    "failed",
    "cancelled",
)


def upgrade() -> None:
    _replace_status_check(REQUEST_STATUSES)


def downgrade() -> None:
    _replace_status_check(PREVIOUS_REQUEST_STATUSES)


def _replace_status_check(values: tuple[str, ...]) -> None:
    op.drop_constraint(
        "assistant_requests_status_check",
        "assistant_requests",
        type_="check",
    )
    quoted = ", ".join(f"'{value}'" for value in values)
    op.create_check_constraint(
        "assistant_requests_status_check",
        "assistant_requests",
        sa.text(f"status in ({quoted})"),
    )
