from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_conversation_integrity"
down_revision = "0005_memory_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    duplicate_user_messages = connection.scalar(
        sa.text(
            "select count(*) from ("
            "select user_message_id from assistant_requests "
            "group by user_message_id having count(*) > 1"
            ") duplicates",
        ),
    )
    if duplicate_user_messages:
        raise RuntimeError(
            "cannot apply 0006_conversation_integrity: duplicate assistant_requests.user_message_id",
        )

    orphan_message_requests = connection.scalar(
        sa.text(
            "select count(*) from messages m "
            "left join assistant_requests r on r.request_id = m.request_id "
            "where m.request_id is not null and r.request_id is null",
        ),
    )
    if orphan_message_requests:
        raise RuntimeError(
            "cannot apply 0006_conversation_integrity: messages.request_id has no assistant_request",
        )

    message_request_mismatches = connection.scalar(
        sa.text(
            "select count(*) from messages m "
            "join assistant_requests r on r.request_id = m.request_id "
            "where m.conversation_id <> r.conversation_id",
        ),
    )
    if message_request_mismatches:
        raise RuntimeError(
            "cannot apply 0006_conversation_integrity: messages.request_id crosses conversations",
        )

    invalid_user_messages = connection.scalar(
        sa.text(
            "select count(*) from assistant_requests r "
            "join messages m on m.message_id = r.user_message_id "
            "where m.conversation_id <> r.conversation_id or m.role <> 'user'",
        ),
    )
    if invalid_user_messages:
        raise RuntimeError(
            "cannot apply 0006_conversation_integrity: user_message_id is not a same-conversation user message",
        )

    invalid_assistant_messages = connection.scalar(
        sa.text(
            "select count(*) from assistant_requests r "
            "join messages m on m.message_id = r.assistant_message_id "
            "where r.assistant_message_id is not null "
            "and (m.conversation_id <> r.conversation_id "
            "or m.role <> 'assistant' "
            "or m.request_id is distinct from r.request_id)",
        ),
    )
    if invalid_assistant_messages:
        raise RuntimeError(
            "cannot apply 0006_conversation_integrity: assistant_message_id is not a same-request assistant message",
        )

    op.drop_index("assistant_requests_user_message_idx", table_name="assistant_requests")
    op.create_index(
        "assistant_requests_user_message_idx",
        "assistant_requests",
        ["user_message_id"],
        unique=True,
    )
    op.create_foreign_key(
        "messages_request_fk",
        "messages",
        "assistant_requests",
        ["request_id"],
        ["request_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("messages_request_fk", "messages", type_="foreignkey")
    op.drop_index("assistant_requests_user_message_idx", table_name="assistant_requests")
    op.create_index(
        "assistant_requests_user_message_idx",
        "assistant_requests",
        ["user_message_id"],
        unique=False,
    )
