from __future__ import annotations

from alembic import op


revision = "0013_events_truncate_guard"
down_revision = "0012_waiting_approval_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
        do $$
        begin
            if not exists (
                select 1
                from pg_trigger
                where tgname = 'events_append_only_truncate_trigger'
            ) then
                create trigger events_append_only_truncate_trigger
                before truncate on events
                for each statement execute function events_append_only_guard();
            end if;
        end;
        $$;
        """,
    )


def downgrade() -> None:
    op.execute("drop trigger if exists events_append_only_truncate_trigger on events")
    op.execute(
        """
        create or replace function events_append_only_guard()
        returns trigger
        language plpgsql
        as $$
        begin
            raise exception 'events table is append-only';
        end;
        $$;
        """,
    )
