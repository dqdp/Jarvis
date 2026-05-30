from __future__ import annotations

from pathlib import Path

import pytest

from assistant_core.storage.database import DatabaseSafetyError
from assistant_core.storage.migrations import assert_safe_migration_database_url


pytestmark = pytest.mark.unit


def test_migration_guard_allows_local_database() -> None:
    assert_safe_migration_database_url(
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:5432/jarvis",
    )


def test_migration_guard_rejects_remote_database_by_default() -> None:
    with pytest.raises(DatabaseSafetyError, match="local database host"):
        assert_safe_migration_database_url(
            "postgresql+asyncpg://jarvis:jarvis@db.example.com:5432/jarvis",
        )


def test_events_truncate_guard_has_no_unqualified_session_guc_bypass() -> None:
    migration = Path(
        "src/assistant_core/storage/migrations/versions/0013_events_truncate_guard.py",
    ).read_text()

    assert "jarvis.allow_events_truncate" in migration
    assert "current_database()" in migration
    assert "_test" in migration
