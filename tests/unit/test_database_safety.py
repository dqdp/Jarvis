from __future__ import annotations

import pytest

from assistant_core.storage.database import DatabaseSafetyError, assert_test_database_url


pytestmark = pytest.mark.unit


def test_allows_explicit_test_database_url() -> None:
    assert_test_database_url(
        "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:55432/jarvis_test",
    )


def test_rejects_non_test_database_name_for_destructive_helpers() -> None:
    with pytest.raises(DatabaseSafetyError):
        assert_test_database_url(
            "postgresql+asyncpg://jarvis:jarvis@127.0.0.1:5432/jarvis",
        )


def test_rejects_unexpected_host_for_destructive_helpers() -> None:
    with pytest.raises(DatabaseSafetyError):
        assert_test_database_url(
            "postgresql+asyncpg://jarvis:jarvis@db.internal:55432/jarvis_test",
        )
