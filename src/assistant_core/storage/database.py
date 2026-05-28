from __future__ import annotations

from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class DatabaseSafetyError(ValueError):
    """Raised when a destructive test helper points at a non-test database."""


def create_database_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def assert_test_database_url(database_url: str) -> None:
    parsed = urlparse(database_url)
    database_name = parsed.path.rsplit("/", 1)[-1]
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise DatabaseSafetyError("destructive test helpers require a local database host")
    if database_name != "jarvis_test" and not database_name.endswith("_test"):
        raise DatabaseSafetyError("destructive test helpers require an explicit test database")
