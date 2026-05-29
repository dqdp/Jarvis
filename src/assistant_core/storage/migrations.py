from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from alembic import command
from alembic.config import Config

from assistant_core.storage.database import DatabaseSafetyError


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def assert_safe_migration_database_url(database_url: str) -> None:
    parsed = urlparse(database_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise DatabaseSafetyError("migrations require a local database host by default")


def run_migrations(database_url: str, *, allow_remote: bool = False) -> None:
    if not allow_remote:
        assert_safe_migration_database_url(database_url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(PROJECT_ROOT / "src" / "assistant_core" / "storage" / "migrations"),
    )
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


def main() -> None:
    run_migrations(
        os.environ["DATABASE_URL"],
        allow_remote=_env_bool(os.environ.get("JARVIS_ALLOW_REMOTE_MIGRATIONS")),
    )


def _env_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
