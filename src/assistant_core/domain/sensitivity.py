from __future__ import annotations

from enum import StrEnum


class Sensitivity(StrEnum):
    PUBLIC = "public"
    PROJECT = "project"
    PERSONAL = "personal"
    INFRA = "infra"
    SECRET = "secret"
