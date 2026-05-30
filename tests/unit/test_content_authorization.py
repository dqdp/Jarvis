from __future__ import annotations

import asyncio

import pytest

from assistant_core.api.content_authorization import authorize_content_operation
from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.policy import Capability, PolicyDecision, RiskClass


pytestmark = pytest.mark.unit


class RecordingPolicy:
    def __init__(self) -> None:
        self.risk_classes: list[frozenset[RiskClass]] = []

    async def evaluate_capability_request(self, request):
        self.risk_classes.append(request.risk_classes)
        return PolicyDecision(allowed=True, code="allowed", reason="test")


def _settings():
    return ConfigLoader("config").load("test")


def test_content_ingest_is_authorized_as_local_write() -> None:
    async def scenario() -> RecordingPolicy:
        policy = RecordingPolicy()

        denied = await authorize_content_operation(
            policy,
            settings=_settings(),
            capability=Capability.CONTENT_INGEST,
            operation="project_docs_ingest",
        )

        assert denied is None
        return policy

    policy = asyncio.run(scenario())

    assert policy.risk_classes == [frozenset({RiskClass.WRITES_LOCAL})]


def test_content_retrieve_is_authorized_as_read_only() -> None:
    async def scenario() -> RecordingPolicy:
        policy = RecordingPolicy()

        denied = await authorize_content_operation(
            policy,
            settings=_settings(),
            capability=Capability.CONTENT_RETRIEVE,
            operation="content_status",
        )

        assert denied is None
        return policy

    policy = asyncio.run(scenario())

    assert policy.risk_classes == [frozenset({RiskClass.READ_ONLY})]
