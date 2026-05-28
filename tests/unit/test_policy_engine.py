from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant_core.config.settings import ConfigLoader
from assistant_core.domain.policy import (
    ContextPolicyRequest,
    MemoryWritePolicyRequest,
    ModelPolicyRequest,
)
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.policy.engine import ConfigPolicyEngine


pytestmark = pytest.mark.unit


def _engine() -> ConfigPolicyEngine:
    settings = ConfigLoader(Path("config")).load("test")
    return ConfigPolicyEngine(settings)


def test_local_model_project_allowed() -> None:
    decision = asyncio.run(
        _engine().evaluate_model_request(
            ModelPolicyRequest(profile="local_main", sensitivity=Sensitivity.PROJECT),
        ),
    )

    assert decision.allowed is True


def test_local_model_secret_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_model_request(
            ModelPolicyRequest(profile="local_main", sensitivity=Sensitivity.SECRET),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "sensitivity_denied"


def test_cloud_model_denied_by_default() -> None:
    decision = asyncio.run(
        _engine().evaluate_model_request(
            ModelPolicyRequest(profile="cloud_reasoning", sensitivity=Sensitivity.PUBLIC),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "cloud_models_disabled"


def test_secret_memory_write_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_memory_write(
            MemoryWritePolicyRequest(
                namespace="project.personal_assistant",
                sensitivity=Sensitivity.SECRET,
            ),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "sensitivity_denied"


def test_secret_context_inclusion_denied() -> None:
    decision = asyncio.run(
        _engine().evaluate_context_inclusion(
            ContextPolicyRequest(source_ref="memory:secret", sensitivity=Sensitivity.SECRET),
        ),
    )

    assert decision.allowed is False
    assert decision.code == "sensitivity_denied"
