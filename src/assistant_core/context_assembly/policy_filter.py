from __future__ import annotations

from assistant_core.domain.policy import PolicyDecision
from assistant_core.domain.sensitivity import Sensitivity


def dropped_reason(sensitivity: Sensitivity, decision: PolicyDecision) -> str:
    if sensitivity == Sensitivity.SECRET and decision.code == "sensitivity_denied":
        return "secret"
    return decision.code
