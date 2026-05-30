from __future__ import annotations

from fastapi.responses import JSONResponse

from assistant_core.api.errors import error_response
from assistant_core.config.settings import Settings
from assistant_core.domain.policy import Capability, CapabilityPolicyRequest, RiskClass
from assistant_core.domain.sensitivity import Sensitivity


async def authorize_content_operation(
    policy,
    *,
    settings: Settings,
    capability: Capability,
    operation: str,
) -> JSONResponse | None:
    if policy is None:
        return error_response(
            503,
            "policy_not_configured",
            "policy engine is required for content operations",
        )
    decision = await policy.evaluate_capability_request(
        CapabilityPolicyRequest(
            capability=capability,
            risk_classes=_content_operation_risk_classes(capability),
            sensitivity=Sensitivity.PROJECT,
            permission_mode=settings.permissions.mode,
            user_id=settings.app.default_user_id,
            project_namespace="project.personal_assistant",
            redacted_payload={"operation": operation},
        ),
    )
    if decision.allowed:
        return None
    return error_response(
        403,
        decision.code,
        decision.reason,
        details={
            "capability": capability.value,
            "outcome": str(
                decision.outcome.value if hasattr(decision.outcome, "value") else decision.outcome,
            ),
        },
    )


def _content_operation_risk_classes(capability: Capability) -> frozenset[RiskClass]:
    if capability in {Capability.CONTENT_INGEST, Capability.CONTENT_INDEX}:
        return frozenset({RiskClass.WRITES_LOCAL})
    return frozenset({RiskClass.READ_ONLY})
