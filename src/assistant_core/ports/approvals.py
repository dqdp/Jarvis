from __future__ import annotations

from datetime import datetime
from typing import Protocol

from assistant_core.domain.approvals import (
    ApprovalRequest,
    ApprovalScope,
    CreateApprovalCommand,
)


class ApprovalStorePort(Protocol):
    async def create_approval(self, command: CreateApprovalCommand) -> ApprovalRequest: ...

    async def get_approval(self, approval_id: str) -> ApprovalRequest | None: ...

    async def grant_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest: ...

    async def deny_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest: ...

    async def cancel_approval(
        self,
        approval_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> ApprovalRequest: ...

    async def cancel_pending_for_request(
        self,
        request_id: str,
        *,
        actor_id: str | None,
        reason: str | None = None,
    ) -> list[ApprovalRequest]: ...

    async def expire_stale(self, *, now: datetime | None = None) -> list[ApprovalRequest]: ...

    async def consume_granted_approval(
        self,
        approval_id: str,
        *,
        scope: ApprovalScope,
    ) -> ApprovalRequest: ...
