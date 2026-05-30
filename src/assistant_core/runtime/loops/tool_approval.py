from __future__ import annotations

import asyncio
from datetime import UTC, datetime


class ApprovalWaiter:
    def __init__(self, approval_store) -> None:
        self._approval_store = approval_store

    async def wait(self, approval_id: str, *, loop_deadline: float, actor_id: str | None) -> None:
        try:
            while True:
                _raise_if_wall_time_exceeded(loop_deadline)
                await self._approval_store.expire_stale(now=datetime.now(UTC))
                approval = await self._approval_store.get_approval(approval_id)
                if approval is None:
                    raise RuntimeError("approval_not_found")
                if approval.status.value == "granted":
                    return
                if approval.status.value != "pending":
                    raise RuntimeError(f"approval_{approval.status.value}")
                await asyncio.sleep(
                    min(0.05, max(0.001, loop_deadline - asyncio.get_running_loop().time())),
                )
        except asyncio.CancelledError:
            approval = await self._approval_store.get_approval(approval_id)
            if approval is not None and approval.status.value == "pending":
                await self._approval_store.cancel_approval(
                    approval_id,
                    actor_id=actor_id,
                    reason="request cancelled",
                )
            raise


def _raise_if_wall_time_exceeded(deadline: float) -> None:
    if asyncio.get_running_loop().time() >= deadline:
        raise RuntimeError("max_wall_time_exceeded")
