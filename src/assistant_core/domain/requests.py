from __future__ import annotations

from enum import StrEnum


class RequestStatus(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ALLOWED_REQUEST_STATUS_TRANSITIONS = {
    (RequestStatus.ACCEPTED, RequestStatus.RUNNING),
    (RequestStatus.ACCEPTED, RequestStatus.FAILED),
    (RequestStatus.ACCEPTED, RequestStatus.CANCELLED),
    (RequestStatus.RUNNING, RequestStatus.WAITING_APPROVAL),
    (RequestStatus.RUNNING, RequestStatus.COMPLETED),
    (RequestStatus.RUNNING, RequestStatus.FAILED),
    (RequestStatus.RUNNING, RequestStatus.CANCELLED),
    (RequestStatus.WAITING_APPROVAL, RequestStatus.RUNNING),
    (RequestStatus.WAITING_APPROVAL, RequestStatus.FAILED),
    (RequestStatus.WAITING_APPROVAL, RequestStatus.CANCELLED),
}


def is_request_status_transition_allowed(
    current: RequestStatus,
    next_status: RequestStatus,
) -> bool:
    return (current, next_status) in ALLOWED_REQUEST_STATUS_TRANSITIONS
