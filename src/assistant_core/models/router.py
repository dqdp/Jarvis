from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import json
from typing import Any
from uuid import uuid4

from assistant_core.config.settings import ModelProfileConfig, Settings
from assistant_core.domain.events import ActorType, EventEnvelope, EventType, EventVisibility
from assistant_core.domain.models import (
    ChatModelRequest,
    ChatModelResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelStreamEvent,
    StructuredModelRequest,
    StructuredModelResponse,
)
from assistant_core.domain.policy import ModelPolicyRequest, PolicyDecision
from assistant_core.domain.sensitivity import Sensitivity
from assistant_core.ports.event_log import EventLogPort
from assistant_core.ports.model_invocations import (
    FinishModelInvocationCommand,
    ModelInvocationRepositoryPort,
    StartModelInvocationCommand,
)
from assistant_core.ports.policy import PolicyPort


class ModelRouterError(Exception):
    """Base error for model routing failures."""


class ModelPolicyDenied(ModelRouterError):
    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class ModelProviderError(ModelRouterError):
    """Raised when a provider call fails."""


class StructuredOutputValidationError(ModelRouterError):
    """Raised when structured output cannot be parsed after retries."""


class ModelRouter:
    def __init__(
        self,
        *,
        settings: Settings,
        policy: PolicyPort,
        invocation_repository: ModelInvocationRepositoryPort,
        providers: dict[str, object],
        event_log: EventLogPort,
    ) -> None:
        self._settings = settings
        self._policy = policy
        self._invocations = invocation_repository
        self._providers = providers
        self._event_log = event_log

    async def health_status(self) -> dict[str, str]:
        required_profiles = [
            name
            for name, profile in self._settings.model_profiles.items()
            if profile.enabled and not profile.cloud
        ]
        missing_profiles = [
            profile_name
            for profile_name in required_profiles
            if not self._has_provider_for_profile(profile_name)
        ]
        if missing_profiles:
            return {
                "status": "failed",
                "reason": f"missing providers: {', '.join(sorted(missing_profiles))}",
            }
        unhealthy_profiles: list[str] = []
        for profile_name in required_profiles:
            profile = self._settings.model_profiles[profile_name]
            provider = self._provider(profile.provider, profile_name=profile_name)
            if not await _provider_health(provider):
                unhealthy_profiles.append(profile_name)
        if unhealthy_profiles:
            return {
                "status": "failed",
                "reason": f"unhealthy providers: {', '.join(sorted(unhealthy_profiles))}",
            }
        return {"status": "ok"}

    async def chat(self, request: ChatModelRequest) -> ChatModelResponse:
        profile = self._profile(request.profile)
        await self._authorize(profile, request.sensitivity, request.request_id, request.conversation_id)
        provider = self._provider(profile.provider, profile_name=request.profile)
        invocation = await self._start_invocation(
            profile,
            request,
            purpose="chat",
            streaming=False,
        )
        try:
            response = await provider.chat(request)  # type: ignore[attr-defined]
        except asyncio.CancelledError as exc:
            await self._finish_cancelled(invocation.model_invocation_id, exc)
            raise
        except Exception as exc:
            await self._finish_failed(invocation.model_invocation_id, exc)
            raise

        await self._finish_completed(invocation.model_invocation_id)
        return response

    async def stream_chat(
        self,
        request: ChatModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        profile = self._profile(request.profile)
        await self._authorize(profile, request.sensitivity, request.request_id, request.conversation_id)
        provider = self._provider(profile.provider, profile_name=request.profile)
        invocation = await self._start_invocation(
            profile,
            request,
            purpose="chat",
            streaming=True,
        )
        finished = False
        try:
            async for token in provider.stream_chat(request):  # type: ignore[attr-defined]
                yield ModelStreamEvent(event_type="token", delta=token)
        except asyncio.CancelledError as exc:
            await self._finish_cancelled(invocation.model_invocation_id, exc)
            finished = True
            raise
        except Exception as exc:
            await self._finish_failed(invocation.model_invocation_id, exc)
            finished = True
            raise
        else:
            await self._finish_completed(invocation.model_invocation_id)
            finished = True
        finally:
            if not finished:
                await self._finish_cancelled(
                    invocation.model_invocation_id,
                    asyncio.CancelledError(),
                )

    async def structured(
        self,
        request: StructuredModelRequest,
    ) -> StructuredModelResponse:
        profile = self._profile(request.profile)
        await self._authorize(
            profile,
            request.sensitivity,
            request.request_id,
            request.conversation_id,
        )
        provider = self._provider(profile.provider, profile_name=request.profile)
        invocation = await self._start_invocation(
            profile,
            request,
            purpose="structured",
            streaming=False,
        )
        retry_count = int((profile.structured_output or {}).get("validation_retry", 0))

        last_error: Exception | None = None
        for attempt in range(retry_count + 1):
            try:
                raw = await provider.structured(request)  # type: ignore[attr-defined]
            except asyncio.CancelledError as exc:
                await self._finish_cancelled(invocation.model_invocation_id, exc)
                raise
            except Exception as exc:
                await self._finish_failed(invocation.model_invocation_id, exc)
                raise

            try:
                value = _parse_structured_object(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < retry_count:
                    continue
                await self._finish_failed(invocation.model_invocation_id, exc)
                raise StructuredOutputValidationError(str(exc)) from exc
            else:
                await self._finish_completed(
                    invocation.model_invocation_id,
                    metadata={"attempts": attempt + 1},
                )
                return StructuredModelResponse(value=value)

        raise StructuredOutputValidationError(str(last_error))

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        profile = self._profile(request.profile)
        await self._authorize(profile, request.sensitivity, None, None)
        provider = self._provider(profile.provider, profile_name=request.profile)
        invocation = await self._start_invocation(
            profile,
            request,
            purpose="embedding",
            streaming=False,
        )
        retry_count = int(profile.retry)

        last_error: Exception | None = None
        for attempt in range(retry_count + 1):
            try:
                response = await provider.embed(request)  # type: ignore[attr-defined]
            except asyncio.CancelledError as exc:
                await self._finish_cancelled(invocation.model_invocation_id, exc)
                raise
            except Exception as exc:
                last_error = exc
                if attempt < retry_count:
                    continue
                await self._finish_failed(invocation.model_invocation_id, exc)
                raise
            else:
                await self._finish_completed(
                    invocation.model_invocation_id,
                    metadata={"attempts": attempt + 1},
                )
                return response

        raise ModelProviderError(str(last_error))

    def _profile(self, name: str) -> ModelProfileConfig:
        profile = self._settings.model_profiles.get(name)
        if profile is None:
            raise ModelProviderError(f"unknown model profile: {name}")
        return profile

    def _provider(self, name: str, *, profile_name: str | None = None) -> Any:
        if profile_name is not None and profile_name in self._providers:
            return self._providers[profile_name]
        provider = self._providers.get(name)
        if provider is None:
            raise ModelProviderError(f"provider is not registered: {name}")
        return provider

    def _has_provider_for_profile(self, profile_name: str) -> bool:
        profile = self._settings.model_profiles[profile_name]
        return profile_name in self._providers or profile.provider in self._providers

    async def _authorize(
        self,
        profile: ModelProfileConfig,
        sensitivity,
        request_id: str | None,
        conversation_id: str | None,
    ) -> None:
        profile_name = _profile_name(self._settings, profile)
        decision = await self._policy.evaluate_model_request(
            ModelPolicyRequest(
                profile=profile_name,
                provider=profile.provider,
                cloud=profile.cloud,
                purpose=profile.purpose,
                sensitivity=sensitivity,
                request_id=request_id,
                conversation_id=conversation_id,
            ),
        )
        await self._record_policy_decision(
            profile_name=profile_name,
            decision=decision,
            sensitivity=sensitivity,
            request_id=request_id,
            conversation_id=conversation_id,
        )
        if not decision.allowed:
            await self._record_model_denied(
                profile_name=profile_name,
                decision=decision,
                sensitivity=sensitivity,
                request_id=request_id,
                conversation_id=conversation_id,
            )
            raise ModelPolicyDenied(decision)

    async def _record_policy_decision(
        self,
        *,
        profile_name: str,
        decision: PolicyDecision,
        sensitivity: Sensitivity,
        request_id: str | None,
        conversation_id: str | None,
    ) -> None:
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=EventType.POLICY_DECISION_RECORDED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=conversation_id,
                request_id=request_id,
                correlation_id=request_id,
                causation_id=None,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="model_router",
                source_node=None,
                sensitivity=sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "source_ref": f"model_request:{profile_name}",
                    "allowed": decision.allowed,
                    "code": decision.code,
                    "reason": decision.reason,
                },
                metadata={},
            ),
        )

    async def _record_model_denied(
        self,
        *,
        profile_name: str,
        decision: PolicyDecision,
        sensitivity: Sensitivity,
        request_id: str | None,
        conversation_id: str | None,
    ) -> None:
        now = datetime.now(UTC)
        await self._event_log.append(
            EventEnvelope(
                event_id=str(uuid4()),
                event_seq=0,
                event_type=EventType.MODEL_REQUEST_DENIED,
                event_version=1,
                occurred_at=now,
                recorded_at=now,
                conversation_id=conversation_id,
                request_id=request_id,
                correlation_id=request_id,
                causation_id=None,
                parent_event_id=None,
                actor_type=ActorType.SYSTEM,
                actor_id=None,
                source_component="model_router",
                source_node=None,
                sensitivity=sensitivity,
                visibility=EventVisibility.INTERNAL,
                idempotency_key=None,
                payload={
                    "profile": profile_name,
                    "code": decision.code,
                    "reason": decision.reason,
                },
                metadata={},
            ),
        )

    async def _start_invocation(
        self,
        profile: ModelProfileConfig,
        request: ChatModelRequest | StructuredModelRequest | EmbeddingRequest,
        *,
        purpose: str,
        streaming: bool,
    ):
        return await self._invocations.start(
            StartModelInvocationCommand(
                request_id=getattr(request, "request_id", None),
                conversation_id=getattr(request, "conversation_id", None),
                profile=_profile_name(self._settings, profile),
                provider=profile.provider,
                model=profile.model,
                purpose=purpose,
                sensitivity=request.sensitivity,
                streaming=streaming,
                input_token_estimate=_input_token_estimate(request),
                context_manifest_id=getattr(request, "context_manifest_id", None),
            ),
        )

    async def _finish_completed(
        self,
        model_invocation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._invocations.finish(
            FinishModelInvocationCommand(
                model_invocation_id=model_invocation_id,
                status="completed",
                metadata=metadata,
            ),
        )

    async def _finish_failed(self, model_invocation_id: str, exc: Exception) -> None:
        await self._invocations.finish(
            FinishModelInvocationCommand(
                model_invocation_id=model_invocation_id,
                status="failed",
                error_type=type(exc).__name__,
                error_message="model provider call failed",
            ),
        )

    async def _finish_cancelled(self, model_invocation_id: str, exc: BaseException) -> None:
        await self._invocations.finish(
            FinishModelInvocationCommand(
                model_invocation_id=model_invocation_id,
                status="cancelled",
                error_type=type(exc).__name__,
                error_message="model provider call cancelled",
            ),
        )


def _parse_structured_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("structured output must be a JSON object")
    return value


def _input_token_estimate(
    request: ChatModelRequest | StructuredModelRequest | EmbeddingRequest,
) -> int | None:
    if isinstance(request, EmbeddingRequest):
        return sum(len(text.split()) for text in request.texts)
    messages = getattr(request, "messages", [])
    return sum(
        len(part.text.split())
        for message in messages
        for part in message.content
        if hasattr(part, "text")
    )


def _profile_name(settings: Settings, profile: ModelProfileConfig) -> str:
    for name, candidate in settings.model_profiles.items():
        if candidate is profile:
            return name
    raise ModelProviderError("model profile is not registered in settings")


async def _provider_health(provider: object) -> bool:
    health_status = getattr(provider, "health_status", None)
    if health_status is not None:
        try:
            result = health_status()
            if hasattr(result, "__await__"):
                result = await result
        except Exception:
            return False
        if isinstance(result, dict):
            return result.get("status") in {"ok", "ready"}
        return bool(result)

    health_check = getattr(provider, "health_check", None)
    if health_check is None:
        return True
    try:
        result = health_check()
        if hasattr(result, "__await__"):
            result = await result
    except Exception:
        return False
    return bool(result)
