from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
from typing import Any
import os
from urllib.parse import urlparse

import yaml

from assistant_core.domain.policy import Capability, PermissionMode, PolicyDecisionOutcome


SENSITIVITY_VALUES = {"public", "project", "personal", "infra", "secret"}
MEMORY_TYPES = {"fact", "preference", "procedure", "summary"}
REQUIRED_MODEL_PROFILES = {
    "local_main",
    "local_structured",
    "local_embedding",
    "cloud_reasoning",
}
REQUIRED_MEMORY_NAMESPACES = {
    "user.preferences",
    "user.working_style",
    "project.personal_assistant",
    "system.runtime_rules",
    "environment.inference_node",
}


class ConfigError(ValueError):
    """Raised when startup configuration is invalid."""


@dataclass(frozen=True)
class AppConfig:
    environment: str
    instance_id: str
    default_user_id: str
    default_language: str


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    request_timeout_seconds: int
    sse_heartbeat_seconds: int


@dataclass(frozen=True)
class ModelProfileConfig:
    purpose: str
    provider: str
    enabled: bool
    cloud: bool
    model: str
    endpoint: str | None = None
    timeout_seconds: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    supports_streaming: bool = False
    dimension: int | None = None
    batch_size: int | None = None
    retry: int = 0
    api_key_env: str | None = None
    structured_output: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryNamespaceConfig:
    sensitivity: str
    allowed_types: list[str]
    default_retrieval: str | bool


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    max_hits_total: int
    max_hits_per_namespace: int
    min_score: float | None
    include_statuses: list[str]
    exclude_sensitivity: list[str]


@dataclass(frozen=True)
class MemoryConfig:
    allowed_types: set[str]
    namespaces: dict[str, MemoryNamespaceConfig]
    retrieval: MemoryRetrievalConfig


@dataclass(frozen=True)
class ContextAssemblyConfig:
    full_prompt_logging: bool
    sections: dict[str, Any]
    conversation_window: dict[str, Any]
    context_budget: dict[str, Any]


@dataclass(frozen=True)
class RuntimeBudgetConfig:
    max_model_calls: int
    max_tool_calls: int
    max_wall_time_seconds: int
    max_context_assembly_seconds: int
    max_memory_retrieval_seconds: int
    max_model_call_seconds: int
    max_output_tokens: int
    allow_cloud: bool
    allow_tools: bool
    allow_autonomous_memory_write: bool
    max_steps: int = 1
    max_consecutive_failures: int = 1


@dataclass(frozen=True)
class SensitivityDecisionConfig:
    deny_sensitivity: list[str]


@dataclass(frozen=True)
class PolicyConfig:
    cloud_models_enabled: bool
    tools_enabled: bool
    autonomous_memory_write_enabled: bool
    model_access: dict[str, Any]
    memory_write: SensitivityDecisionConfig
    context_inclusion: SensitivityDecisionConfig


@dataclass(frozen=True)
class PermissionsConfig:
    mode: PermissionMode
    modes: dict[str, dict[str, str]]


@dataclass(frozen=True)
class PrivacyConfig:
    sensitivity_order: list[str]
    raw_prompt_logging: bool
    raw_secret_logging: bool
    store_context_manifest: bool


@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: str
    structured_logs: bool
    log_raw_messages: bool
    log_raw_prompts: bool
    log_model_outputs: bool
    metrics_enabled: bool


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    api: ApiConfig
    model_profiles: dict[str, ModelProfileConfig]
    memory: MemoryConfig
    context_assembly: ContextAssemblyConfig
    runtime_budgets: dict[str, RuntimeBudgetConfig]
    policy: PolicyConfig
    permissions: PermissionsConfig
    capabilities: dict[str, Any]
    privacy: PrivacyConfig
    observability: ObservabilityConfig
    raw: dict[str, Any]


class ConfigLoader:
    def __init__(self, config_dir: str | Path, env_prefix: str = "JARVIS_") -> None:
        self._config_dir = Path(config_dir)
        self._env_prefix = env_prefix

    def load(self, profile: str = "default") -> Settings:
        data = self._load_yaml("default")
        if profile != "default":
            data = _deep_merge(data, self._load_yaml(profile))
        self._apply_env_overrides(data)
        settings = _settings_from_mapping(data, project_root=self._config_dir.resolve().parent)
        _validate(settings)
        return settings

    def _load_yaml(self, profile: str) -> dict[str, Any]:
        path = self._config_dir / f"{profile}.yaml"
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config root must be a mapping: {path}")
        return loaded

    def _apply_env_overrides(self, data: dict[str, Any]) -> None:
        for key, value in os.environ.items():
            if not key.startswith(self._env_prefix):
                continue
            path = key.removeprefix(self._env_prefix).lower().split("__")
            if not all(path):
                raise ConfigError(f"invalid environment override key: {key}")
            _set_nested(data, path, _parse_env_value(value))


def _settings_from_mapping(data: dict[str, Any], *, project_root: Path) -> Settings:
    model_profiles = {
        name: ModelProfileConfig(**profile)
        for name, profile in data["model_profiles"].items()
    }
    namespaces = {
        name: MemoryNamespaceConfig(**namespace)
        for name, namespace in data["memory"]["namespaces"].items()
    }
    runtime_budgets = {
        name: RuntimeBudgetConfig(**budget)
        for name, budget in data["runtime_budgets"].items()
    }
    return Settings(
        app=AppConfig(**data["app"]),
        api=ApiConfig(**data["api"]),
        model_profiles=model_profiles,
        memory=MemoryConfig(
            allowed_types=set(data["memory"]["allowed_types"]),
            namespaces=namespaces,
            retrieval=MemoryRetrievalConfig(**data["memory"]["retrieval"]),
        ),
        context_assembly=ContextAssemblyConfig(**data["context_assembly"]),
        runtime_budgets=runtime_budgets,
        policy=PolicyConfig(
            cloud_models_enabled=data["policy"]["cloud_models_enabled"],
            tools_enabled=data["policy"]["tools_enabled"],
            autonomous_memory_write_enabled=data["policy"][
                "autonomous_memory_write_enabled"
            ],
            model_access=data["policy"]["model_access"],
            memory_write=SensitivityDecisionConfig(
                **data["policy"]["memory_write"],
            ),
            context_inclusion=SensitivityDecisionConfig(
                **data["policy"]["context_inclusion"],
            ),
        ),
        permissions=PermissionsConfig(
            mode=PermissionMode(data["permissions"]["mode"]),
            modes=data["permissions"]["modes"],
        ),
        capabilities=_normalize_capabilities(data.get("capabilities", {}), project_root),
        privacy=PrivacyConfig(**data["privacy"]),
        observability=ObservabilityConfig(**data["observability"]),
        raw=data,
    )


def _validate(settings: Settings) -> None:
    if settings.api.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("api.host must be loopback in Phase 1")

    missing_profiles = REQUIRED_MODEL_PROFILES - set(settings.model_profiles)
    if missing_profiles:
        raise ConfigError(f"missing model profiles: {sorted(missing_profiles)}")

    if settings.policy.cloud_models_enabled:
        raise ConfigError("cloud models must be disabled in Phase 1")
    for name, profile in settings.model_profiles.items():
        if profile.cloud and profile.enabled:
            raise ConfigError(f"cloud profile must be disabled in Phase 1: {name}")
        if not profile.cloud and profile.enabled and profile.endpoint is not None:
            if not _is_local_endpoint(profile.endpoint):
                raise ConfigError(f"local model endpoint must stay local in Phase 1: {name}")

    if settings.privacy.raw_prompt_logging:
        raise ConfigError("raw prompt logging must be disabled")
    if settings.privacy.raw_secret_logging:
        raise ConfigError("raw secret logging must be disabled")
    if settings.context_assembly.full_prompt_logging:
        raise ConfigError("full prompt logging must be disabled")
    if settings.observability.log_raw_messages:
        raise ConfigError("raw message logs must be disabled")
    if settings.observability.log_raw_prompts:
        raise ConfigError("raw prompt logs must be disabled")
    if settings.observability.log_model_outputs:
        raise ConfigError("model output logs must be disabled")

    if settings.memory.allowed_types != MEMORY_TYPES:
        raise ConfigError("memory allowed_types must match Phase 1 memory types")
    missing_namespaces = REQUIRED_MEMORY_NAMESPACES - set(settings.memory.namespaces)
    if missing_namespaces:
        raise ConfigError(f"missing memory namespaces: {sorted(missing_namespaces)}")
    for name, namespace in settings.memory.namespaces.items():
        if namespace.sensitivity not in SENSITIVITY_VALUES:
            raise ConfigError(f"invalid sensitivity for namespace {name}")
        if not set(namespace.allowed_types).issubset(settings.memory.allowed_types):
            raise ConfigError(f"invalid allowed_types for namespace {name}")

    budget = settings.runtime_budgets.get("memory_augmented_answer")
    if budget is None:
        raise ConfigError("missing runtime budget: memory_augmented_answer")
    if budget.max_model_calls != 1 or budget.max_tool_calls != 0:
        raise ConfigError("memory_augmented_answer must be deterministic in Phase 1")
    if budget.allow_cloud or budget.allow_tools or budget.allow_autonomous_memory_write:
        raise ConfigError("Phase 1 runtime capabilities exceed MVP scope")

    if "secret" not in settings.policy.memory_write.deny_sensitivity:
        raise ConfigError("secret memory writes must be denied")
    if "secret" not in settings.policy.context_inclusion.deny_sensitivity:
        raise ConfigError("secret context inclusion must be denied")
    if settings.permissions.mode not in {
        PermissionMode.LOCKED_DOWN,
        PermissionMode.DEVELOPER_LOCAL,
        PermissionMode.AUTOMATION,
    }:
        raise ConfigError("invalid permission mode")
    missing_modes = {"locked_down", "developer_local", "automation"} - set(
        settings.permissions.modes,
    )
    if missing_modes:
        raise ConfigError(f"missing permission modes: {sorted(missing_modes)}")
    for mode_name, capability_actions in settings.permissions.modes.items():
        try:
            PermissionMode(mode_name)
        except ValueError as exc:
            raise ConfigError(f"invalid permission mode: {mode_name}") from exc
        if not isinstance(capability_actions, dict):
            raise ConfigError(f"permission mode must be a mapping: {mode_name}")
        for capability_name, action in capability_actions.items():
            try:
                Capability(capability_name)
            except ValueError as exc:
                raise ConfigError(f"invalid capability in permissions: {capability_name}") from exc
            try:
                PolicyDecisionOutcome(action)
            except ValueError as exc:
                raise ConfigError(
                    f"invalid permission action for {mode_name}.{capability_name}",
                ) from exc

    shell_read = settings.capabilities.get("tool.shell.read")
    if not isinstance(shell_read, dict):
        raise ConfigError("capabilities.tool.shell.read must be configured")
    allowed_roots = shell_read.get("allowed_roots")
    if (
        not isinstance(allowed_roots, list)
        or not allowed_roots
        or not all(isinstance(root, str) and root for root in allowed_roots)
    ):
        raise ConfigError("capabilities.tool.shell.read.allowed_roots must be non-empty")
    for key in ("max_output_bytes", "timeout_seconds"):
        value = shell_read.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"capabilities.tool.shell.read.{key} must be a positive integer")

    system_read = settings.capabilities.get("tool.system.read")
    if not isinstance(system_read, dict):
        raise ConfigError("capabilities.tool.system.read must be configured")
    system_allowed_roots = system_read.get("allowed_roots")
    if (
        not isinstance(system_allowed_roots, list)
        or not system_allowed_roots
        or not all(isinstance(root, str) and root for root in system_allowed_roots)
    ):
        raise ConfigError("capabilities.tool.system.read.allowed_roots must be non-empty")
    enabled_families = system_read.get("enabled_families")
    valid_families = {"process", "resources", "hardware", "network", "sensors"}
    if (
        not isinstance(enabled_families, list)
        or not enabled_families
        or not all(isinstance(family, str) and family in valid_families for family in enabled_families)
    ):
        raise ConfigError("capabilities.tool.system.read.enabled_families must be valid")
    for key in ("max_output_bytes", "max_lines", "timeout_seconds"):
        value = system_read.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"capabilities.tool.system.read.{key} must be a positive integer")


def _normalize_capabilities(
    capabilities: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    normalized = dict(capabilities)
    shell_read = dict(normalized.get("tool.shell.read", {}))
    if shell_read:
        roots = shell_read.get("allowed_roots", [])
        if isinstance(roots, list) and all(isinstance(root, str) for root in roots):
            shell_read["allowed_roots"] = [
                _normalize_config_path(root, project_root)
                for root in roots
            ]
        else:
            shell_read["allowed_roots"] = roots
        normalized["tool.shell.read"] = shell_read
    system_read = dict(normalized.get("tool.system.read", {}))
    if system_read:
        roots = system_read.get("allowed_roots", [])
        if isinstance(roots, list) and all(isinstance(root, str) for root in roots):
            system_read["allowed_roots"] = [
                _normalize_config_path(root, project_root)
                for root in roots
            ]
        else:
            system_read["allowed_roots"] = roots
        normalized["tool.system.read"] = system_read
    return normalized


def _normalize_config_path(value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def _is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False

    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host and host not in {"0", "0.0.0.0"}
    return address.is_loopback or address.is_private or address.is_link_local


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = data
    for part in path[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ConfigError(f"environment override crosses scalar key: {part}")
        cursor = next_value
    cursor[path[-1]] = value


def _parse_env_value(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return value if parsed is None and value.lower() != "null" else parsed
