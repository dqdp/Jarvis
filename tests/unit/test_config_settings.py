from __future__ import annotations

from pathlib import Path

import pytest

from assistant_core.config import ConfigError, ConfigLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def load_settings(config_name: str = "default"):
    return ConfigLoader(PROJECT_ROOT / "config").load(config_name)


def test_default_config_validates() -> None:
    settings = load_settings("default")

    assert settings.app.environment == "local"


def test_test_config_validates() -> None:
    settings = load_settings("test")

    assert settings.app.environment == "test"


def test_ollama_config_uses_local_models_and_endpoints() -> None:
    settings = load_settings("ollama")

    assert settings.model_profiles["local_main"].provider == "ollama"
    assert settings.model_profiles["local_main"].model == "qwen3.5:9b"
    assert settings.model_profiles["local_structured"].model == "qwen3.5:9b"
    assert settings.model_profiles["local_embedding"].model == "embeddinggemma:latest"
    assert settings.model_profiles["local_main"].max_output_tokens == 1024
    assert settings.model_profiles["local_structured"].max_output_tokens == 1024
    assert settings.model_profiles["local_main"].endpoint == "http://127.0.0.1:11434"
    assert settings.model_profiles["local_embedding"].endpoint == "http://127.0.0.1:11434"
    assert settings.model_profiles["cloud_reasoning"].enabled is False


def test_env_override_nested_keys_apply_with_jarvis_prefix(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_API__PORT", "9090")

    settings = load_settings("default")

    assert settings.api.port == 9090


def test_cloud_reasoning_disabled_by_default() -> None:
    settings = load_settings("default")

    assert settings.model_profiles["cloud_reasoning"].enabled is False
    assert settings.policy.cloud_models_enabled is False


def test_raw_prompt_logging_disabled_by_default() -> None:
    settings = load_settings("default")

    assert settings.privacy.raw_prompt_logging is False
    assert settings.context_assembly.full_prompt_logging is False
    assert settings.observability.log_raw_prompts is False
    assert settings.observability.log_raw_messages is False
    assert settings.observability.log_model_outputs is False


@pytest.mark.parametrize(
    "env_name",
    [
        "JARVIS_OBSERVABILITY__LOG_RAW_MESSAGES",
        "JARVIS_OBSERVABILITY__LOG_MODEL_OUTPUTS",
    ],
)
def test_raw_message_and_model_output_logging_cannot_be_enabled(
    monkeypatch,
    env_name: str,
) -> None:
    monkeypatch.setenv(env_name, "true")

    with pytest.raises(ConfigError):
        load_settings("default")


def test_required_model_profiles_exist() -> None:
    settings = load_settings("default")

    assert {"local_main", "local_structured", "local_embedding"}.issubset(
        settings.model_profiles
    )


def test_memory_namespace_registry_valid() -> None:
    settings = load_settings("default")

    assert set(settings.memory.namespaces) == {
        "user.preferences",
        "user.working_style",
        "project.personal_assistant",
        "system.runtime_rules",
        "environment.inference_node",
    }
    for namespace in settings.memory.namespaces.values():
        assert set(namespace.allowed_types).issubset(settings.memory.allowed_types)


def test_runtime_budget_memory_augmented_answer_limits() -> None:
    settings = load_settings("default")
    budget = settings.runtime_budgets["memory_augmented_answer"]

    assert budget.max_model_calls == 1
    assert budget.max_tool_calls == 0
    assert budget.allow_cloud is False
    assert budget.allow_tools is False
    assert budget.allow_autonomous_memory_write is False


def test_secret_not_allowed_in_memory_write_policy_config() -> None:
    settings = load_settings("default")

    assert "secret" in settings.policy.memory_write.deny_sensitivity


def test_invalid_config_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_POLICY__CLOUD_MODELS_ENABLED", "true")

    with pytest.raises(ConfigError):
        load_settings("default")


def test_permission_actions_must_be_known(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_PERMISSIONS__MODES__DEVELOPER_LOCAL__TOOL.SAFE", "maybe")

    with pytest.raises(ConfigError):
        load_settings("default")


def test_shell_read_allowed_roots_must_be_explicit(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_CAPABILITIES__TOOL.SHELL.READ__ALLOWED_ROOTS", "[]")

    with pytest.raises(ConfigError):
        load_settings("default")


def test_shell_read_allowed_roots_must_be_a_list(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_CAPABILITIES__TOOL.SHELL.READ__ALLOWED_ROOTS", '"/tmp"')

    with pytest.raises(ConfigError):
        load_settings("default")


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("JARVIS_CAPABILITIES__TOOL.SHELL.READ__MAX_OUTPUT_BYTES", "0"),
        ("JARVIS_CAPABILITIES__TOOL.SHELL.READ__TIMEOUT_SECONDS", "0"),
    ],
)
def test_shell_read_limits_must_be_positive(monkeypatch, env_name: str, value: str) -> None:
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ConfigError):
        load_settings("default")


def test_local_model_endpoint_must_not_be_external(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_MODEL_PROFILES__LOCAL_MAIN__ENDPOINT", "https://api.openai.com/v1")

    with pytest.raises(ConfigError):
        load_settings("default")


def test_api_host_must_be_loopback_without_auth(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_API__HOST", "0.0.0.0")

    with pytest.raises(ConfigError):
        load_settings("default")


def test_committed_yaml_does_not_contain_secret_values() -> None:
    forbidden_markers = ("sk-", "-----BEGIN", "password:", "token:")

    for config_path in (PROJECT_ROOT / "config").glob("*.yaml"):
        content = config_path.read_text(encoding="utf-8")
        assert not any(marker in content for marker in forbidden_markers)
