import pytest

import app.config as app_config
from app.config import ModelConfig, ProviderConfig, Settings, load_models_config, resolve


def test_settings_reads_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql://example\nGROQ_API_KEY=test-key\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.database_url == "postgresql://example"
    assert settings.groq_api_key == "test-key"


def test_load_models_config_parses_yaml_models():
    config = load_models_config()

    assert isinstance(config.providers["groq"], ProviderConfig)
    assert isinstance(config.models["groq-8b"], ModelConfig)
    assert config.models["groq-8b"].model_id == "llama-3.1-8b-instant"


def test_resolve_returns_provider_model_id_and_environment_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    provider, model_id, api_key = resolve("groq-8b")

    assert provider == ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    )
    assert model_id == "llama-3.1-8b-instant"
    assert api_key == "test-groq-key"


def test_resolve_unknown_alias_has_clear_error():
    with pytest.raises(ValueError, match="Unknown model alias 'missing'"):
        resolve("missing")


def test_resolve_reports_missing_provider_key(tmp_path, monkeypatch):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        "providers:\n"
        "  test:\n"
        "    base_url: https://example.test/v1\n"
        "    api_key_env: TEST_PROVIDER_API_KEY\n"
        "models:\n"
        "  test-model:\n"
        "    provider: test\n"
        "    model_id: test-model-id\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_config, "MODELS_CONFIG_FILE", config_file)
    monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing API key 'TEST_PROVIDER_API_KEY'"):
        resolve("test-model")
