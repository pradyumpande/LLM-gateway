import pytest

import app.config as app_config
from app.config import (
    ModelConfig,
    ModelsConfig,
    ProviderConfig,
    RoutingConfig,
    Settings,
    load_models_config,
    resolve,
    validate_config,
)


def test_settings_reads_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql://example\nGROQ_API_KEY=test-key\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.database_url == "postgresql://example"
    assert settings.groq_api_key == "test-key"


def test_settings_reads_cache_options_and_resolves_project_cache_path(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CACHE_ENABLED=false\n"
        "CACHE_TTL_SECONDS=90\n"
        "SEMANTIC_THRESHOLD=0.94\n"
        "FASTEMBED_CACHE_DIR=.cache/custom\n"
        "QDRANT_API_KEY=\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.cache_enabled is False
    assert settings.cache_ttl_seconds == 90
    assert settings.semantic_threshold == 0.94
    assert settings.fastembed_cache_dir == app_config.PROJECT_ROOT / ".cache/custom"
    assert settings.qdrant_api_key == ""


def test_settings_reads_routing_options_and_resolves_models_config_path(tmp_path):
    models_path = tmp_path / "models.yaml"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"MODELS_CONFIG_PATH={models_path}\n"
        "REQUEST_DEADLINE_S=45\n"
        "PROVIDER_CALL_TIMEOUT_S=30\n"
        "STREAM_FIRST_CHUNK_TIMEOUT_S=15\n"
        "MAX_ATTEMPTS_PER_CANDIDATE=2\n"
        "RETRY_BACKOFF_BASE_S=0.25\n"
        "RETRY_BACKOFF_CAP_S=2.0\n"
        "BREAKER_FAILURE_THRESHOLD=5\n"
        "BREAKER_COOLDOWN_S=30\n"
        "BREAKER_MAX_COOLDOWN_S=300\n"
        "PENALTY_DEFAULT_S=10\n"
        "PENALTY_MAX_S=60\n"
        "PENALTY_MODEL_NOT_FOUND_S=300\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.models_config_path == models_path
    assert settings.request_deadline_s == 45
    assert settings.provider_call_timeout_s == 30
    assert settings.stream_first_chunk_timeout_s == 15
    assert settings.max_attempts_per_candidate == 2
    assert settings.retry_backoff_base_s == 0.25
    assert settings.retry_backoff_cap_s == 2.0
    assert settings.breaker_failure_threshold == 5
    assert settings.breaker_cooldown_s == 30
    assert settings.breaker_max_cooldown_s == 300
    assert settings.penalty_default_s == 10
    assert settings.penalty_max_s == 60
    assert settings.penalty_model_not_found_s == 300


def test_settings_rejects_inverted_semantic_character_range():
    with pytest.raises(ValueError, match="SEMANTIC_MAX_CHARS"):
        Settings(
            _env_file=None,
            semantic_min_chars=20,
            semantic_max_chars=10,
        )


def test_load_models_config_parses_yaml_models():
    config = load_models_config()

    assert isinstance(config.providers["groq"], ProviderConfig)
    assert isinstance(config.models["groq-8b"], ModelConfig)
    assert config.models["groq-8b"].model_id == "llama-3.1-8b-instant"
    assert config.routing.tiers["cheap"] == [
        "groq-8b",
        "gemini-flash-lite",
        "openrouter-free",
    ]


def test_load_models_config_uses_settings_path(tmp_path, monkeypatch):
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "providers: {mock: {base_url: http://localhost, api_key_env: MOCK_API_KEY}}\n"
        "models: {mock-fast: {provider: mock, model_id: mock-1}}\n"
        "routing: {tiers: {cheap: [mock-fast]}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(models_file))

    config = load_models_config()

    assert config.models["mock-fast"].provider == "mock"
    assert config.routing.tiers["cheap"] == ["mock-fast"]


def _validation_config(
    *,
    models=None,
    tiers=None,
):
    return ModelsConfig(
        providers={
            "provider-a": ProviderConfig(
                base_url="https://a.test",
                api_key_env="KEY_A",
            ),
            "provider-b": ProviderConfig(
                base_url="https://b.test",
                api_key_env="KEY_B",
            ),
        },
        models=models
        or {
            "cheap-a": ModelConfig(provider="provider-a", model_id="a"),
            "cheap-b": ModelConfig(provider="provider-b", model_id="b"),
            "strong-a": ModelConfig(provider="provider-a", model_id="c"),
        },
        routing=RoutingConfig(
            strong_threshold=3,
            tiers=tiers
            or {"cheap": ["cheap-a", "cheap-b"], "strong": ["strong-a"]},
        ),
    )


def test_validate_config_drops_unavailable_aliases_from_tiers(monkeypatch, caplog):
    monkeypatch.setattr(
        app_config,
        "pricing_aliases",
        lambda: frozenset({"cheap-a", "cheap-b", "strong-a"}),
    )

    validated = validate_config(
        _validation_config(),
        {"KEY_A": "configured", "KEY_B": ""},
    )

    assert validated.routing.tiers == {
        "cheap": ["cheap-a"],
        "strong": ["strong-a"],
    }
    assert "Dropping model alias 'cheap-b'" in caplog.text


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_validation_config(tiers={"auto": ["cheap-a"]}), "tier name 'auto' is reserved"),
        (_validation_config(tiers={"cheap-a": ["cheap-a"]}), "conflicts with a model alias"),
        (_validation_config(tiers={"cheap": ["unknown"]}), "unknown model alias 'unknown'"),
        (
            _validation_config(
                models={
                    "cheap-a": ModelConfig(provider="missing", model_id="a")
                },
                tiers={"cheap": ["cheap-a"]},
            ),
            "unknown provider 'missing'",
        ),
    ],
)
def test_validate_config_rejects_invalid_tiers_and_providers(
    config,
    message,
    monkeypatch,
):
    monkeypatch.setattr(
        app_config,
        "pricing_aliases",
        lambda: frozenset(config.models),
    )

    with pytest.raises(ValueError, match=message):
        validate_config(config, {"KEY_A": "configured", "KEY_B": "configured"})


def test_validate_config_requires_prices_and_nonempty_tiers(monkeypatch):
    config = _validation_config()
    monkeypatch.setattr(
        app_config,
        "pricing_aliases",
        lambda: frozenset({"cheap-a", "strong-a"}),
    )

    with pytest.raises(ValueError, match="cheap-b.*missing from config/pricing.yaml"):
        validate_config(config, {"KEY_A": "configured", "KEY_B": "configured"})

    monkeypatch.setattr(
        app_config,
        "pricing_aliases",
        lambda: frozenset(config.models),
    )
    with pytest.raises(ValueError, match="tier 'cheap'.*no models"):
        validate_config(config, {"KEY_A": "", "KEY_B": ""})


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
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(config_file))
    monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing API key 'TEST_PROVIDER_API_KEY'"):
        resolve("test-model")
