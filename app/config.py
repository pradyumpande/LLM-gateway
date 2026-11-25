import os
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.usage.pricing import pricing_aliases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
MODELS_CONFIG_FILE = PROJECT_ROOT / "config" / "models.yaml"
FASTEMBED_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"
logger = logging.getLogger(__name__)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(FASTEMBED_CACHE_DIR))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = Field(default=None, repr=False)
    admin_token: Optional[str] = Field(default=None, repr=False)
    groq_api_key: Optional[str] = Field(default=None, repr=False)
    gemini_api_key: Optional[str] = Field(default=None, repr=False)
    openrouter_api_key: Optional[str] = Field(default=None, repr=False)
    mock_api_key: Optional[str] = Field(default=None, repr=False)
    models_config_path: Path = MODELS_CONFIG_FILE
    request_deadline_s: float = Field(default=45, gt=0)
    provider_call_timeout_s: float = Field(default=30, gt=0)
    stream_first_chunk_timeout_s: float = Field(default=15, gt=0)
    max_attempts_per_candidate: int = Field(default=2, gt=0)
    retry_backoff_base_s: float = Field(default=0.25, ge=0)
    retry_backoff_cap_s: float = Field(default=2.0, ge=0)
    breaker_failure_threshold: int = Field(default=5, gt=0)
    breaker_cooldown_s: float = Field(default=30, gt=0)
    breaker_max_cooldown_s: float = Field(default=300, gt=0)
    penalty_default_s: float = Field(default=10, ge=0)
    penalty_max_s: float = Field(default=60, ge=0)
    penalty_model_not_found_s: float = Field(default=300, ge=0)
    injection_flag_threshold: float = Field(default=0.4, ge=0, le=1)
    injection_block_threshold: float = Field(default=0.8, ge=0, le=1)
    pii_score_threshold: float = Field(default=0.5, ge=0, le=1)
    max_input_chars: int = Field(default=200_000, gt=0, le=1_000_000)
    pii_entities: list[str] = Field(
        default_factory=lambda: [
            "EMAIL_ADDRESS",
            "IN_PHONE",
            "AADHAAR",
            "IN_PAN",
            "CREDIT_CARD",
        ]
    )
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    cache_max_temperature: float = Field(default=0.7, ge=0, le=2)
    cache_max_bytes: int = Field(default=102_400, gt=0)
    cache_lookup_timeout_s: float = Field(default=1.5, gt=0)
    cache_cleanup_interval_s: int = Field(default=600, gt=0)
    semantic_enabled: bool = True
    semantic_threshold: float = Field(default=0.95, ge=0, le=1)
    semantic_min_chars: int = Field(default=8, gt=0)
    semantic_max_chars: int = Field(default=1000, gt=0)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: Path = FASTEMBED_CACHE_DIR

    @field_validator("fastembed_cache_dir")
    @classmethod
    def resolve_fastembed_cache_dir(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

    @field_validator("models_config_path")
    @classmethod
    def resolve_models_config_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

    @model_validator(mode="after")
    def validate_semantic_character_range(self) -> "Settings":
        if self.semantic_max_chars < self.semantic_min_chars:
            raise ValueError(
                "SEMANTIC_MAX_CHARS must be greater than or equal to "
                "SEMANTIC_MIN_CHARS."
            )
        return self


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str


class RoutingConfig(BaseModel):
    strong_threshold: int = Field(default=3, ge=0)
    tiers: Dict[str, list[str]] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


class UnknownModelAliasError(ValueError):
    pass


def load_models_config(path: Optional[Path] = None) -> ModelsConfig:
    config_path = path or Settings().models_config_path
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return ModelsConfig.model_validate(raw_config)


def validate_config(
    cfg: ModelsConfig,
    env: Mapping[str, Optional[str]],
) -> ModelsConfig:
    aliases = set(cfg.models)
    configured_prices = pricing_aliases()
    for alias in sorted(aliases - configured_prices):
        raise ValueError(
            f"Model alias {alias!r} is missing from config/pricing.yaml."
        )

    for alias, model in cfg.models.items():
        if model.provider not in cfg.providers:
            raise ValueError(
                f"Model alias {alias!r} references unknown provider "
                f"{model.provider!r}."
            )

    for tier in cfg.routing.tiers:
        if tier.lower() == "auto":
            raise ValueError("Routing tier name 'auto' is reserved.")
        if tier in aliases:
            raise ValueError(
                f"Routing tier name {tier!r} conflicts with a model alias."
            )
        for alias in cfg.routing.tiers[tier]:
            if alias not in aliases:
                raise ValueError(
                    f"Routing tier {tier!r} references unknown model alias "
                    f"{alias!r}."
                )

    available_aliases: set[str] = set()
    for alias, model in cfg.models.items():
        provider = cfg.providers[model.provider]
        api_key = env.get(provider.api_key_env)
        if api_key is None or not api_key.strip():
            logger.warning(
                "Dropping model alias %r from routing tiers: API key %s is empty.",
                alias,
                provider.api_key_env,
            )
        else:
            available_aliases.add(alias)

    filtered_tiers: dict[str, list[str]] = {}
    for tier, tier_aliases in cfg.routing.tiers.items():
        available = [alias for alias in tier_aliases if alias in available_aliases]
        if not available:
            raise ValueError(
                f"Routing tier {tier!r} has no models with configured provider "
                "API keys."
            )
        filtered_tiers[tier] = available

    return cfg.model_copy(
        update={
            "routing": cfg.routing.model_copy(update={"tiers": filtered_tiers})
        }
    )


def resolve(alias: str) -> Tuple[ProviderConfig, str, str]:
    config = load_models_config()
    model = config.models.get(alias)
    if model is None:
        aliases = ", ".join(sorted(config.models)) or "(none)"
        raise UnknownModelAliasError(
            f"Unknown model alias {alias!r}. Available aliases: {aliases}."
        )

    provider = config.providers.get(model.provider)
    if provider is None:
        raise ValueError(
            f"Model alias {alias!r} references unknown provider "
            f"{model.provider!r}."
        )

    api_key = os.getenv(provider.api_key_env)
    if api_key is None:
        settings = Settings()
        api_key = getattr(settings, provider.api_key_env.lower(), None)
    if api_key is None or not api_key.strip():
        raise ValueError(
            f"Missing API key {provider.api_key_env!r}; set it in the "
            "environment or .env file."
        )

    return provider, model.model_id, api_key