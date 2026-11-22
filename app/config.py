import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
MODELS_CONFIG_FILE = PROJECT_ROOT / "config" / "models.yaml"
FASTEMBED_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"
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
    provider: str
    model_id: str
    tier: Optional[str] = None


class ModelsConfig(BaseModel):
    providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelConfig]


class UnknownModelAliasError(ValueError):
    pass


def load_models_config(path: Optional[Path] = None) -> ModelsConfig:
    config_path = path or MODELS_CONFIG_FILE
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return ModelsConfig.model_validate(raw_config)


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