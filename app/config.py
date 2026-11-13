import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
MODELS_CONFIG_FILE = PROJECT_ROOT / "config" / "models.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    qdrant_url: Optional[str] = None
    groq_api_key: Optional[str] = Field(default=None, repr=False)
    gemini_api_key: Optional[str] = Field(default=None, repr=False)
    openrouter_api_key: Optional[str] = Field(default=None, repr=False)
    mock_api_key: Optional[str] = Field(default=None, repr=False)


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