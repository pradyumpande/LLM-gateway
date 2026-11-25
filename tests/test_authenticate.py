import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

import app.main as main
from app.auth import hash_key
from app.config import ProviderConfig


def _configure_auth_dependencies(monkeypatch, *, auth_row):
    database = SimpleNamespace(
        fetchrow=AsyncMock(return_value=auth_row),
        close=AsyncMock(),
    )
    redis = SimpleNamespace(
        get=AsyncMock(return_value=None),
        set=AsyncMock(),
        delete=AsyncMock(),
        ping=AsyncMock(),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        main,
        "Settings",
        lambda: SimpleNamespace(
            database_url="postgresql://test",
            redis_url="redis://test",
            models_config_path="config/models.yaml",
            qdrant_url="http://qdrant.test",
            qdrant_api_key=None,
            pii_score_threshold=0.5,
            max_input_chars=200_000,
            injection_flag_threshold=0.4,
            injection_block_threshold=0.8,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            cache_max_temperature=0.7,
            cache_max_bytes=102_400,
            cache_lookup_timeout_s=1.5,
            cache_cleanup_interval_s=600,
            semantic_enabled=False,
            semantic_threshold=0.92,
            semantic_min_chars=8,
            semantic_max_chars=1000,
            embed_model="BAAI/bge-small-en-v1.5",
            fastembed_cache_dir=".cache/fastembed",
            pii_entities=[
                "EMAIL_ADDRESS",
                "IN_PHONE",
                "AADHAAR",
                "IN_PAN",
                "CREDIT_CARD",
            ],
        ),
    )
    monkeypatch.setattr(
        main,
        "load_models_config",
        lambda _path: SimpleNamespace(providers={}),
    )
    monkeypatch.setattr(main, "validate_config", lambda config, _env: config)
    monkeypatch.setattr(
        main,
        "FastEmbedEmbedder",
        lambda model_name, cache_dir: SimpleNamespace(dim=384, name=model_name),
    )
    monkeypatch.setattr(
        main,
        "build_analyzer",
        Mock(return_value=SimpleNamespace(analyze=Mock(return_value=[]))),
    )
    monkeypatch.setattr(main.db, "create_pool", AsyncMock(return_value=database))
    monkeypatch.setattr(
        main.redis_client,
        "create_client",
        Mock(return_value=redis),
    )
    monkeypatch.setattr(main, "check_rate_limit", AsyncMock(return_value=(True, 0)))
    monkeypatch.setattr(
        main,
        "check_budget",
        AsyncMock(return_value=(True, "spend:12:2026-11", 0.0)),
    )
    monkeypatch.setattr(main, "current_month", lambda: "2026-11")
    monkeypatch.setattr(
        main,
        "cost_usd",
        lambda alias, prompt_tokens, completion_tokens: (
            Decimal(prompt_tokens + completion_tokens) / Decimal(1000)
        ),
    )
    monkeypatch.setattr(main, "record_spend", AsyncMock(return_value=0.002))
    return database, redis


def test_chat_without_authorization_returns_openai_401(monkeypatch):
    database, _ = _configure_auth_dependencies(monkeypatch, auth_row=None)

    with TestClient(main.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "groq-8b", "messages": []},
        )

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert response.json()["error"]["code"] == "invalid_api_key"
    database.fetchrow.assert_not_awaited()


def test_chat_with_invalid_key_returns_openai_401_and_caches_miss(monkeypatch):
    database, redis = _configure_auth_dependencies(monkeypatch, auth_row=None)

    with TestClient(main.app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer not-a-valid-key"},
            json={"model": "groq-8b", "messages": []},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    database.fetchrow.assert_awaited_once()
    redis.set.assert_awaited_once_with(
        f"auth:{hash_key('not-a-valid-key')}",
        "invalid",
        ex=10,
    )


def test_valid_key_authenticates_caches_context_and_proceeds(monkeypatch):
    key = "gw_valid-example-key"
    row = {
        "team_id": 12,
        "key_id": 34,
        "team_name": "search-team",
        "rpm": 60,
        "burst": 10,
        "budget_usd": Decimal("5"),
        "pii_action": "mask",
    }
    database, redis = _configure_auth_dependencies(monkeypatch, auth_row=row)
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_KEY",
            ),
            "provider-model",
            "provider-key",
        ),
    )

    class FakeProvider:
        def __init__(self, **kwargs):
            pass

        async def chat(self, payload):
            return {
                "choices": [{"message": {"content": "authenticated"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    monkeypatch.setattr(main, "OpenAICompatProvider", FakeProvider)

    with TestClient(main.app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "mock-fast", "messages": []},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "authenticated"
    database.fetchrow.assert_awaited_once()
    cache_key, cache_json = redis.set.await_args.args
    assert cache_key == f"auth:{hash_key(key)}"
    assert json.loads(cache_json) == {
        "team_id": 12,
        "key_id": 34,
        "team_name": "search-team",
        "rpm": 60,
        "burst": 10,
        "budget_usd": "5",
        "pii_action": "mask",
    }
    assert redis.set.await_args.kwargs == {"ex": 60}
