import asyncio
import secrets
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import redis.asyncio as redis
from fastapi import HTTPException
from starlette.requests import Request

from app import admin
from app.auth import AuthContext, authenticate, hash_key
from app.budget import current_month, spend_key
from app.config import ProviderConfig, Settings
from app.main import app
from app.ratelimit import check_rate_limit


@pytest.fixture
def redis_url():
    return Settings().redis_url or "redis://localhost:6379/0"


@pytest.fixture
async def live_redis(redis_url):
    client = redis.from_url(
        redis_url,
        decode_responses=True,
        max_connections=300,
    )
    await client.ping()
    yield client
    await client.aclose()


def _new_team_id():
    return secrets.randbelow(2_000_000_000) + 1


def _rate_limit_context(team_id, *, burst=10, rpm=1):
    return AuthContext(
        team_id=team_id,
        key_id=team_id,
        team_name="redis-integration-test",
        rpm=rpm,
        burst=burst,
        budget_usd=Decimal("10"),
        pii_action="mask",
    )


@pytest.mark.asyncio
async def test_token_bucket_is_atomic_under_concurrent_requests(live_redis):
    team_id = _new_team_id()
    bucket_key = f"rl:{team_id}"
    naive_key = f"naive:{team_id}"
    context = _rate_limit_context(team_id)
    await live_redis.delete(bucket_key, naive_key)
    try:
        results = await asyncio.gather(
            *(check_rate_limit(live_redis, context) for _ in range(200))
        )
        assert sum(allowed for allowed, _ in results) == 10

        # All checks read before any write, making the GET/SET race deterministic.
        arrived = 0
        all_reads_complete = asyncio.Event()

        async def naive_get_set_check():
            nonlocal arrived
            tokens = int(await live_redis.get(naive_key) or 10)
            arrived += 1
            if arrived == 200:
                all_reads_complete.set()
            await all_reads_complete.wait()
            if tokens < 1:
                return False
            await live_redis.set(naive_key, tokens - 1)
            return True

        naive_results = await asyncio.gather(
            *(naive_get_set_check() for _ in range(200))
        )
        assert sum(naive_results) > 10
    finally:
        await live_redis.delete(bucket_key, naive_key)


@pytest.mark.asyncio
async def test_token_bucket_refills_at_one_token_per_second(live_redis):
    team_id = _new_team_id()
    bucket_key = f"rl:{team_id}"
    context = _rate_limit_context(team_id, burst=10, rpm=60)
    await live_redis.delete(bucket_key)
    try:
        drained = await asyncio.gather(
            *(check_rate_limit(live_redis, context) for _ in range(10))
        )
        assert all(allowed for allowed, _ in drained)

        await asyncio.sleep(2.1)
        refilled = await asyncio.gather(
            *(check_rate_limit(live_redis, context) for _ in range(3))
        )
        assert sum(allowed for allowed, _ in refilled) == 2
    finally:
        await live_redis.delete(bucket_key)


def _auth_request(app_instance, key=None):
    headers = []
    if key is not None:
        headers.append((b"authorization", f"Bearer {key}".encode()))
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": headers,
            "client": ("testclient", 123),
            "server": ("testserver", 80),
            "app": app_instance,
        }
    )


@pytest.mark.asyncio
async def test_authentication_with_redis_and_immediate_key_revocation(live_redis):
    key = "gw_integration_" + secrets.token_urlsafe(20)
    bad_key = "gw_bad_" + secrets.token_urlsafe(20)
    hashed_key = hash_key(key)
    auth_cache_key = f"auth:{hashed_key}"
    bad_cache_key = f"auth:{hash_key(bad_key)}"
    database = SimpleNamespace(
        fetchrow=AsyncMock(
            side_effect=[
                None,
                {
                    "team_id": 88,
                    "key_id": 99,
                    "team_name": "integration-team",
                    "rpm": 60,
                    "burst": 10,
                    "budget_usd": Decimal("1"),
                    "pii_action": "mask",
                },
                {"id": 99, "key_hash": hashed_key},
                None,
            ]
        )
    )
    request_app = SimpleNamespace(
        state=SimpleNamespace(db=database, redis=live_redis)
    )
    await live_redis.delete(auth_cache_key, bad_cache_key)

    try:
        with pytest.raises(HTTPException) as missing:
            await authenticate(_auth_request(request_app))
        assert missing.value.status_code == 401

        with pytest.raises(HTTPException) as invalid:
            await authenticate(_auth_request(request_app, bad_key))
        assert invalid.value.status_code == 401
        assert await live_redis.get(bad_cache_key) == "invalid"

        context = await authenticate(_auth_request(request_app, key))
        assert context.team_id == 88
        assert (await authenticate(_auth_request(request_app, key))).key_id == 99
        assert database.fetchrow.await_count == 2

        await admin.routes.deactivate_key(99, _auth_request(request_app))
        assert await live_redis.get(auth_cache_key) is None

        with pytest.raises(HTTPException) as revoked:
            await authenticate(_auth_request(request_app, key))
        assert revoked.value.status_code == 401
        assert await live_redis.get(auth_cache_key) == "invalid"
    finally:
        await live_redis.delete(auth_cache_key, bad_cache_key)


@pytest.mark.asyncio
async def test_mock_fast_budget_rejects_after_first_success(
    live_redis,
    redis_url,
    monkeypatch,
):
    team_id = _new_team_id()
    month = current_month()
    context = AuthContext(
        team_id=team_id,
        key_id=team_id,
        team_name="budget-integration-test",
        rpm=600,
        burst=20,
        budget_usd=Decimal("0.01"),
        pii_action="mask",
    )
    bucket_key = f"rl:{team_id}"
    budget_key = spend_key(team_id, month)
    alert_key = f"alert80:{team_id}:{month}"
    await live_redis.delete(bucket_key, budget_key, alert_key)

    database = SimpleNamespace(close=AsyncMock(), execute=AsyncMock())
    monkeypatch.setattr(
        "app.main.Settings",
        lambda: SimpleNamespace(
            database_url="postgresql://test",
            redis_url=redis_url,
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
        "app.main.load_models_config",
        lambda _path: SimpleNamespace(providers={}),
    )
    monkeypatch.setattr("app.main.validate_config", lambda config, _env: config)
    monkeypatch.setattr(
        "app.main.FastEmbedEmbedder",
        lambda model_name, cache_dir: SimpleNamespace(dim=384, name=model_name),
    )
    monkeypatch.setattr(
        "app.main.build_analyzer",
        lambda: SimpleNamespace(analyze=lambda **kwargs: []),
    )
    monkeypatch.setattr(
        "app.main.db.create_pool",
        AsyncMock(return_value=database),
    )
    monkeypatch.setattr(
        "app.main.redis_client.create_client",
        lambda url: redis.from_url(url, decode_responses=True),
    )
    monkeypatch.setattr("app.main.check_rate_limit", AsyncMock(return_value=(True, 0)))
    monkeypatch.setattr("app.main.resolve", lambda alias: (
        ProviderConfig(base_url="https://provider.test/v1", api_key_env="TEST_KEY"),
        "mock-1",
        "test-key",
    ))
    monkeypatch.setattr(
        "app.main.OpenAICompatProvider",
        lambda **kwargs: SimpleNamespace(
            chat=AsyncMock(
                return_value={
                    "choices": [{"message": {"content": "mock reply"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }
            )
        ),
    )
    app.dependency_overrides[authenticate] = lambda: context

    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                first = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "mock-fast",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                second = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "mock-fast",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        assert first.status_code == 200
        assert first.headers["x-cost"] == "0.04"
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "budget_exceeded"
    finally:
        app.dependency_overrides.pop(authenticate, None)
        await live_redis.delete(bucket_key, budget_key, alert_key)
