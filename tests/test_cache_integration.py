import asyncio
import secrets
import time
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import redis as redis_sync
import redis.asyncio as redis_async
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

import app.admin.routes as admin_routes
import app.main as main
from app.auth import AuthContext, authenticate
from app.budget import current_month, spend_key
from app.cache.embedder import FastEmbedEmbedder
from app.cache.keys import exact_key, params_hash, semantic_scope, sha
from app.cache.policy import build_cache_info
from app.cache.semantic import SemanticCache
from app.cache.service import CacheService
from app.config import ProviderConfig, Settings


@pytest.fixture
def redis_url():
    return Settings().redis_url or "redis://localhost:6379/0"


@pytest.fixture
def qdrant_url():
    return Settings().qdrant_url or "http://localhost:6333"


@pytest.fixture
async def live_redis(redis_url):
    client = redis_async.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Docker Redis is unavailable: {exc}")

    yield client
    await client.aclose()


@pytest.fixture
async def live_qdrant(qdrant_url):
    settings = Settings()
    client = AsyncQdrantClient(
        url=qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=5,
    )
    try:
        await client.get_collections()
    except Exception as exc:
        await client.close()
        pytest.skip(f"Docker Qdrant is unavailable: {exc}")

    yield client
    await client.close()


@pytest.fixture
async def semantic_cache(live_qdrant):
    collection = f"test_cache_{uuid.uuid4().hex}"
    cache = SemanticCache(
        client=live_qdrant,
        collection=collection,
        dim=4,
        threshold=0.92,
        ttl_s=3600,
        embed_model="fake-four-dimensional",
    )
    await cache.ensure_collection()
    yield cache
    await live_qdrant.delete_collection(collection)


@pytest.fixture
def docker_inspectors(redis_url, qdrant_url):
    settings = Settings()
    redis_client = redis_sync.Redis.from_url(
        redis_url,
        decode_responses=True,
    )
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=5,
    )
    try:
        redis_client.ping()
        qdrant_client.get_collections()
    except Exception as exc:
        redis_client.close()
        qdrant_client.close()
        pytest.skip(f"Docker Redis and Qdrant are required: {exc}")

    yield SimpleNamespace(
        redis=redis_client,
        qdrant=qdrant_client,
        redis_url=redis_url,
        qdrant_url=qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
    )
    redis_client.close()
    qdrant_client.close()


@pytest.fixture(scope="session")
def real_embedder():
    settings = Settings()
    redis_probe = redis_sync.Redis.from_url(
        settings.redis_url or "redis://localhost:6379/0",
        decode_responses=True,
    )
    qdrant_probe = QdrantClient(
        url=settings.qdrant_url or "http://localhost:6333",
        api_key=settings.qdrant_api_key or None,
        timeout=5,
    )
    try:
        redis_probe.ping()
        qdrant_probe.get_collections()
    except Exception as exc:
        pytest.skip(f"Docker Redis and Qdrant are required: {exc}")
    finally:
        redis_probe.close()
        qdrant_probe.close()
    return FastEmbedEmbedder(
        "BAAI/bge-small-en-v1.5",
        str(settings.fastembed_cache_dir),
    )


def _entry(content="A complete semantic-cache answer."):
    return {
        "response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ]
        },
        "cost_usd": 0.001,
        "model_alias": "mock-fast",
        "provider": "mock",
        "created_at": int(time.time()),
    }


def _scope(**changes):
    return {
        "team_id": 401,
        "model_requested": "mock-fast",
        "context_hash": "context-a",
        "params_hash": "params-a",
        "num_sig": "none",
        **changes,
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_exact_cache_ttl(live_redis, tmp_path):
    settings = Settings(
        _env_file=None,
        fastembed_cache_dir=tmp_path,
        cache_ttl_seconds=60,
    )
    payload = {
        "model": "mock-fast",
        "messages": [{"role": "user", "content": "Store an exact response."}],
    }
    info = build_cache_info(
        payload,
        404,
        enabled=True,
        max_temperature=settings.cache_max_temperature,
        pii_found=False,
        injection_flagged=False,
        cache_control=None,
        semantic_enabled=False,
        sem_min=settings.semantic_min_chars,
        sem_max=settings.semantic_max_chars,
    )
    service = CacheService(
        live_redis,
        None,
        embedder=None,
        ttl_s=settings.cache_ttl_seconds,
        max_bytes=settings.cache_max_bytes,
        lookup_timeout_s=settings.cache_lookup_timeout_s,
    )

    await service.store(
        info,
        None,
        _entry()["response"],
        cost_usd=0.001,
        model_alias="mock-fast",
        provider="mock",
    )
    ttl = await live_redis.ttl(info.exact_key)

    try:
        assert 0 < ttl <= settings.cache_ttl_seconds
    finally:
        await live_redis.delete(info.exact_key)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_qdrant_similarity_scopes_and_deterministic_upsert(
    semantic_cache,
):
    assert semantic_cache.collection.startswith("test_cache_")
    vector_a = [1.0, 0.0, 0.0, 0.0]
    vector_b95 = [0.95, 0.3122499, 0.0, 0.0]
    scope = _scope()

    await semantic_cache.store(
        vector_a,
        text_hash=sha("same question")[:32],
        entry=_entry(),
        **scope,
    )

    hit_a = await semantic_cache.lookup(vector_a, **scope)
    assert hit_a is not None
    assert hit_a.score == pytest.approx(1.0, abs=1e-5)

    hit_b95 = await semantic_cache.lookup(vector_b95, **scope)
    assert hit_b95 is not None
    assert hit_b95.score == pytest.approx(0.95, abs=1e-4)

    strict_cache = SemanticCache(
        client=semantic_cache.client,
        collection=semantic_cache.collection,
        dim=4,
        threshold=0.97,
        ttl_s=3600,
        embed_model="fake-four-dimensional",
    )
    assert await strict_cache.lookup(vector_b95, **scope) is None

    for changed_scope in (
        {"team_id": 402},
        {"model_requested": "other-model"},
        {"context_hash": "context-b"},
        {"params_hash": "params-b"},
        {"num_sig": "numbers"},
    ):
        assert await semantic_cache.lookup(
            vector_a,
            **{**scope, **changed_scope},
        ) is None

    await semantic_cache.store(
        vector_a,
        text_hash=sha("same question")[:32],
        entry=_entry("Updated answer."),
        **scope,
    )
    assert await semantic_cache.count_team(scope["team_id"]) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_qdrant_ttl_cleanup_and_collection_dimension_validation(
    live_qdrant,
):
    collection = f"test_cache_{uuid.uuid4().hex}"
    cache = SemanticCache(
        client=live_qdrant,
        collection=collection,
        dim=4,
        threshold=0.92,
        ttl_s=1,
        embed_model="fake-four-dimensional",
    )
    scope = _scope(team_id=403)
    vector = [1.0, 0.0, 0.0, 0.0]

    try:
        await cache.store(
            vector,
            text_hash="ttl-test",
            entry=_entry(),
            **scope,
        )
        assert await cache.count_team(scope["team_id"]) == 1
        await asyncio.sleep(1.5)
        assert await cache.lookup(vector, **scope) is None
        await cache.purge_expired()
        assert await cache.count_team(scope["team_id"]) == 0
    finally:
        await live_qdrant.delete_collection(collection)

    wrong_dimension_collection = f"test_cache_{uuid.uuid4().hex}"
    try:
        await live_qdrant.create_collection(
            collection_name=wrong_dimension_collection,
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        )
        wrong_dimension_cache = SemanticCache(
            client=live_qdrant,
            collection=wrong_dimension_collection,
            dim=4,
            threshold=0.92,
            ttl_s=60,
            embed_model="fake-four-dimensional",
        )
        with pytest.raises(RuntimeError, match="has dim 3"):
            await wrong_dimension_cache.ensure_collection()
    finally:
        await live_qdrant.delete_collection(wrong_dimension_collection)


@pytest.mark.integration
def test_api_cache_flow_with_live_redis_and_qdrant(
    docker_inspectors,
    real_embedder,
    monkeypatch,
):
    inspector = docker_inspectors
    collection = f"test_cache_{uuid.uuid4().hex}"
    team_id = secrets.randbelow(1_000_000_000) + 1
    other_team_id = team_id + 1
    settings = Settings(
        _env_file=None,
        database_url="postgresql://test",
        redis_url=inspector.redis_url,
        qdrant_url=inspector.qdrant_url,
        qdrant_api_key=inspector.qdrant_api_key,
        admin_token="cache-integration-admin",
        cache_enabled=True,
        cache_ttl_seconds=3600,
        cache_max_temperature=0.7,
        cache_max_bytes=102_400,
        cache_lookup_timeout_s=10,
        cache_cleanup_interval_s=600,
        semantic_enabled=True,
        semantic_threshold=0.95,
        semantic_min_chars=8,
        semantic_max_chars=1000,
        embed_model=collection,
        fastembed_cache_dir=Settings().fastembed_cache_dir,
        pii_entities=["IN_PHONE"],
    )
    context = {
        "auth": AuthContext(
            team_id=team_id,
            key_id=team_id,
            team_name="cache-integration-team",
            rpm=600,
            burst=1000,
            budget_usd=Decimal("100"),
            pii_action="mask",
        )
    }
    provider_calls = []
    phone = "98765 43210"

    class FakeAnalyzer:
        def analyze(self, *, text, **kwargs):
            start = text.find(phone)
            if start < 0:
                return []
            return [
                SimpleNamespace(
                    entity_type="IN_PHONE",
                    start=start,
                    end=start + len(phone),
                    score=1.0,
                )
            ]

    class FakeProvider:
        def __init__(self, **kwargs):
            self.model_id = kwargs["model_id"]

        async def chat(self, payload):
            content = payload["messages"][-1]["content"]
            provider_calls.append(content)
            finish_reason = (
                "length"
                if content == "A deliberately truncated cache response."
                else "stop"
            )
            return {
                "id": f"chatcmpl-integration-{len(provider_calls)}",
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": f"Mock reply to: {content}",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

    monkeypatch.setattr(main, "Settings", lambda: settings)
    monkeypatch.setattr(
        main,
        "load_models_config",
        lambda _path: SimpleNamespace(providers={}),
    )
    monkeypatch.setattr(main, "validate_config", lambda config, _env: config)
    monkeypatch.setattr(admin_routes, "Settings", lambda: settings)
    monkeypatch.setattr(
        main.db,
        "create_pool",
        AsyncMock(
            return_value=SimpleNamespace(
                execute=AsyncMock(),
                close=AsyncMock(),
            )
        ),
    )
    monkeypatch.setattr(
        main.redis_client,
        "create_client",
        lambda url: redis_async.from_url(url, decode_responses=True),
    )
    monkeypatch.setattr(
        main,
        "FastEmbedEmbedder",
        Mock(return_value=real_embedder),
    )
    monkeypatch.setattr(main, "build_analyzer", Mock(return_value=FakeAnalyzer()))
    monkeypatch.setattr(main, "collection_name", lambda _model: collection)
    monkeypatch.setattr(main, "check_rate_limit", AsyncMock(return_value=(True, 0)))
    monkeypatch.setattr(
        main,
        "resolve",
        lambda _alias: (
            ProviderConfig(
                base_url="http://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "mock-1",
            "test-key",
        ),
    )
    monkeypatch.setattr(main, "OpenAICompatProvider", FakeProvider)
    monkeypatch.setattr(
        main,
        "cost_usd",
        lambda _alias, _prompt, _completion: Decimal("0.001"),
    )
    monkeypatch.setattr(
        main,
        "cleanup_loop",
        AsyncMock(),
    )
    main.app.dependency_overrides[authenticate] = lambda: context["auth"]

    def request(client, text, *, team=None, headers=None, temperature=None):
        if team is not None:
            context["auth"] = AuthContext(
                team_id=team,
                key_id=team,
                team_name="cache-integration-team",
                rpm=600,
                burst=1000,
                budget_usd=Decimal("100"),
                pii_action="mask",
            )
        body = {
            "model": "mock-fast",
            "messages": [{"role": "user", "content": text}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        return client.post(
            "/v1/chat/completions",
            json=body,
            headers=headers,
        )

    def exact_cache_key(text, team, temperature=None):
        payload = {
            "model": "mock-fast",
            "messages": [{"role": "user", "content": text}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        return exact_key(
            team,
            "mock-fast",
            payload["messages"],
            params_hash(payload),
        )

    def qdrant_team_count():
        return inspector.qdrant.count(
            collection_name=collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="team_id",
                        match=MatchValue(value=team_id),
                    )
                ]
            ),
            exact=True,
        ).count

    try:
        with TestClient(main.app) as client:
            france = "What is the capital of France?"
            first = request(client, france, team=team_id)
            assert first.status_code == 200
            assert first.headers["x-cache"] == "MISS"
            assert len(provider_calls) == 1

            france_key = exact_cache_key(france, team_id)
            initial_ttl = inspector.redis.ttl(france_key)
            assert 0 < initial_ttl <= settings.cache_ttl_seconds
            spend_key_name = spend_key(team_id, current_month())
            spend_after_miss = inspector.redis.get(spend_key_name)
            assert spend_after_miss is not None

            exact_hit = request(client, france)
            assert exact_hit.headers["x-cache"] == "HIT"
            assert exact_hit.headers["x-cache-type"] == "exact"
            assert len(provider_calls) == 1

            semantic_hit = request(
                client,
                "Which city is the capital of France?",
            )
            assert semantic_hit.headers["x-cache"] == "HIT"
            assert semantic_hit.headers["x-cache-type"] == "semantic"
            assert float(semantic_hit.headers["x-cache-score"]) >= 0.92
            assert (
                semantic_hit.json()["choices"][0]["message"]["content"]
                == first.json()["choices"][0]["message"]["content"]
            )
            assert len(provider_calls) == 1
            assert inspector.redis.get(spend_key_name) == spend_after_miss

            hard_negative = request(client, "What is the capital of Germany?")
            assert hard_negative.headers["x-cache"] == "MISS"
            assert len(provider_calls) == 2

            other_team = request(client, france, team=other_team_id)
            assert other_team.headers["x-cache"] == "MISS"
            assert len(provider_calls) == 3
            context["auth"] = AuthContext(
                team_id=team_id,
                key_id=team_id,
                team_name="cache-integration-team",
                rpm=600,
                burst=1000,
                budget_usd=Decimal("100"),
                pii_action="mask",
            )

            points_before_pii = qdrant_team_count()
            calls_before_pii = len(provider_calls)
            pii_request = "Call 98765 43210 about the capital of France"
            first_pii = request(client, pii_request)
            second_pii = request(client, pii_request)
            assert first_pii.headers["x-cache"] == "BYPASS"
            assert first_pii.headers["x-cache-bypass"] == "pii"
            assert second_pii.headers["x-cache"] == "BYPASS"
            assert second_pii.headers["x-cache-bypass"] == "pii"
            assert len(provider_calls) == calls_before_pii + 2
            assert provider_calls[-1] == "Call <PHONE_1> about the capital of France"
            assert qdrant_team_count() == points_before_pii

            points_before_bypass = qdrant_team_count()
            calls_before_temperature = len(provider_calls)
            hot_temperature = request(
                client,
                "A high-temperature cache bypass question",
                temperature=0.9,
            )
            assert hot_temperature.headers["x-cache"] == "BYPASS"
            assert hot_temperature.headers["x-cache-bypass"] == "temperature"
            assert len(provider_calls) == calls_before_temperature + 1

            no_store_text = "A no-store cache bypass question"
            no_store_key = exact_cache_key(no_store_text, team_id)
            calls_before_no_store = len(provider_calls)
            no_store_first = request(
                client,
                no_store_text,
                headers={"Cache-Control": "no-store"},
            )
            no_store_second = request(
                client,
                no_store_text,
                headers={"Cache-Control": "no-store"},
            )
            assert no_store_first.headers["x-cache-bypass"] == "no-store"
            assert no_store_second.headers["x-cache-bypass"] == "no-store"
            assert len(provider_calls) == calls_before_no_store + 2
            assert not inspector.redis.exists(no_store_key)
            assert qdrant_team_count() == points_before_bypass

            inspector.redis.expire(france_key, 2)
            calls_before_no_cache = len(provider_calls)
            no_cache = request(
                client,
                france,
                headers={"Cache-Control": "no-cache"},
            )
            assert no_cache.headers["x-cache"] == "BYPASS"
            assert no_cache.headers["x-cache-bypass"] == "no-cache"
            assert len(provider_calls) == calls_before_no_cache + 1
            refreshed_ttl = inspector.redis.ttl(france_key)
            assert 2 < refreshed_ttl <= settings.cache_ttl_seconds

            truncated = "A deliberately truncated cache response."
            calls_before_truncated = len(provider_calls)
            truncated_first = request(client, truncated)
            truncated_second = request(client, truncated)
            assert truncated_first.headers["x-cache"] == "MISS"
            assert truncated_second.headers["x-cache"] == "MISS"
            assert len(provider_calls) == calls_before_truncated + 2

            failing_query = "A semantic lookup backend failure test."
            calls_before_failure = len(provider_calls)
            with monkeypatch.context() as scoped:
                scoped.setattr(
                    main.app.state.cache.semantic.client,
                    "query_points",
                    AsyncMock(side_effect=RuntimeError("Qdrant is unavailable")),
                )
                fail_open = request(client, failing_query)
            assert fail_open.status_code == 200
            assert fail_open.headers["x-cache"] == "MISS"
            assert len(provider_calls) == calls_before_failure + 1

            assert client.delete(
                "/admin/cache",
                headers={"X-Admin-Token": settings.admin_token},
            ).status_code == 400
            assert client.delete(
                "/admin/cache?team_id=" + str(team_id),
                headers={"X-Admin-Token": "wrong-token"},
            ).status_code == 401
            purged = client.delete(
                f"/admin/cache?team_id={team_id}",
                headers={"X-Admin-Token": settings.admin_token},
            )
            assert purged.status_code == 200
            calls_before_purge_request = len(provider_calls)
            after_purge = request(client, france)
            assert after_purge.headers["x-cache"] == "MISS"
            assert len(provider_calls) == calls_before_purge_request + 1
    finally:
        main.app.dependency_overrides.pop(authenticate, None)
        for cleanup_team in (team_id, other_team_id):
            keys = list(
                inspector.redis.scan_iter(
                    match=f"cache:exact:{cleanup_team}:*",
                    count=500,
                )
            )
            if keys:
                inspector.redis.delete(*keys)
            inspector.redis.delete(
                spend_key(cleanup_team, current_month()),
                f"rl:{cleanup_team}",
            )
        if inspector.qdrant.collection_exists(collection):
            inspector.qdrant.delete_collection(collection)
