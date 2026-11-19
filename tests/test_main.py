import asyncio
import json
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.auth import AuthContext, authenticate
from app.config import ProviderConfig, UnknownModelAliasError


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    db_pool = SimpleNamespace(close=AsyncMock())
    redis_client = SimpleNamespace(ping=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(
        main,
        "Settings",
        lambda: SimpleNamespace(
            database_url="postgresql://test",
            redis_url="redis://test",
            pii_score_threshold=0.5,
            max_input_chars=200_000,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_analyzer",
        Mock(return_value=SimpleNamespace(analyze=Mock(return_value=[]))),
    )
    monkeypatch.setattr(
        main.db,
        "create_pool",
        AsyncMock(return_value=db_pool),
    )
    monkeypatch.setattr(
        main.redis_client,
        "create_client",
        Mock(return_value=redis_client),
    )
    monkeypatch.setattr(
        main,
        "check_rate_limit",
        AsyncMock(return_value=(True, 0)),
    )
    monkeypatch.setattr(
        main,
        "check_budget",
        AsyncMock(return_value=(True, "spend:1:2026-11", 0.0)),
    )
    monkeypatch.setattr(main, "current_month", lambda: "2026-11")
    monkeypatch.setattr(main, "cost_usd", _test_cost)
    monkeypatch.setattr(main, "record_spend", AsyncMock(return_value=0.006))
    main.app.dependency_overrides[authenticate] = lambda: AuthContext(
        team_id=1,
        key_id=1,
        team_name="test-team",
        rpm=60,
        burst=10,
        budget_usd=10,
        pii_action="mask",
    )
    yield
    main.app.dependency_overrides.pop(authenticate, None)


def _test_cost(alias, prompt_tokens, completion_tokens):
    return Decimal(prompt_tokens + completion_tokens) / Decimal(1000)


def run_request(method, path, *, json=None, configure_app=None):
    async def run():
        async with main.app.router.lifespan_context(main.app):
            replacement_client = None
            if configure_app is not None:
                replacement_client = configure_app(main.app)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main.app),
                    base_url="http://test",
                ) as client:
                    return await client.request(method, path, json=json)
            finally:
                if replacement_client is not None:
                    await replacement_client.aclose()

    return asyncio.run(run())


def test_health_returns_status_and_request_id():
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    request_id = uuid.UUID(response.headers["x-request-id"])
    assert request_id.version == 4


def test_lifespan_creates_and_closes_shared_http_client():
    async def run():
        async with main.app.router.lifespan_context(main.app):
            client = main.app.state.http_client
            assert isinstance(client, httpx.AsyncClient)
            assert not client.is_closed

        assert client.is_closed

    asyncio.run(run())


def test_lifespan_builds_stores_and_warms_analyzer(monkeypatch):
    settings = SimpleNamespace(
        database_url="postgresql://test",
        redis_url="redis://test",
        pii_score_threshold=0.5,
        max_input_chars=200_000,
    )
    analyzer = SimpleNamespace(analyze=Mock(return_value=[]))
    build = Mock(return_value=analyzer)
    monkeypatch.setattr(main, "Settings", lambda: settings)
    monkeypatch.setattr(main, "build_analyzer", build)

    with TestClient(main.app):
        assert main.app.state.analyzer is analyzer
        assert main.app.state.settings is settings

    build.assert_called_once_with()
    analyzer.analyze.assert_called_once_with(
        text="warm up 98765 43210",
        language="en",
        entities=main.DEFAULT_ENTITIES,
    )


def test_lifespan_connects_and_closes_database_and_redis(monkeypatch):
    settings = SimpleNamespace(
        database_url="postgresql://test",
        redis_url="redis://test",
        pii_score_threshold=0.5,
        max_input_chars=200_000,
    )
    monkeypatch.setattr(main, "Settings", lambda: settings)

    db_pool = SimpleNamespace(close=AsyncMock())
    redis_client = SimpleNamespace(ping=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(
        main,
        "build_analyzer",
        Mock(return_value=SimpleNamespace(analyze=Mock(return_value=[]))),
    )
    create_pool = AsyncMock(return_value=db_pool)
    create_client = Mock(return_value=redis_client)
    monkeypatch.setattr(main.db, "create_pool", create_pool)
    monkeypatch.setattr(main.redis_client, "create_client", create_client)

    with TestClient(main.app):
        create_pool.assert_awaited_once_with("postgresql://test")
        create_client.assert_called_once_with("redis://test")
        redis_client.ping.assert_awaited_once()
        assert main.app.state.db is db_pool
        assert main.app.state.redis is redis_client

    redis_client.aclose.assert_awaited_once()
    db_pool.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("database_url", "redis_url", "missing_setting"),
    [
        (None, "redis://test", "DATABASE_URL"),
        ("postgresql://test", None, "REDIS_URL"),
    ],
)
def test_lifespan_fails_clearly_when_url_is_missing(
    monkeypatch,
    database_url,
    redis_url,
    missing_setting,
):
    monkeypatch.setattr(
        main,
        "Settings",
        lambda: SimpleNamespace(
            database_url=database_url,
            redis_url=redis_url,
        ),
    )

    with pytest.raises(RuntimeError, match=f"{missing_setting} must be set"):
        with TestClient(main.app):
            pass


def test_chat_passes_through_extra_fields_and_translates_model(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "real-model-id",
            "test-key",
        ),
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.read())
        captured["authorization"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    def configure_app(app):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        app.state.http_client = client
        return client

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "alias",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "tools": [{"type": "function"}],
        },
        configure_app=configure_app,
    )

    assert response.status_code == 200
    assert response.json() == {
        "choices": [],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    assert response.headers["x-request-id"]
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "real-model-id"
    assert captured["payload"]["temperature"] == 0.3
    assert captured["payload"]["tools"] == [{"type": "function"}]
    assert response.headers["x-ratelimit-limit"] == "60"
    assert response.headers["x-team"] == "test-team"
    assert response.headers["x-cost"] == "0.006"
    log_entry = json.loads(capsys.readouterr().out)
    assert log_entry == {
        "team": "test-team",
        "model": "alias",
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "cost_usd": "0.006",
    }
    main.record_spend.assert_awaited_once_with(
        main.app.state.redis,
        main.app.state.db,
        AuthContext(
            team_id=1,
            key_id=1,
            team_name="test-team",
            rpm=60,
            burst=10,
            budget_usd=10,
            pii_action="mask",
        ),
        "2026-11",
        Decimal("0.006"),
    )


def test_chat_masks_pii_before_calling_provider(monkeypatch):
    from types import SimpleNamespace

    from app.auth import AuthContext

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
                    score=0.6,
                )
            ]

    monkeypatch.setattr(
        main,
        "build_analyzer",
        Mock(return_value=FakeAnalyzer()),
    )
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "real-model-id",
            "test-key",
        ),
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    def configure_app(app):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.state.http_client = client
        return client

    original_messages = [{"role": "user", "content": f"Call {phone}"}]
    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "alias", "messages": original_messages},
        configure_app=configure_app,
    )

    assert response.status_code == 200
    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "Call <PHONE>"}
    ]
    assert original_messages == [{"role": "user", "content": f"Call {phone}"}]


def test_block_pii_action_rejects_before_model_resolution(monkeypatch):
    match_text = "98765 43210"

    class FakeAnalyzer:
        def analyze(self, *, text, **kwargs):
            start = text.find(match_text)
            if start < 0:
                return []
            return [
                SimpleNamespace(
                    entity_type="IN_PHONE",
                    start=start,
                    end=start + len(match_text),
                    score=0.6,
                )
            ]

    monkeypatch.setattr(
        main,
        "build_analyzer",
        Mock(return_value=FakeAnalyzer()),
    )
    monkeypatch.setattr(
        main,
        "resolve",
        Mock(side_effect=AssertionError("Blocked PII must not resolve a model")),
    )
    main.app.dependency_overrides[authenticate] = lambda: AuthContext(
        team_id=1,
        key_id=1,
        team_name="test-team",
        rpm=60,
        burst=10,
        budget_usd=10,
        pii_action="block",
    )

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-fast",
            "messages": [{"role": "user", "content": f"Call {match_text}"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "pii_detected"
    main.resolve.assert_not_called()


def test_budget_exceeded_returns_openai_429_before_provider_call(monkeypatch):
    monkeypatch.setattr(
        main,
        "check_budget",
        AsyncMock(return_value=(False, "spend:1:2026-11", 10.0)),
    )
    resolve = Mock(side_effect=AssertionError("Model resolution must not run"))
    monkeypatch.setattr(main, "resolve", resolve)
    provider_chat = AsyncMock()

    class FakeProvider:
        def __init__(self, **kwargs):
            self.chat = provider_chat

    monkeypatch.setattr(main, "OpenAICompatProvider", FakeProvider)
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "real-model-id",
            "test-key",
        ),
    )

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "alias", "messages": []},
    )

    assert response.status_code == 429
    assert response.json()["error"] == {
        "message": "Monthly budget exceeded for this team.",
        "type": "budget_error",
        "code": "budget_exceeded",
    }
    assert response.headers["x-team"] == "test-team"
    assert response.headers["x-cost"] == "0"
    resolve.assert_not_called()
    provider_chat.assert_not_awaited()


def test_unpriced_model_returns_503_before_provider_call(monkeypatch):
    from app.usage.pricing import cost_usd

    monkeypatch.setattr(main, "cost_usd", cost_usd)
    provider_chat = AsyncMock()

    class FakeProvider:
        def __init__(self, **kwargs):
            self.chat = provider_chat

    monkeypatch.setattr(main, "OpenAICompatProvider", FakeProvider)
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "real-model-id",
            "test-key",
        ),
    )

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "groq-8b", "messages": []},
    )

    assert response.status_code == 503
    assert "No published token price" in response.json()["error"]["message"]
    provider_chat.assert_not_awaited()


def test_rate_limited_request_returns_retry_headers_and_openai_error(monkeypatch):
    monkeypatch.setattr(
        main,
        "check_rate_limit",
        AsyncMock(return_value=(False, 1501)),
    )

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "alias", "messages": []},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"
    assert response.headers["x-ratelimit-limit"] == "60"
    assert response.headers["x-team"] == "test-team"
    assert response.headers["x-cost"] == "0"
    assert response.json()["error"] == {
        "message": "Rate limit exceeded.",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
    }


def test_rate_limiter_failure_fails_open_and_logs_request_id(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "real-model-id",
            "test-key",
        ),
    )
    monkeypatch.setattr(
        main,
        "check_rate_limit",
        AsyncMock(side_effect=RuntimeError("Redis unavailable")),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    def configure_app(app):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.state.http_client = client
        return client

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "alias", "messages": []},
        configure_app=configure_app,
    )

    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit"] == "60"
    assert response.headers["x-request-id"] in caplog.text


def test_streaming_is_rejected_with_clear_400():
    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "alias", "messages": [], "stream": True},
    )

    assert response.status_code == 400
    assert "streaming is not supported" in response.json()["error"]["message"].lower()


def test_unknown_model_alias_returns_openai_shaped_404(monkeypatch):
    def unknown_alias(alias):
        raise UnknownModelAliasError(f"Unknown model alias {alias!r}.")

    monkeypatch.setattr(main, "resolve", unknown_alias)
    with TestClient(main.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": []},
        )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "provider_error"
    assert response.json()["error"]["code"] == 404
    assert response.headers["x-request-id"]


def test_provider_error_is_returned_in_openai_shape(monkeypatch):
    monkeypatch.setattr(
        main,
        "resolve",
        lambda alias: (
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_API_KEY",
            ),
            "real-model-id",
            "test-key",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
        )

    def configure_app(app):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        app.state.http_client = client
        return client

    response = run_request(
        "POST",
        "/v1/chat/completions",
        json={"model": "alias", "messages": []},
        configure_app=configure_app,
    )

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "message": "rate limited",
            "type": "provider_error",
            "code": 429,
        }
    }
