import asyncio
import json
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import ProviderConfig, UnknownModelAliasError


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


def test_chat_passes_through_extra_fields_and_translates_model(monkeypatch):
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
        return httpx.Response(200, json={"choices": []})

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
    assert response.json() == {"choices": []}
    assert response.headers["x-request-id"]
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "real-model-id"
    assert captured["payload"]["temperature"] == 0.3
    assert captured["payload"]["tools"] == [{"type": "function"}]


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
