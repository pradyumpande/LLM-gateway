import asyncio
import json

import httpx
import pytest

from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider


def test_chat_translates_model_and_uses_injected_client():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": []})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatProvider(
                base_url="https://api.groq.com/openai/v1/",
                model_id="llama-3.1-8b-instant",
                api_key="test-key",
                client=client,
            )
            payload = {
                "model": "groq-8b",
                "messages": [{"role": "user", "content": "hi"}],
            }

            result = await provider.chat(payload)

            assert result == {"choices": []}
            assert captured["url"] == (
                "https://api.groq.com/openai/v1/chat/completions"
            )
            assert captured["authorization"] == "Bearer test-key"
            assert captured["payload"]["model"] == "llama-3.1-8b-instant"
            assert payload["model"] == "groq-8b"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [
        (400, False),
        (401, False),
        (404, False),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_chat_maps_http_errors_to_provider_error(status_code, retryable):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "provider failure"}},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatProvider(
                base_url="https://provider.test/v1",
                model_id="test-model",
                api_key="test-key",
                client=client,
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.chat({"model": "alias"})

            assert exc_info.value.status_code == status_code
            assert exc_info.value.message == "provider failure"
            assert exc_info.value.retryable is retryable

    asyncio.run(run())


def test_chat_maps_timeout_to_retryable_gateway_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatProvider(
                base_url="https://provider.test/v1",
                model_id="test-model",
                api_key="test-key",
                client=client,
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.chat({"model": "alias"})

            assert exc_info.value.status_code == 504
            assert exc_info.value.retryable is True

    asyncio.run(run())


def test_chat_maps_connection_error_to_retryable_bad_gateway():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenAICompatProvider(
                base_url="https://provider.test/v1",
                model_id="test-model",
                api_key="test-key",
                client=client,
            )

            with pytest.raises(ProviderError) as exc_info:
                await provider.chat({"model": "alias"})

            assert exc_info.value.status_code == 502
            assert exc_info.value.message == "Could not connect to the provider."
            assert exc_info.value.retryable is True

    asyncio.run(run())
