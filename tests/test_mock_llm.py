import asyncio

import httpx

import mock_llm.main as mock_main


def test_chat_echoes_last_user_message_and_computes_usage(monkeypatch):
    observed_delays = []

    async def record_sleep(delay):
        observed_delays.append(delay)

    monkeypatch.setattr(mock_main, "MOCK_DELAY_MS", 200)
    monkeypatch.setattr(mock_main, "MOCK_FAIL_RATE", 0)
    monkeypatch.setattr(mock_main.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(mock_main.random, "random", lambda: 0.5)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_main.app),
            base_url="http://mock",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-1",
                    "messages": [
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "response"},
                        {"role": "user", "content": "last user text"},
                    ],
                },
            )

    response = asyncio.run(run())
    body = response.json()
    reply = "Mock reply to: last user text"

    assert response.status_code == 200
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": reply,
    }
    assert body["usage"] == {
        "prompt_tokens": len("last user text") // 4,
        "completion_tokens": len(reply) // 4,
        "total_tokens": len("last user text") // 4 + len(reply) // 4,
    }
    assert observed_delays == [0.2]


def test_chat_returns_500_when_failure_is_selected(monkeypatch):
    async def no_wait(delay):
        return None

    monkeypatch.setattr(mock_main, "MOCK_DELAY_MS", 0)
    monkeypatch.setattr(mock_main, "MOCK_FAIL_RATE", 1)
    monkeypatch.setattr(mock_main.asyncio, "sleep", no_wait)
    monkeypatch.setattr(mock_main.random, "random", lambda: 0.5)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_main.app),
            base_url="http://mock",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

    response = asyncio.run(run())

    assert response.status_code == 500
