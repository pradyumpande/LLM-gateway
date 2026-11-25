import asyncio

import httpx
import pytest

import mock_llm.main as mock_main
from app.guardrails.pii import DEFAULT_ENTITIES, mask_messages


@pytest.fixture(autouse=True)
def reset_mock_state():
    mock_main.ctl.update(
        {
            "delay_ms": 200,
            "fail_rate": 0,
            "fail_status": 503,
            "retry_after": None,
            "stream_fail_after": None,
        }
    )
    for key in mock_main.stats:
        mock_main.stats[key] = 0
    yield
    mock_main.ctl.update(
        {
            "delay_ms": 200,
            "fail_rate": 0,
            "fail_status": 503,
            "retry_after": None,
            "stream_fail_after": None,
        }
    )
    for key in mock_main.stats:
        mock_main.stats[key] = 0


def test_chat_echoes_last_user_message_and_computes_usage(monkeypatch, capsys):
    observed_delays = []

    async def record_sleep(delay):
        observed_delays.append(delay)

    monkeypatch.setattr(mock_main, "NAME", "test")
    mock_main.ctl["delay_ms"] = 200
    mock_main.ctl["fail_rate"] = 0
    monkeypatch.setattr(mock_main.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(mock_main.random, "random", lambda: 0.5)

    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "last user text"},
    ]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_main.app),
            base_url="http://mock",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-1",
                    "messages": messages,
                },
            )

    response = asyncio.run(run())
    body = response.json()
    reply = "[test] Mock reply to: last user text"
    prompt = " ".join(message["content"] for message in messages)
    prompt_tokens = mock_main.tokens(prompt)

    assert response.status_code == 200
    assert body["model"] == "mock-1"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": reply,
    }
    assert body["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": mock_main.tokens(reply),
        "total_tokens": prompt_tokens + mock_main.tokens(reply),
    }
    assert observed_delays == [0.2]
    assert capsys.readouterr().out.strip() == (
        "[mock-test] model=mock-1 last user message: last user text"
    )
    assert mock_main.stats == {
        "requests": 1,
        "failures": 0,
        "streams": 0,
        "stream_disconnects": 0,
        "stream_aborts": 0,
    }


def test_chat_returns_configured_failure_before_delay(monkeypatch):
    observed_delays = []

    async def no_wait(delay):
        observed_delays.append(delay)
        return None

    mock_main.ctl.update(
        {"delay_ms": 1000, "fail_rate": 1, "fail_status": 503, "retry_after": 4}
    )
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

    assert response.status_code == 503
    assert response.headers["retry-after"] == "4"
    assert response.json()["error"]["message"] == "[mock] injected failure"
    assert observed_delays == []
    assert mock_main.stats["requests"] == 1
    assert mock_main.stats["failures"] == 1


def test_controls_reset_stats_and_stream_includes_usage():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_main.app),
            base_url="http://mock",
        ) as client:
            await client.post(
                "/_control",
                json={
                    "delay_ms": 0,
                    "fail_rate": 0,
                    "retry_after": 8,
                    "ignored": "value",
                },
            )
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-b",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hello there"}],
                },
            )
            stats = (await client.get("/_stats")).json()
            reset = (
                await client.post("/_reset_stats")
            ).json()
            return response, stats, reset

    response, stats, reset = asyncio.run(run())
    body = response.text

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: " in body
    assert '"content": "[mock] "' in body
    assert '"usage": {' in body
    assert body.rstrip().endswith("data: [DONE]")
    assert stats["requests"] == 1
    assert stats["streams"] == 1
    assert stats["failures"] == 0
    assert reset == {key: 0 for key in mock_main.stats}
    assert mock_main.ctl["retry_after"] == 8
    assert "ignored" not in mock_main.ctl


def test_stream_can_be_aborted_after_configured_chunk(monkeypatch):
    async def no_wait(delay):
        return None

    mock_main.ctl.update({"delay_ms": 0, "stream_fail_after": 1})
    monkeypatch.setattr(mock_main.asyncio, "sleep", no_wait)

    async def run():
        transport = httpx.ASGITransport(
            app=mock_main.app,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://mock",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello there"}],
                },
            )

    response = asyncio.run(run())

    assert response.status_code == 200
    assert "[DONE]" not in response.text
    assert mock_main.stats["streams"] == 1
    assert mock_main.stats["stream_aborts"] == 1
    assert mock_main.stats["stream_disconnects"] == 0


def test_chat_prints_all_masked_pii(analyzer, monkeypatch, capsys):
    async def no_wait(delay):
        return None

    mock_main.ctl["delay_ms"] = 0
    mock_main.ctl["fail_rate"] = 0
    monkeypatch.setattr(mock_main.asyncio, "sleep", no_wait)
    monkeypatch.setattr(mock_main.random, "random", lambda: 0.5)

    messages = [
        {
            "role": "user",
            "content": (
                "Phone 98765 43210, Aadhaar 2345 6789 0124, "
                "email asha@example.com, PAN ABCPE1234F, "
                "card 4111 1111 1111 1111"
            ),
        }
    ]
    masked, report = mask_messages(
        analyzer,
        messages,
        DEFAULT_ENTITIES,
        0.5,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_main.app),
            base_url="http://mock",
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json={"model": "mock-1", "messages": masked},
            )

    response = asyncio.run(run())

    assert response.status_code == 200
    assert report["total"] == 5
    assert capsys.readouterr().out.strip() == (
        "[mock-mock] model=mock-1 last user message: Phone <PHONE_1>, Aadhaar <AADHAAR_1>, "
        "email <EMAIL_1>, PAN <PAN_1>, card <CARD_1>"
    )
