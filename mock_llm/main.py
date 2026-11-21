import asyncio
import os
import random
import time
from typing import Any

from fastapi import FastAPI, HTTPException


def _read_mock_settings() -> tuple[int, float]:
    try:
        delay_ms = int(os.getenv("MOCK_DELAY_MS", "200"))
        fail_rate = float(os.getenv("MOCK_FAIL_RATE", "0"))
    except ValueError as exc:
        raise ValueError(
            "MOCK_DELAY_MS must be an integer and MOCK_FAIL_RATE must be a number."
        ) from exc

    if delay_ms < 0:
        raise ValueError("MOCK_DELAY_MS must be non-negative.")
    if not 0 <= fail_rate <= 1:
        raise ValueError("MOCK_FAIL_RATE must be between 0 and 1.")

    return delay_ms, fail_rate


MOCK_DELAY_MS, MOCK_FAIL_RATE = _read_mock_settings()

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(MOCK_DELAY_MS / 1000)

    if random.random() < MOCK_FAIL_RATE:
        raise HTTPException(status_code=500, detail="Mock provider failure.")

    messages = payload.get("messages", [])
    last_user_text = ""
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                last_user_text = message["content"]
                break

    reply = f"Mock reply to: {last_user_text}"
    print(f"[mock] last user message: {last_user_text}")
    prompt_tokens = len(last_user_text) // 4
    completion_tokens = len(reply) // 4

    return {
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mock-1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": reply},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
