import asyncio
import json
import os
import random
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


NAME = os.getenv("MOCK_NAME", "mock")
app = FastAPI()
ctl: dict[str, Any] = {
    "delay_ms": int(os.getenv("MOCK_DELAY_MS", "200")),
    "fail_rate": float(os.getenv("MOCK_FAIL_RATE", "0")),
    "fail_status": 503,
    "retry_after": None,
    "stream_fail_after": None,
}
stats = {
    "requests": 0,
    "failures": 0,
    "streams": 0,
    "stream_disconnects": 0,
    "stream_aborts": 0,
}


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text", ""), str)
        )
    return ""


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


@app.get("/_control")
async def get_control() -> dict[str, Any]:
    return ctl


@app.post("/_control")
async def set_control(body: dict[str, Any]) -> dict[str, Any]:
    ctl.update({key: value for key, value in body.items() if key in ctl})
    return ctl


@app.get("/_stats")
async def get_stats() -> dict[str, int]:
    return stats


@app.post("/_reset_stats")
async def reset_stats() -> dict[str, int]:
    for key in stats:
        stats[key] = 0
    return stats


async def stream_gen(
    model: str,
    reply: str,
    usage: dict[str, int],
    include_usage: bool,
    fail_after: int | None,
):
    stats["streams"] += 1
    done = aborted = False
    base = {
        "id": "chatcmpl-mock-stream",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    try:
        words = reply.split(" ")
        for index, word in enumerate(words):
            if fail_after is not None and index >= fail_after:
                aborted = True
                stats["stream_aborts"] += 1
                raise RuntimeError("injected mid-stream failure")
            piece = word + (" " if index < len(words) - 1 else "")
            chunk = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.05)

        final_chunk = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        if include_usage:
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"
        done = True
    finally:
        if not done and not aborted:
            stats["stream_disconnects"] += 1


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    stats["requests"] += 1
    if random.random() < ctl["fail_rate"]:
        stats["failures"] += 1
        headers = (
            {"Retry-After": str(ctl["retry_after"])}
            if ctl["retry_after"]
            else {}
        )
        return JSONResponse(
            {
                "error": {
                    "message": f"[{NAME}] injected failure",
                    "type": "mock_error",
                }
            },
            status_code=int(ctl["fail_status"]),
            headers=headers,
        )

    await asyncio.sleep(ctl["delay_ms"] / 1000)
    messages = body.get("messages", [])
    user_texts = [
        text_of(message)
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    last = user_texts[-1] if user_texts else ""
    print(f"[mock-{NAME}] model={body.get('model')} last user message: {last}")

    reply = f"[{NAME}] Mock reply to: {last}"
    prompt = " ".join(text_of(message) for message in messages if isinstance(message, dict))
    usage = {
        "prompt_tokens": tokens(prompt),
        "completion_tokens": tokens(reply),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    model = body.get("model", "mock-1")

    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(
            stream_gen(
                model,
                reply,
                usage,
                include_usage,
                ctl["stream_fail_after"],
            ),
            media_type="text/event-stream",
        )

    return JSONResponse(
        {
            "id": "chatcmpl-mock-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": reply},
                }
            ],
            "usage": usage,
        }
    )
