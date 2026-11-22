import os
import time
from typing import Any

import httpx


KEY = os.environ.get("KEY")
if not KEY:
    raise SystemExit(
        'Set KEY to the generated gateway key (for example: $env:KEY="gw_...").'
    )


def ask(
    q: str,
    model: str = "mock-fast",
    headers: dict[str, str] | None = None,
    **params: Any,
) -> None:
    started = time.perf_counter()
    response = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", **(headers or {})},
        json={
            "model": model,
            "messages": [{"role": "user", "content": q}],
            **params,
        },
        timeout=30,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response_headers = response.headers
    if response.status_code == 200:
        try:
            body = response.json()
            reply = body["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError, ValueError):
            reply = response.text
    else:
        reply = response.text

    print(
        f"{response.status_code} {elapsed_ms:6.0f} ms  "
        f"cache={response_headers.get('x-cache')}/{response_headers.get('x-cache-type')} "
        f"score={response_headers.get('x-cache-score')} "
        f"bypass={response_headers.get('x-cache-bypass')} | {str(reply)[:60]}"
    )


ask("What is the capital of France?")
time.sleep(0.5)
ask("What is the capital of France?")
ask("Which city is the capital of France?")
ask("What is the capital of Germany?")
ask("What is 2+2?")
time.sleep(0.5)
ask("What is 2+3?")
ask("What is the capital of France?", headers={"Cache-Control": "no-cache"})
ask("What is the capital of France?", temperature=0.9)
ask("Call 98765 43210 about the capital of France")
time.sleep(0.5)
ask("Call 98765 43210 about the capital of France")
