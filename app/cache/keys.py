import hashlib
import json
import re
from typing import Any


_NON_PARAMS = {"model", "messages", "stream", "stream_options", "user", "metadata"}
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def canonical(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        normalized_message = dict(message)
        if isinstance(normalized_message.get("content"), str):
            normalized_message["content"] = normalized_message["content"].strip()
        normalized.append(normalized_message)
    return normalized


def params_hash(payload: dict[str, Any]) -> str:
    params = {key: value for key, value in payload.items() if key not in _NON_PARAMS}
    return sha(canonical(params))[:32]


def context_hash(messages: list[dict[str, Any]]) -> str:
    return sha(canonical(normalise_messages(messages[:-1])))[:32]


def exact_key(
    team_id: int,
    model_requested: str,
    messages: list[dict[str, Any]],
    p_hash: str,
) -> str:
    body = canonical(
        {
            "m": model_requested,
            "msgs": normalise_messages(messages),
            "p": p_hash,
        }
    )
    return f"cache:exact:{team_id}:{sha(body)}"


def num_sig(text: str) -> str:
    numbers = _NUM_RE.findall(text)
    return sha("|".join(numbers))[:16] if numbers else "none"


def semantic_scope(
    team_id: int,
    model_requested: str,
    p_hash: str,
    ctx_hash: str,
    number_signature: str,
) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "model_requested": model_requested,
        "params_hash": p_hash,
        "context_hash": ctx_hash,
        "num_sig": number_signature,
    }
