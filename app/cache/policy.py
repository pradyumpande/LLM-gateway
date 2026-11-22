from dataclasses import dataclass
from typing import Any

from app.cache import keys


TOOL_KEYS = ("tools", "functions", "tool_choice", "function_call")


@dataclass
class CacheInfo:
    team_id: int
    model_requested: str = ""
    read: bool = False
    write: bool = False
    bypass_reason: str | None = None
    exact_key: str | None = None
    params_hash: str | None = None
    context_hash: str | None = None
    num_sig: str | None = None
    semantic_text: str | None = None
    text_hash: str | None = None


def build_cache_info(
    payload: dict[str, Any],
    team_id: int,
    *,
    enabled: bool,
    max_temperature: float,
    pii_found: bool,
    injection_flagged: bool,
    cache_control: str | None,
    semantic_enabled: bool,
    sem_min: int,
    sem_max: int,
) -> CacheInfo:
    def bypass(reason: str) -> CacheInfo:
        return CacheInfo(team_id=team_id, bypass_reason=reason)

    if not enabled:
        return bypass("disabled")
    if payload.get("stream"):
        return bypass("stream")
    if any(key in payload for key in TOOL_KEYS):
        return bypass("tools")
    n = payload.get("n")
    if isinstance(n, int) and n > 1:
        return bypass("n_gt_1")
    temperature = payload.get("temperature")
    if (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and temperature > max_temperature
    ):
        return bypass("temperature")
    if pii_found:
        return bypass("pii")
    if injection_flagged:
        return bypass("injection_flag")

    messages = payload.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(not isinstance(message, dict) for message in messages)
    ):
        return bypass("no_messages")

    tokens = {
        token.strip().lower()
        for token in (cache_control or "").split(",")
        if token.strip()
    }
    if "no-store" in tokens:
        return bypass("no-store")

    model_requested = str(payload.get("model", ""))
    p_hash = keys.params_hash(payload)
    info = CacheInfo(
        team_id=team_id,
        model_requested=model_requested,
        read="no-cache" not in tokens,
        write=True,
        bypass_reason="no-cache" if "no-cache" in tokens else None,
        exact_key=keys.exact_key(team_id, model_requested, messages, p_hash),
        params_hash=p_hash,
        context_hash=keys.context_hash(messages),
    )

    last_message = messages[-1]
    if (
        semantic_enabled
        and last_message.get("role") == "user"
        and isinstance(last_message.get("content"), str)
    ):
        text = last_message["content"].strip()
        if sem_min <= len(text) <= sem_max:
            info.semantic_text = text
            info.num_sig = keys.num_sig(text)
            info.text_hash = keys.sha(text)[:32]
    return info
