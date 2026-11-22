import asyncio
import copy
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.cache.policy import CacheInfo
from app.cache.semantic import Hit, SemanticCache

log = logging.getLogger("gateway.cache")


@dataclass
class LookupResult:
    kind: str | None = None
    entry: dict[str, Any] | None = None
    score: float | None = None
    age_s: int | None = None
    vector: list[float] | None = None


def cacheable_response(response: dict[str, Any]) -> bool:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        return (
            choice.get("finish_reason") == "stop"
            and isinstance(message.get("content"), str)
            and bool(message["content"].strip())
            and not message.get("tool_calls")
        )
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def valid_cache_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    cost = entry.get("cost_usd")
    created = entry.get("created_at")
    return (
        isinstance(entry.get("response"), dict)
        and cacheable_response(entry["response"])
        and isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(float(cost))
        and isinstance(created, int)
        and not isinstance(created, bool)
        and isinstance(entry.get("model_alias"), str)
        and isinstance(entry.get("provider"), str)
    )


def materialise(entry: dict[str, Any]) -> dict[str, Any]:
    response = copy.deepcopy(entry["response"])
    response["id"] = f"chatcmpl-cache-{uuid.uuid4().hex[:24]}"
    response["created"] = int(time.time())
    return response


class CacheService:
    def __init__(
        self,
        redis: Any,
        semantic: SemanticCache | None,
        embedder: Any,
        *,
        ttl_s: int,
        max_bytes: int,
        lookup_timeout_s: float,
    ) -> None:
        self.redis = redis
        self.semantic = semantic
        self.embedder = embedder
        self.ttl_s = ttl_s
        self.max_bytes = max_bytes
        self.lookup_timeout_s = lookup_timeout_s

    async def lookup(self, info: CacheInfo) -> LookupResult:
        if not info.read or not info.exact_key:
            return LookupResult()

        try:
            raw = await asyncio.wait_for(
                self.redis.get(info.exact_key),
                timeout=self.lookup_timeout_s,
            )
            if raw:
                entry = json.loads(raw)
                if not valid_cache_entry(entry):
                    raise ValueError("malformed or incomplete cache entry")
                return LookupResult(
                    kind="exact",
                    entry=entry,
                    age_s=int(time.time()) - entry["created_at"],
                )
        except Exception:
            log.warning("exact cache lookup failed", exc_info=True)

        if (
            self.semantic is None
            or info.semantic_text is None
            or info.params_hash is None
            or info.context_hash is None
            or info.num_sig is None
        ):
            return LookupResult()
        try:
            return await asyncio.wait_for(
                self._semantic(info),
                timeout=self.lookup_timeout_s,
            )
        except Exception:
            log.warning("semantic cache lookup failed", exc_info=True)
            return LookupResult()

    async def _semantic(self, info: CacheInfo) -> LookupResult:
        if (
            info.semantic_text is None
            or info.params_hash is None
            or info.context_hash is None
            or info.num_sig is None
        ):
            return LookupResult()
        vector = await self.embedder.embed(info.semantic_text)
        hit: Hit | None = await self.semantic.lookup(
            vector,
            team_id=info.team_id,
            model_requested=info.model_requested,
            context_hash=info.context_hash,
            params_hash=info.params_hash,
            num_sig=info.num_sig,
        )
        if hit is None:
            return LookupResult(vector=vector)
        if not valid_cache_entry(hit.payload):
            return LookupResult(vector=vector)
        return LookupResult(
            kind="semantic",
            entry=hit.payload,
            score=hit.score,
            age_s=int(time.time()) - hit.payload["created_at"],
            vector=vector,
        )

    async def store(
        self,
        info: CacheInfo,
        vector: list[float] | None,
        response: dict[str, Any],
        cost_usd: float,
        model_alias: str,
        provider: str,
    ) -> None:
        if not info.write or not info.exact_key or not cacheable_response(response):
            return

        entry = {
            "response": response,
            "cost_usd": cost_usd,
            "model_alias": model_alias,
            "provider": provider,
            "created_at": int(time.time()),
        }
        try:
            blob = json.dumps(entry)
            if len(blob.encode("utf-8")) > self.max_bytes:
                return
        except (TypeError, ValueError):
            log.warning("cache response could not be serialized", exc_info=True)
            return

        try:
            await asyncio.wait_for(
                self.redis.set(info.exact_key, blob, ex=self.ttl_s),
                timeout=self.lookup_timeout_s,
            )
        except Exception:
            log.warning("exact cache store failed", exc_info=True)

        if (
            self.semantic is None
            or info.semantic_text is None
            or info.text_hash is None
            or info.params_hash is None
            or info.context_hash is None
            or info.num_sig is None
        ):
            return
        try:
            if vector is None:
                vector = await asyncio.wait_for(
                    self.embedder.embed(info.semantic_text),
                    timeout=self.lookup_timeout_s,
                )
            await asyncio.wait_for(
                self.semantic.store(
                    vector,
                    text_hash=info.text_hash,
                    entry=entry,
                    team_id=info.team_id,
                    model_requested=info.model_requested,
                    context_hash=info.context_hash,
                    params_hash=info.params_hash,
                    num_sig=info.num_sig,
                ),
                timeout=self.lookup_timeout_s,
            )
        except Exception:
            log.warning("semantic cache store failed", exc_info=True)


async def cleanup_loop(semantic: SemanticCache, interval_s: int) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await semantic.purge_expired()
        except Exception:
            log.warning("cache cleanup failed", exc_info=True)
