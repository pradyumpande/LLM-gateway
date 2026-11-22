from contextlib import asynccontextmanager
import asyncio
import json
import logging
import math
import time
from decimal import Decimal
from typing import Any, AsyncGenerator
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient

from app import db, redis_client
from app.admin.routes import router as admin_router
from app.audit import audit
from app.auth import AuthContext, authenticate
from app.budget import check_budget, current_month, record_spend
from app.cache.embedder import FastEmbedEmbedder
from app.cache.policy import build_cache_info
from app.cache.semantic import SemanticCache, collection_name
from app.cache.service import CacheService, cleanup_loop, materialise
from app.config import Settings, UnknownModelAliasError, resolve
from app.errors import OpenAIAPIError, openai_error
from app.guardrails.pii import build_analyzer
from app.guardrails.pipeline import run_input_guardrails
from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider
from app.ratelimit import check_rate_limit
from app.usage.pricing import cost_usd


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict]
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = Settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set before the gateway can start.")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL must be set before the gateway can start.")

    db_pool = None
    redis_connection = None
    qdrant_client = None
    cleanup_task = None

    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        app.state.db = None
        app.state.redis = None
        app.state.analyzer = None
        app.state.cache = None
        app.state.settings = settings
        try:
            db_pool = await db.create_pool(settings.database_url)
            redis_connection = redis_client.create_client(settings.redis_url)
            await redis_connection.ping()

            analyzer = await asyncio.to_thread(build_analyzer)
            await asyncio.to_thread(
                analyzer.analyze,
                text="warm up 98765 43210",
                language="en",
                entities=settings.pii_entities,
            )
            app.state.db = db_pool
            app.state.redis = redis_connection
            app.state.analyzer = analyzer
            embedder = await asyncio.to_thread(
                FastEmbedEmbedder,
                settings.embed_model,
                settings.fastembed_cache_dir,
            )
            semantic = None
            if settings.semantic_enabled:
                qdrant_client = AsyncQdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=5,
                )
                semantic = SemanticCache(
                    qdrant_client,
                    collection_name(settings.embed_model),
                    embedder.dim,
                    settings.semantic_threshold,
                    settings.cache_ttl_seconds,
                    settings.embed_model,
                )
                try:
                    await semantic.ensure_collection()
                except Exception:
                    logger.error(
                        "Qdrant unavailable: semantic cache disabled; exact cache only",
                        exc_info=True,
                    )
                    semantic = None

            app.state.cache = CacheService(
                app.state.redis,
                semantic,
                embedder,
                ttl_s=settings.cache_ttl_seconds,
                max_bytes=settings.cache_max_bytes,
                lookup_timeout_s=settings.cache_lookup_timeout_s,
            )
            if semantic is not None:
                cleanup_task = asyncio.create_task(
                    cleanup_loop(semantic, settings.cache_cleanup_interval_s)
                )
            yield
        finally:
            try:
                if cleanup_task is not None:
                    cleanup_task.cancel()
                    try:
                        await cleanup_task
                    except asyncio.CancelledError:
                        pass
            finally:
                try:
                    if qdrant_client is not None:
                        await qdrant_client.close()
                finally:
                    try:
                        if redis_connection is not None:
                            await redis_connection.aclose()
                    finally:
                        if db_pool is not None:
                            await db_pool.close()


app = FastAPI(lifespan=lifespan)
app.include_router(admin_router)
logger = logging.getLogger(__name__)


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any):
    request_id = str(uuid4())
    request.state.request_id = request_id
    request.state.pii_found = False
    request.state.injection_flag = False
    request.state.pii_masked_count = 0
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    if request.state.pii_masked_count:
        response.headers["X-PII-Masked"] = str(request.state.pii_masked_count)
    if request.state.injection_flag:
        response.headers["X-Injection-Flag"] = "true"
    return response


def _error_response(status_code: int, message: str) -> JSONResponse:
    return openai_error(status_code, message, "provider_error", status_code)


def _finish_chat_response(
    result: Any,
    response: Response,
    auth: AuthContext,
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost: Decimal | None = Decimal(0),
    cache_status: str = "not_checked",
    saved_usd: Decimal | float = Decimal(0),
) -> Any:
    cost_value = "unknown" if cost is None else format(cost, "f")
    saved_value = format(Decimal(str(saved_usd)), "f")
    print(
        json.dumps(
            {
                "team": auth.team_name,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_value,
                "cache_status": cache_status,
                "saved_usd": saved_value,
            },
            separators=(",", ":"),
        )
    )
    target_response = result if isinstance(result, Response) else response
    target_response.headers["X-Team"] = auth.team_name
    target_response.headers["X-Cost"] = cost_value
    return result


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    req: ChatRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(authenticate),
) -> Any:
    t0 = time.perf_counter()
    response.headers["X-RateLimit-Limit"] = str(auth.rpm)
    try:
        allowed, retry_after_ms = await check_rate_limit(request.app.state.redis, auth)
    except Exception:
        logger.exception(
            "Rate limiter unavailable; allowing request (fail-open). request_id=%s",
            request.state.request_id,
        )
    else:
        if not allowed:
            rate_limit_response = openai_error(
                429,
                "Rate limit exceeded.",
                "rate_limit_error",
                "rate_limit_exceeded",
            )
            rate_limit_response.headers["Retry-After"] = str(
                math.ceil(retry_after_ms / 1000)
            )
            rate_limit_response.headers["X-RateLimit-Limit"] = str(auth.rpm)
            return _finish_chat_response(
                rate_limit_response, response, auth, req.model
            )

    month = current_month()
    budget_available, _, _ = await check_budget(
        request.app.state.redis,
        auth,
        month,
    )
    if not budget_available:
        return _finish_chat_response(
            openai_error(
                429,
                "Monthly budget exceeded for this team.",
                "budget_error",
                "budget_exceeded",
            ),
            response,
            auth,
            req.model,
        )

    payload = req.model_dump(exclude_none=True)
    guard = await run_input_guardrails(payload, auth, request.app.state)
    request.state.pii_found = guard.pii_total > 0
    request.state.injection_flag = guard.injection_flagged
    if auth.pii_action == "mask" and guard.pii_total:
        request.state.pii_masked_count = guard.pii_total
    if auth.pii_action == "log_only" and guard.pii_total:
        logger.info(
            "PII detected; passing original content per team policy. "
            "request_id=%s counts=%s",
            request.state.request_id,
            guard.pii_counts,
        )

    if guard.injection_flagged:
        logger.warning(
            "Prompt-injection signal request_id=%s score=%s rules=%s",
            request.state.request_id,
            guard.injection_score,
            guard.injection_rules,
        )
    if guard.block:
        if (
            guard.block_status is None
            or guard.block_message is None
            or guard.block_code is None
        ):
            raise RuntimeError("Blocked guardrail result is missing response details.")
        await audit(
            request.app.state.db,
            f"team:{auth.team_id}",
            "guardrail_block",
            {
                "code": guard.block_code,
                "pii": guard.pii_counts,
                "injection_rules": guard.injection_rules,
            },
        )
        return _finish_chat_response(
            openai_error(
                guard.block_status,
                guard.block_message,
                "invalid_request_error",
                guard.block_code,
            ),
            response,
            auth,
            req.model,
        )

    if req.stream:
        return _finish_chat_response(
            _error_response(400, "Streaming is not supported yet."),
            response,
            auth,
            req.model,
        )

    payload = guard.payload

    settings = request.app.state.settings
    cache = request.app.state.cache
    info = build_cache_info(
        payload,
        auth.team_id,
        enabled=settings.cache_enabled,
        max_temperature=settings.cache_max_temperature,
        pii_found=request.state.pii_found,
        injection_flagged=request.state.injection_flag,
        cache_control=request.headers.get("cache-control"),
        semantic_enabled=settings.semantic_enabled and cache.semantic is not None,
        sem_min=settings.semantic_min_chars,
        sem_max=settings.semantic_max_chars,
    )
    lookup = await cache.lookup(info)
    if lookup.kind is not None and lookup.entry is not None:
        body = materialise(lookup.entry)
        saved = float(lookup.entry["cost_usd"])
        request.state.cache_status = f"{lookup.kind}_hit"
        request.state.saved_usd = saved
        headers = dict(response.headers)
        headers.update(
            {
                "X-Cache": "HIT",
                "X-Cache-Type": lookup.kind,
                "X-Cache-Age": str(lookup.age_s or 0),
                "X-Cost": "0",
                "X-Saved-Cost": f"{saved:.6f}",
            }
        )
        if lookup.score is not None:
            headers["X-Cache-Score"] = f"{lookup.score:.4f}"
        print(
            f"req={request.state.request_id} cache={lookup.kind} "
            f"latency_ms={(time.perf_counter() - t0) * 1000:.0f}"
        )
        cached_usage = body.get("usage")
        return _finish_chat_response(
            JSONResponse(body, headers=headers),
            response,
            auth,
            req.model,
            cached_usage.get("prompt_tokens") if isinstance(cached_usage, dict) else None,
            cached_usage.get("completion_tokens") if isinstance(cached_usage, dict) else None,
            Decimal(0),
            cache_status=request.state.cache_status,
            saved_usd=saved,
        )

    request.state.cache_status = "miss"
    request.state.saved_usd = 0.0
    response.headers["X-Cache"] = "BYPASS" if info.bypass_reason else "MISS"
    if info.bypass_reason:
        response.headers["X-Cache-Bypass"] = info.bypass_reason

    try:
        provider_config, model_id, api_key = resolve(req.model)
    except UnknownModelAliasError as exc:
        return _finish_chat_response(
            _error_response(404, str(exc)), response, auth, req.model
        )
    except ValueError as exc:
        return _finish_chat_response(
            _error_response(500, str(exc)), response, auth, req.model
        )

    try:
        cost_usd(req.model, 0, 0)
    except ValueError as exc:
        return _finish_chat_response(
            _error_response(503, str(exc)),
            response,
            auth,
            req.model,
            cost=None,
        )

    provider = OpenAICompatProvider(
        base_url=provider_config.base_url,
        model_id=model_id,
        api_key=api_key,
        client=request.app.state.http_client,
    )

    try:
        result = await provider.chat(payload)
    except ProviderError as exc:
        return _finish_chat_response(
            _error_response(exc.status_code, exc.message),
            response,
            auth,
            req.model,
            cost=None,
        )

    usage = result.get("usage")
    if not isinstance(usage, dict):
        return _finish_chat_response(
            _error_response(
                502,
                "Provider response is missing token usage required for budget accounting.",
            ),
            response,
            auth,
            req.model,
            cost=None,
        )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or prompt_tokens < 0
        or completion_tokens < 0
    ):
        return _finish_chat_response(
            _error_response(
                502,
                "Provider response contains invalid token usage for budget accounting.",
            ),
            response,
            auth,
            req.model,
            cost=None,
        )

    cost = cost_usd(req.model, prompt_tokens, completion_tokens)
    await record_spend(
        request.app.state.redis,
        request.app.state.db,
        auth,
        month,
        cost,
    )

    background_tasks.add_task(
        cache.store,
        info,
        lookup.vector,
        result,
        float(cost),
        req.model,
        provider_config.base_url,
    )
    return _finish_chat_response(
        result,
        response,
        auth,
        req.model,
        prompt_tokens,
        completion_tokens,
        cost,
        cache_status=request.state.cache_status,
        saved_usd=request.state.saved_usd,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(OpenAIAPIError)
async def handle_openai_error(
    request: Request,
    exc: OpenAIAPIError,
) -> JSONResponse:
    return openai_error(exc.status_code, exc.message, exc.type, exc.code)