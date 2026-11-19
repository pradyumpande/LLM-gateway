from contextlib import asynccontextmanager
import asyncio
import json
import logging
import math
from decimal import Decimal
from typing import Any, AsyncGenerator
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app import db, redis_client
from app.admin.routes import router as admin_router
from app.auth import AuthContext, authenticate
from app.budget import check_budget, current_month, record_spend
from app.config import Settings, UnknownModelAliasError, resolve
from app.errors import OpenAIAPIError, openai_error
from app.guardrails.pii import (
    DEFAULT_ENTITIES,
    InputTooLongError,
    build_analyzer,
    mask_messages,
)
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

    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        app.state.db = None
        app.state.redis = None
        app.state.analyzer = None
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
                entities=DEFAULT_ENTITIES,
            )
            app.state.db = db_pool
            app.state.redis = redis_connection
            app.state.analyzer = analyzer
            yield
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
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
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
) -> Any:
    cost_value = "unknown" if cost is None else format(cost, "f")
    print(
        json.dumps(
            {
                "team": auth.team_name,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_value,
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
    auth: AuthContext = Depends(authenticate),
) -> Any:
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

    if req.stream:
        return _finish_chat_response(
            _error_response(400, "Streaming is not supported yet."),
            response,
            auth,
            req.model,
        )

    try:
        masked_messages, pii_results = await asyncio.to_thread(
            mask_messages,
            req.messages,
            request.app.state.analyzer,
            request.app.state.settings.pii_score_threshold,
            request.app.state.settings.max_input_chars,
        )
    except InputTooLongError as exc:
        return _finish_chat_response(
            openai_error(
                413,
                str(exc),
                "invalid_request_error",
                "input_too_large",
            ),
            response,
            auth,
            req.model,
        )

    if auth.pii_action not in {"mask", "block", "log_only"}:
        return _finish_chat_response(
            openai_error(
                500,
                f"Unsupported PII action {auth.pii_action!r}.",
                "configuration_error",
                "invalid_pii_action",
            ),
            response,
            auth,
            req.model,
        )
    if auth.pii_action == "block" and pii_results:
        return _finish_chat_response(
            openai_error(
                400,
                "Request blocked because it contains personal information.",
                "guardrail_error",
                "pii_detected",
            ),
            response,
            auth,
            req.model,
        )
    if auth.pii_action == "log_only" and pii_results:
        logger.info(
            "PII detected; passing original content per team policy. "
            "request_id=%s entity_types=%s",
            request.state.request_id,
            sorted({result.entity_type for result in pii_results}),
        )

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

    provider_payload = req.model_dump(exclude_none=True)
    if auth.pii_action == "mask":
        provider_payload["messages"] = masked_messages

    try:
        result = await provider.chat(provider_payload)
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

    return _finish_chat_response(
        result,
        response,
        auth,
        req.model,
        prompt_tokens,
        completion_tokens,
        cost,
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