from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.config import UnknownModelAliasError, resolve
from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict]
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "provider_error",
                "code": status_code,
            }
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, req: ChatRequest) -> Any:
    if req.stream:
        return _error_response(400, "Streaming is not supported yet.")

    try:
        provider_config, model_id, api_key = resolve(req.model)
    except UnknownModelAliasError as exc:
        return _error_response(404, str(exc))
    except ValueError as exc:
        return _error_response(500, str(exc))

    provider = OpenAICompatProvider(
        base_url=provider_config.base_url,
        model_id=model_id,
        api_key=api_key,
        client=request.app.state.http_client,
    )

    try:
        result = await provider.chat(req.model_dump(exclude_none=True))
    except ProviderError as exc:
        return _error_response(exc.status_code, exc.message)

    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}