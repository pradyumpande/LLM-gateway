import hashlib
import secrets
from decimal import Decimal

from fastapi import Request
from pydantic import BaseModel, ValidationError

from app.errors import OpenAIAPIError


def generate_key() -> str:
    return "gw_" + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def key_prefix(key: str) -> str:
    return key[:8]


class AuthContext(BaseModel):
    team_id: int
    key_id: int
    team_name: str
    rpm: int
    burst: int
    budget_usd: Decimal
    pii_action: str


def _unauthorized() -> OpenAIAPIError:
    return OpenAIAPIError(
        status=401,
        message="Invalid or missing API key.",
        type="authentication_error",
        code="invalid_api_key",
    )


async def authenticate(request: Request) -> AuthContext:
    authorization = request.headers.get("authorization")
    parts = authorization.split() if authorization is not None else []
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise _unauthorized()

    hashed_key = hash_key(parts[1])
    cache_key = f"auth:{hashed_key}"
    redis = request.app.state.redis
    cached = await redis.get(cache_key)
    if cached == "invalid" or cached == b"invalid":
        raise _unauthorized()

    if cached is not None:
        try:
            return AuthContext.model_validate_json(cached)
        except ValidationError:
            await redis.delete(cache_key)

    row = await request.app.state.db.fetchrow(
        """
        SELECT api_keys.team_id AS team_id,
               api_keys.id AS key_id,
               teams.name AS team_name,
               teams.rate_limit_rpm AS rpm,
               teams.burst AS burst,
               teams.monthly_budget_usd AS budget_usd,
               teams.pii_action AS pii_action
        FROM api_keys
        JOIN teams ON teams.id = api_keys.team_id
        WHERE api_keys.key_hash = $1 AND api_keys.active = TRUE
        """,
        hashed_key,
    )
    if row is None:
        await redis.set(cache_key, "invalid", ex=10)
        raise _unauthorized()

    context = AuthContext.model_validate(dict(row))
    await redis.set(cache_key, context.model_dump_json(), ex=60)
    return context