import secrets
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from app.auth import generate_key, hash_key, key_prefix
from app.config import Settings


def require_admin_token(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> None:
    expected_token = Settings().admin_token
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_TOKEN is not configured.",
        )
    if x_admin_token is None or not secrets.compare_digest(
        x_admin_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid admin token.")


class TeamCreate(BaseModel):
    name: str
    monthly_budget_usd: Decimal
    rate_limit_rpm: int
    burst: int


class TeamUpdate(BaseModel):
    monthly_budget_usd: Optional[Decimal] = None
    rate_limit_rpm: Optional[int] = None
    burst: Optional[int] = None


router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)

@router.post("/teams")
async def create_team(team: TeamCreate, request: Request) -> dict:
    row = await request.app.state.db.fetchrow(
        """
        INSERT INTO teams (name, monthly_budget_usd, rate_limit_rpm, burst)
        VALUES ($1, $2, $3, $4)
        RETURNING id, name, monthly_budget_usd, rate_limit_rpm, burst,
                  pii_action, created_at
        """,
        team.name,
        team.monthly_budget_usd,
        team.rate_limit_rpm,
        team.burst,
    )
    return dict(row)


@router.patch("/teams/{team_id}")
async def update_team(
    team_id: int,
    team: TeamUpdate,
    request: Request,
) -> dict:
    if all(
        value is None
        for value in (
            team.monthly_budget_usd,
            team.rate_limit_rpm,
            team.burst,
        )
    ):
        raise HTTPException(status_code=422, detail="Provide at least one field to update.")

    row = await request.app.state.db.fetchrow(
        """
        UPDATE teams
        SET monthly_budget_usd = COALESCE($2, monthly_budget_usd),
            rate_limit_rpm = COALESCE($3, rate_limit_rpm),
            burst = COALESCE($4, burst)
        WHERE id = $1
        RETURNING id, name, monthly_budget_usd, rate_limit_rpm, burst,
                  pii_action, created_at
        """,
        team_id,
        team.monthly_budget_usd,
        team.rate_limit_rpm,
        team.burst,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Team not found.")
    return dict(row)


@router.get("/teams")
async def list_teams(request: Request) -> list[dict]:
    rows = await request.app.state.db.fetch(
        """
        SELECT id, name, monthly_budget_usd, rate_limit_rpm, burst,
               pii_action, created_at
        FROM teams
        ORDER BY id
        """
    )
    return [dict(row) for row in rows]


@router.post("/teams/{team_id}/keys")
async def create_team_key(team_id: int, request: Request) -> dict:
    key = generate_key()
    row = await request.app.state.db.fetchrow(
        """
        INSERT INTO api_keys (team_id, key_prefix, key_hash)
        SELECT $1, $2, $3
        WHERE EXISTS (SELECT 1 FROM teams WHERE id = $1)
        RETURNING id, team_id, key_prefix, active
        """,
        team_id,
        key_prefix(key),
        hash_key(key),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Team not found.")
    return {**dict(row), "key": key}


@router.delete("/keys/{key_id}")
async def deactivate_key(key_id: int, request: Request) -> dict[str, object]:
    row = await request.app.state.db.fetchrow(
        """
        UPDATE api_keys
        SET active = FALSE
        WHERE id = $1
        RETURNING id, key_hash
        """,
        key_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found.")

    await request.app.state.redis.delete(f"auth:{row['key_hash']}")
    return {"id": row["id"], "active": False}