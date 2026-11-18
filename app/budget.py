import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from app.auth import AuthContext


_SPEND_TTL_SECONDS = 40 * 24 * 60 * 60


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def spend_key(team_id: int, month: Optional[str] = None) -> str:
    return f"spend:{team_id}:{month or current_month()}"


def alert_key(team_id: int, month: Optional[str] = None) -> str:
    return f"alert80:{team_id}:{month or current_month()}"


async def check_budget(
    redis: Any,
    ctx: AuthContext,
    month: Optional[str] = None,
) -> tuple[bool, str, float]:
    key = spend_key(ctx.team_id, month)
    spent = float(await redis.get(key) or 0)
    return spent < float(ctx.budget_usd), key, spent


async def record_spend(
    redis: Any,
    db: Any,
    ctx: AuthContext,
    month: str,
    cost: Decimal,
) -> float:
    key = spend_key(ctx.team_id, month)
    new_spend = float(
        await redis.incrbyfloat(key, format(cost, "f"))
    )
    await redis.expire(key, _SPEND_TTL_SECONDS)

    if new_spend >= float(ctx.budget_usd) * 0.8:
        warning_key = alert_key(ctx.team_id, month)
        alert_created = await redis.set(
            warning_key,
            "1",
            nx=True,
            ex=_SPEND_TTL_SECONDS,
        )
        if alert_created:
            details = json.dumps(
                {
                    "team_id": ctx.team_id,
                    "month": month,
                    "spend_usd": new_spend,
                    "budget_usd": str(ctx.budget_usd),
                }
            )
            try:
                await db.execute(
                    """
                    INSERT INTO audit_log (actor, event, details)
                    VALUES ($1, $2, $3::jsonb)
                    """,
                    f"team:{ctx.team_id}",
                    "budget_80_percent",
                    details,
                )
            except Exception:
                await redis.delete(warning_key)
                raise

    return new_spend