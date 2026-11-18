from pathlib import Path
from typing import Any

from app.auth import AuthContext


_SCRIPT_PATH = Path(__file__).parent / "lua" / "token_bucket.lua"
_TOKEN_BUCKET_SCRIPT = _SCRIPT_PATH.read_text(encoding="utf-8")


async def check_rate_limit(redis: Any, ctx: AuthContext) -> tuple[bool, int]:
    script = redis.register_script(_TOKEN_BUCKET_SCRIPT)
    result = await script(
        keys=[f"rl:{ctx.team_id}"],
        args=[ctx.burst, ctx.rpm / 60, 1],
        client=redis,
    )
    return bool(int(result[0])), int(result[1])