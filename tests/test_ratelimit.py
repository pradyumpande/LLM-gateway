import asyncio
from unittest.mock import AsyncMock, Mock

from app.auth import AuthContext
from app.ratelimit import check_rate_limit


def test_check_rate_limit_registers_and_calls_token_bucket_script():
    context = AuthContext(
        team_id=42,
        key_id=7,
        team_name="test-team",
        rpm=120,
        burst=8,
        budget_usd=10,
        pii_action="mask",
    )
    script = AsyncMock(return_value=[0, 2500])
    redis = Mock()
    redis.register_script.return_value = script

    result = asyncio.run(check_rate_limit(redis, context))

    assert result == (False, 2500)
    source = redis.register_script.call_args.args[0]
    assert "redis.call('TIME')" in source
    assert "redis.call('HMGET'" in source
    script.assert_awaited_once_with(
        keys=["rl:42"],
        args=[8, 2.0, 1],
        client=redis,
    )
