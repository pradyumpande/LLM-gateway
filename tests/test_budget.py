import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.auth import AuthContext
from app.budget import check_budget, record_spend, spend_key


def _context(budget=Decimal("10")):
    return AuthContext(
        team_id=5,
        key_id=7,
        team_name="test-team",
        rpm=60,
        burst=10,
        budget_usd=budget,
        pii_action="mask",
    )


def test_check_budget_reads_monthly_team_spend():
    redis = SimpleNamespace(get=AsyncMock(return_value="3.25"))

    result = asyncio.run(check_budget(redis, _context(), "2026-11"))

    assert result == (True, "spend:5:2026-11", 3.25)
    redis.get.assert_awaited_once_with("spend:5:2026-11")
    assert spend_key(5, "2026-11") == "spend:5:2026-11"


def test_record_spend_increments_expires_and_audits_at_threshold():
    redis = SimpleNamespace(
        incrbyfloat=AsyncMock(return_value=8.0),
        expire=AsyncMock(return_value=True),
        set=AsyncMock(return_value=True),
        delete=AsyncMock(),
    )
    database = SimpleNamespace(execute=AsyncMock())

    spent = asyncio.run(
        record_spend(
            redis,
            database,
            _context(),
            "2026-11",
            Decimal("8"),
        )
    )

    assert spent == 8.0
    redis.incrbyfloat.assert_awaited_once_with("spend:5:2026-11", "8")
    redis.expire.assert_awaited_once_with("spend:5:2026-11", 40 * 24 * 60 * 60)
    redis.set.assert_awaited_once_with(
        "alert80:5:2026-11",
        "1",
        nx=True,
        ex=40 * 24 * 60 * 60,
    )
    query, actor, event, details = database.execute.await_args.args
    assert "INSERT INTO audit_log" in query
    assert actor == "team:5"
    assert event == "budget_80_percent"
    assert json.loads(details) == {
        "team_id": 5,
        "month": "2026-11",
        "spend_usd": 8.0,
        "budget_usd": "10",
    }


def test_record_spend_only_audits_once_when_warning_guard_exists():
    redis = SimpleNamespace(
        incrbyfloat=AsyncMock(return_value=9.0),
        expire=AsyncMock(return_value=True),
        set=AsyncMock(return_value=None),
        delete=AsyncMock(),
    )
    database = SimpleNamespace(execute=AsyncMock())

    asyncio.run(
        record_spend(
            redis,
            database,
            _context(),
            "2026-11",
            Decimal("1"),
        )
    )

    database.execute.assert_not_awaited()
