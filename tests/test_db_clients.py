import asyncio
from unittest.mock import AsyncMock, Mock

import asyncpg

from app import db, redis_client


def test_create_pool_uses_configured_pool_size(monkeypatch):
    pool = object()
    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(asyncpg, "create_pool", create_pool)

    result = asyncio.run(db.create_pool("postgresql://example"))

    assert result is pool
    create_pool.assert_awaited_once_with(
        "postgresql://example",
        min_size=1,
        max_size=10,
    )


def test_create_redis_client_decodes_responses(monkeypatch):
    client = object()
    from_url = Mock(return_value=client)
    monkeypatch.setattr(redis_client.redis, "from_url", from_url)

    result = redis_client.create_client("redis://example")

    assert result is client
    from_url.assert_called_once_with(
        "redis://example",
        decode_responses=True,
    )
