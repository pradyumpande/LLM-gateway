from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.admin.routes as admin_routes
import app.main as main
from app.auth import hash_key


ADMIN_TOKEN = "test-admin-token"
HEADERS = {"X-Admin-Token": ADMIN_TOKEN}


@pytest.fixture
def admin_client(monkeypatch):
    database = SimpleNamespace(
        fetchrow=AsyncMock(),
        fetch=AsyncMock(),
        close=AsyncMock(),
    )
    redis = SimpleNamespace(
        ping=AsyncMock(),
        delete=AsyncMock(),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        main,
        "Settings",
        lambda: SimpleNamespace(
            database_url="postgresql://test",
            redis_url="redis://test",
            pii_score_threshold=0.5,
            max_input_chars=200_000,
            injection_flag_threshold=0.4,
            injection_block_threshold=0.8,
            pii_entities=[
                "EMAIL_ADDRESS",
                "IN_PHONE",
                "AADHAAR",
                "IN_PAN",
                "CREDIT_CARD",
            ],
        ),
    )
    monkeypatch.setattr(
        main,
        "build_analyzer",
        Mock(return_value=SimpleNamespace(analyze=Mock(return_value=[]))),
    )
    monkeypatch.setattr(main.db, "create_pool", AsyncMock(return_value=database))
    monkeypatch.setattr(
        main.redis_client,
        "create_client",
        Mock(return_value=redis),
    )
    monkeypatch.setattr(
        admin_routes,
        "Settings",
        lambda: SimpleNamespace(admin_token=ADMIN_TOKEN),
    )

    with TestClient(main.app) as client:
        yield client, database, redis


def test_admin_routes_reject_missing_and_invalid_tokens(admin_client):
    client, database, _ = admin_client

    assert client.get("/admin/teams").status_code == 401
    assert client.get(
        "/admin/teams",
        headers={"X-Admin-Token": "wrong-token"},
    ).status_code == 401
    database.fetch.assert_not_awaited()


def test_non_ascii_invalid_admin_token_is_rejected(monkeypatch):
    monkeypatch.setattr(
        admin_routes,
        "Settings",
        lambda: SimpleNamespace(admin_token=ADMIN_TOKEN),
    )

    with pytest.raises(HTTPException) as exc_info:
        admin_routes.require_admin_token("tést-token")

    assert exc_info.value.status_code == 401


def test_create_team_uses_parameterized_insert(admin_client):
    client, database, _ = admin_client
    database.fetchrow.return_value = {
        "id": 7,
        "name": "search-team",
        "monthly_budget_usd": Decimal("5"),
        "rate_limit_rpm": 60,
        "burst": 10,
        "pii_action": "mask",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }

    response = client.post(
        "/admin/teams",
        headers=HEADERS,
        json={
            "name": "search-team",
            "monthly_budget_usd": 5,
            "rate_limit_rpm": 60,
            "burst": 10,
        },
    )

    assert response.status_code == 200
    assert response.json()["name"] == "search-team"
    query, *values = database.fetchrow.await_args.args
    assert "VALUES ($1, $2, $3, $4)" in query
    assert "search-team" not in query
    assert values == ["search-team", Decimal("5"), 60, 10]


def test_patch_team_and_list_teams(admin_client):
    client, database, redis = admin_client
    team = {
        "id": 7,
        "name": "search-team",
        "monthly_budget_usd": Decimal("8"),
        "rate_limit_rpm": 90,
        "burst": 12,
        "pii_action": "log_only",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    database.fetchrow.return_value = team
    database.fetch.side_effect = [
        [{"key_hash": "cached-hash-one"}, {"key_hash": "cached-hash-two"}],
        [team],
    ]
    update = client.patch(
        "/admin/teams/7",
        headers=HEADERS,
        json={
            "monthly_budget_usd": 8,
            "rate_limit_rpm": 90,
            "burst": 12,
            "pii_action": "log_only",
        },
    )

    assert update.status_code == 200
    assert update.json()["pii_action"] == "log_only"
    query, *values = database.fetchrow.await_args.args
    assert "WHERE id = $1" in query
    assert "search-team" not in query
    assert "pii_action = COALESCE($5, pii_action)" in query
    assert values == [7, Decimal("8"), 90, 12, "log_only"]
    redis.delete.assert_awaited_once_with(
        "auth:cached-hash-one",
        "auth:cached-hash-two",
    )

    listing = client.get("/admin/teams", headers=HEADERS)

    assert listing.status_code == 200
    assert listing.json()[0]["id"] == 7
    assert database.fetch.await_count == 2


def test_patch_team_rejects_unknown_pii_action(admin_client):
    client, database, _ = admin_client

    response = client.patch(
        "/admin/teams/7",
        headers=HEADERS,
        json={"pii_action": "allow_all"},
    )

    assert response.status_code == 422
    database.fetchrow.assert_not_awaited()


def test_create_team_key_returns_secret_once_and_stores_only_hash(admin_client, monkeypatch):
    client, database, _ = admin_client
    key = "gw_once-only-secret"
    monkeypatch.setattr(admin_routes, "generate_key", lambda: key)
    database.fetchrow.return_value = {
        "id": 3,
        "team_id": 7,
        "key_prefix": key[:8],
        "active": True,
    }

    response = client.post("/admin/teams/7/keys", headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["key"] == key
    assert "key_hash" not in response.json()
    query, *values = database.fetchrow.await_args.args
    assert "$1" in query and "$2" in query and "$3" in query
    assert key not in query
    assert values == [7, key[:8], hash_key(key)]


def test_delete_key_deactivates_and_invalidates_redis_cache(admin_client):
    client, database, redis = admin_client
    database.fetchrow.return_value = {"id": 3, "key_hash": "hashed-key"}

    response = client.delete("/admin/keys/3", headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == {"id": 3, "active": False}
    query, *values = database.fetchrow.await_args.args
    assert "SET active = FALSE" in query
    assert "WHERE id = $1" in query
    assert values == [3]
    redis.delete.assert_awaited_once_with("auth:hashed-key")
