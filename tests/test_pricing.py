from decimal import Decimal

import pytest

from app.usage.pricing import cost_usd


def test_cost_usd_matches_hand_calculation():
    # 2M input tokens at $0.30 + 0.5M output tokens at $2.50 = $1.85.
    assert cost_usd("gemini-flash", 2_000_000, 500_000) == Decimal("1.85")


def test_cost_usd_rejects_unknown_alias():
    with pytest.raises(ValueError, match="Unknown pricing alias 'not-a-model'"):
        cost_usd("not-a-model", 100, 50)


def test_cost_usd_reports_unpublished_groq_price():
    with pytest.raises(ValueError, match="No published token price.*groq-8b"):
        cost_usd("groq-8b", 100, 50)
