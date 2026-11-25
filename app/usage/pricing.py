from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml


_PRICING_FILE = Path(__file__).resolve().parents[2] / "config" / "pricing.yaml"
_PER_MILLION = Decimal(1_000_000)


@lru_cache(maxsize=1)
def _load_pricing() -> dict[str, dict[str, Optional[Decimal]]]:
    with _PRICING_FILE.open("r", encoding="utf-8") as pricing_file:
        raw_pricing = yaml.safe_load(pricing_file)

    if not isinstance(raw_pricing, dict):
        raise ValueError(f"Pricing config {_PRICING_FILE} must be a YAML mapping.")

    pricing: dict[str, dict[str, Optional[Decimal]]] = {}
    for alias, rates in raw_pricing.items():
        if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
            raise ValueError(f"Pricing for alias {alias!r} must define input and output.")
        try:
            pricing[alias] = {
                "input": (
                    None if rates["input"] is None else Decimal(str(rates["input"]))
                ),
                "output": (
                    None if rates["output"] is None else Decimal(str(rates["output"]))
                ),
            }
        except InvalidOperation as exc:
            raise ValueError(f"Pricing for alias {alias!r} must be numeric.") from exc

    return pricing


def pricing_aliases() -> frozenset[str]:
    return frozenset(_load_pricing())


def cost_usd(
    alias: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts must be non-negative.")

    pricing = _load_pricing()
    rates = pricing.get(alias)
    if rates is None:
        raise ValueError(f"Unknown pricing alias {alias!r}.")

    input_rate = rates["input"]
    output_rate = rates["output"]
    if input_rate is None or output_rate is None:
        raise ValueError(
            f"No published token price is configured for alias {alias!r}."
        )

    return (
        Decimal(prompt_tokens) * input_rate
        + Decimal(completion_tokens) * output_rate
    ) / _PER_MILLION