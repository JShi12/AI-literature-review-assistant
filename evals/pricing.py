"""Per-model token prices, so eval reports can show what a run costs (or would cost, when replayed).

Production's LLMResult.cost is never filled in (it stays 0), so the harness estimates it here instead.
Prices are USD per 1M tokens, standard (non-batch, uncached) tier, as published by OpenAI in 2025 --
check https://openai.com/api/pricing/ and update this table when they change.
"""

from __future__ import annotations

from decimal import Decimal

PRICES_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    # model prefix: (input, output)
    "gpt-4.1-nano": (Decimal("0.10"), Decimal("0.40")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("1.60")),
    "gpt-4.1": (Decimal("2.00"), Decimal("8.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-5-nano": (Decimal("0.05"), Decimal("0.40")),
    "gpt-5-mini": (Decimal("0.25"), Decimal("2.00")),
    "gpt-5": (Decimal("1.25"), Decimal("10.00")),
    "text-embedding-3-small": (Decimal("0.02"), Decimal("0")),
    "text-embedding-3-large": (Decimal("0.13"), Decimal("0")),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Return the USD cost of a call, or None if the model isn't in the price table.

    Matches the longest known prefix, so dated snapshots like "gpt-4.1-mini-2025-04-14" are priced too.
    """
    prefix = max((known for known in PRICES_PER_MILLION if model.startswith(known)), key=len, default=None)
    if prefix is None:
        return None
    input_price, output_price = PRICES_PER_MILLION[prefix]
    return (input_price * input_tokens + output_price * output_tokens) / Decimal(1_000_000)
