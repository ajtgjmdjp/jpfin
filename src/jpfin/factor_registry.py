"""Shared factor registry for backtest and event-study modules.

Single source of truth for price-based factor functions and their
ranking direction. Avoids duplication across backtest.py and event_study.py.
"""

from __future__ import annotations

from collections.abc import Callable

from japan_finance_factors._models import PriceData
from japan_finance_factors.factors import momentum, risk

# Price-based factor functions: name → callable(PriceData) → float | None
PRICE_FACTOR_FNS: dict[str, Callable[[PriceData], float | None]] = {
    "mom_3m": momentum.mom_3m,
    "mom_12m": momentum.mom_12m,
    "realized_vol_60d": risk.realized_vol_60d,
    "max_drawdown_252d": risk.max_drawdown_252d,
}

# Ranking direction: True = higher value is better
HIGHER_IS_BETTER: dict[str, bool] = {
    "mom_3m": True,
    "mom_12m": True,
    "max_drawdown_252d": True,  # less negative = less risk
    "realized_vol_60d": False,  # lower vol = less risk
}


def compute_price_factors(
    price_data: PriceData,
    factors: list[str] | None = None,
) -> dict[str, float | None]:
    """Compute price-based factors for a single ticker.

    Args:
        price_data: Price history for one ticker.
        factors: Specific factor names. ``None`` for all available.

    Returns:
        Dict mapping factor name to value (or ``None`` if insufficient data).
    """
    requested = factors or list(PRICE_FACTOR_FNS.keys())
    result: dict[str, float | None] = {}
    for name in requested:
        fn = PRICE_FACTOR_FNS.get(name)
        if fn is not None:
            result[name] = fn(price_data)
        else:
            result[name] = None
    return result


def supported_factors() -> list[str]:
    """Return list of supported price-based factor names."""
    return list(PRICE_FACTOR_FNS.keys())
