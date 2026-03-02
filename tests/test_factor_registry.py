"""Tests for jpfin.factor_registry — Shared factor registry."""

from __future__ import annotations

from jpfin.factor_registry import (
    HIGHER_IS_BETTER,
    PRICE_FACTOR_FNS,
    compute_price_factors,
    supported_factors,
)


def test_all_factors_have_direction() -> None:
    """Every registered factor must have a ranking direction."""
    for name in PRICE_FACTOR_FNS:
        assert name in HIGHER_IS_BETTER, f"{name} missing from HIGHER_IS_BETTER"


def test_supported_factors_matches_registry() -> None:
    assert set(supported_factors()) == set(PRICE_FACTOR_FNS.keys())


def test_compute_returns_all_by_default() -> None:
    from japan_finance_factors._models import PriceData

    # Minimal price data (may not compute anything, but should not crash)
    pd = PriceData(ticker="9999", prices=[{"date": "2025-05-01", "close": 100.0}])
    result = compute_price_factors(pd)
    assert set(result.keys()) == set(PRICE_FACTOR_FNS.keys())


def test_compute_specific_factors() -> None:
    from japan_finance_factors._models import PriceData

    pd = PriceData(ticker="9999", prices=[{"date": "2025-05-01", "close": 100.0}])
    result = compute_price_factors(pd, factors=["mom_3m"])
    assert set(result.keys()) == {"mom_3m"}


def test_compute_unknown_factor_returns_none() -> None:
    from japan_finance_factors._models import PriceData

    pd = PriceData(ticker="9999", prices=[{"date": "2025-05-01", "close": 100.0}])
    result = compute_price_factors(pd, factors=["nonexistent_factor"])
    assert result == {"nonexistent_factor": None}
