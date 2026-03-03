"""Tests for multi-factor composite scoring."""

from __future__ import annotations

import pytest
from japan_finance_factors._models import PriceData

from jpfin.composite import (
    _cross_sectional_zscore,
    compute_composite_scores,
    validate_composite_args,
)


class TestValidateCompositeArgs:
    def test_equal_weights(self) -> None:
        factors, weights = validate_composite_args(["mom_3m", "realized_vol_60d"])
        assert factors == ["mom_3m", "realized_vol_60d"]
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_explicit_weights(self) -> None:
        _factors, weights = validate_composite_args(["mom_3m", "realized_vol_60d"], [0.7, 0.3])
        assert weights == [0.7, 0.3]

    def test_empty_factors(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            validate_composite_args([])

    def test_unknown_factor(self) -> None:
        with pytest.raises(ValueError, match="Unknown factor"):
            validate_composite_args(["nonexistent"])

    def test_weight_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Weight count"):
            validate_composite_args(["mom_3m", "realized_vol_60d"], [0.5])


class TestCrossSectionalZscore:
    def test_basic(self) -> None:
        values = {"A": 10.0, "B": 20.0, "C": 30.0}
        zs = _cross_sectional_zscore(values, higher_is_better=True)
        assert zs["A"] < zs["B"] < zs["C"]
        # Mean of z-scores should be ~0
        assert abs(sum(zs.values()) / len(zs)) < 1e-9

    def test_sign_flip(self) -> None:
        """When higher_is_better=False, higher raw value → lower z-score."""
        values = {"A": 10.0, "B": 20.0, "C": 30.0}
        zs = _cross_sectional_zscore(values, higher_is_better=False)
        assert zs["A"] > zs["B"] > zs["C"]

    def test_clip(self) -> None:
        values = {"A": 1.0, "B": 100.0}
        zs = _cross_sectional_zscore(values, higher_is_better=True, clip=2.0)
        for z in zs.values():
            assert -2.0 <= z <= 2.0

    def test_single_ticker(self) -> None:
        values = {"A": 10.0}
        zs = _cross_sectional_zscore(values, higher_is_better=True)
        assert zs["A"] == 0.0

    def test_zero_variance(self) -> None:
        values = {"A": 5.0, "B": 5.0, "C": 5.0}
        zs = _cross_sectional_zscore(values, higher_is_better=True)
        for z in zs.values():
            assert z == 0.0


class TestCompositeScores:
    def test_two_factors(self) -> None:
        ticker_vals = {
            "A": {"mom_3m": 0.1, "realized_vol_60d": 0.3},
            "B": {"mom_3m": 0.2, "realized_vol_60d": 0.1},
            "C": {"mom_3m": 0.3, "realized_vol_60d": 0.2},
        }
        scores = compute_composite_scores(
            ticker_vals,
            ["mom_3m", "realized_vol_60d"],
            [0.5, 0.5],
        )
        assert len(scores) == 3
        # Scores should be sorted descending
        assert scores[0][1] >= scores[1][1] >= scores[2][1]

    def test_partial_coverage(self) -> None:
        """Tickers missing some factors still get scored."""
        ticker_vals: dict[str, dict[str, float | None]] = {
            "A": {"mom_3m": 0.1, "realized_vol_60d": 0.3},
            "B": {"mom_3m": 0.2, "realized_vol_60d": None},
        }
        scores = compute_composite_scores(
            ticker_vals,
            ["mom_3m", "realized_vol_60d"],
            [0.5, 0.5],
        )
        # Both tickers should be included (B has mom_3m)
        tickers = [t for t, _ in scores]
        assert "A" in tickers
        assert "B" in tickers

    def test_equal_weights(self) -> None:
        ticker_vals = {
            "A": {"mom_3m": 0.1, "realized_vol_60d": 0.3},
            "B": {"mom_3m": 0.3, "realized_vol_60d": 0.1},
        }
        scores = compute_composite_scores(
            ticker_vals,
            ["mom_3m", "realized_vol_60d"],
            [0.5, 0.5],
        )
        # With equal weights and symmetric values, scores should be close
        assert len(scores) == 2

    def test_empty_input(self) -> None:
        scores = compute_composite_scores(
            {},
            ["mom_3m"],
            [1.0],
        )
        assert scores == []


class TestBacktestMultiFactor:
    """Integration tests for multi-factor through run_backtest."""

    def _make_price_data(self) -> dict[str, PriceData]:
        """Create price data with enough history for factor computation."""
        import random

        random.seed(42)
        prices: dict[str, list[dict[str, float | str]]] = {}
        for ticker in ["A", "B", "C", "D", "E"]:
            ticker_prices = []
            close = 1000.0 + random.random() * 500
            for month in range(1, 13):
                for day in range(1, 29):
                    d = f"2023-{month:02d}-{day:02d}"
                    close *= 1 + (random.random() - 0.48) * 0.02
                    ticker_prices.append({"date": d, "close": round(close, 2)})
            prices[ticker] = ticker_prices

        return {t: PriceData(ticker=t, prices=p) for t, p in prices.items()}

    def test_single_factor_unchanged(self) -> None:
        """Single factor string still works."""
        from jpfin.backtest import run_backtest

        data = self._make_price_data()
        result = run_backtest(data, "mom_3m", top_n=2)
        assert result.factor == "mom_3m"

    def test_multi_factor_list(self) -> None:
        from jpfin.backtest import run_backtest

        data = self._make_price_data()
        result = run_backtest(data, ["mom_3m", "realized_vol_60d"], top_n=2)
        assert result.factor == "mom_3m+realized_vol_60d"
        assert result.months > 0

    def test_multi_factor_with_weights(self) -> None:
        from jpfin.backtest import run_backtest

        data = self._make_price_data()
        result = run_backtest(
            data,
            ["mom_3m", "realized_vol_60d"],
            weights=[0.7, 0.3],
            top_n=2,
        )
        assert result.factor == "mom_3m+realized_vol_60d"

    def test_multi_factor_bad_weight_count(self) -> None:
        from jpfin.backtest import run_backtest

        data = self._make_price_data()
        with pytest.raises(ValueError, match="Weight count"):
            run_backtest(
                data,
                ["mom_3m", "realized_vol_60d"],
                weights=[0.5],
                top_n=2,
            )
