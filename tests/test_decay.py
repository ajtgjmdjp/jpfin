"""Tests for factor decay analysis."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from japan_finance_factors._models import PriceData

from jpfin.decay import compute_decay
from jpfin.formatters import format_decay_table
from jpfin.models import FactorDecayResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_data(
    ticker: str,
    n_days: int = 800,
    start_price: float = 1000.0,
    step: float = 1.0,
) -> PriceData:
    return PriceData(
        ticker=ticker,
        prices=[
            {
                "date": (date(2022, 1, 3) + timedelta(days=i)).isoformat(),
                "close": start_price + step * i,
            }
            for i in range(n_days)
        ],
    )


def _make_universe(n_tickers: int = 10, n_days: int = 800) -> dict[str, PriceData]:
    """Build a synthetic universe with varying trends."""
    return {
        f"T{i:04d}": _make_price_data(
            f"T{i:04d}",
            n_days=n_days,
            start_price=1000 + i * 100,
            step=0.5 + i * 0.3,
        )
        for i in range(n_tickers)
    }


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestComputeDecayBasic:
    def test_returns_result(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        assert isinstance(result, FactorDecayResult)
        assert result.factor == "mom_3m"
        assert result.max_lag == 3

    def test_lag_count_matches_max_lag(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=4)
        assert len(result.lags) == 4

    def test_lag_numbers_sequential(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=5)
        assert [dl.lag for dl in result.lags] == [1, 2, 3, 4, 5]

    def test_n_obs_positive(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        for dl in result.lags:
            assert dl.n_obs > 0

    def test_lag1_has_most_observations(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=6)
        # Lag 1 should have >= lag 6 observations
        assert result.lags[0].n_obs >= result.lags[-1].n_obs


# ---------------------------------------------------------------------------
# IC values
# ---------------------------------------------------------------------------


class TestComputeDecayIC:
    def test_ic_values_in_range(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        for dl in result.lags:
            if dl.mean_ic is not None:
                assert -1.0 <= dl.mean_ic <= 1.0

    def test_std_nonnegative(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        for dl in result.lags:
            if dl.std_ic is not None:
                assert dl.std_ic >= 0.0

    def test_different_factors(self) -> None:
        pd = _make_universe()
        r1 = compute_decay(pd, "mom_3m", max_lag=2)
        r2 = compute_decay(pd, "realized_vol_60d", max_lag=2)
        assert r1.factor == "mom_3m"
        assert r2.factor == "realized_vol_60d"


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


class TestComputeDecayHalfLife:
    def test_half_life_type(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=6)
        # half_life is either None or a positive float
        if result.half_life_months is not None:
            assert result.half_life_months >= 2.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestComputeDecayErrors:
    def test_unsupported_factor_raises(self) -> None:
        pd = _make_universe()
        with pytest.raises(ValueError, match="Unsupported factor"):
            compute_decay(pd, "nonexistent_factor")

    def test_max_lag_zero_raises(self) -> None:
        pd = _make_universe()
        with pytest.raises(ValueError, match="max_lag must be >= 1"):
            compute_decay(pd, "mom_3m", max_lag=0)

    def test_insufficient_data_raises(self) -> None:
        # Only 30 days → not enough for 2 rebalance periods
        pd = {
            "T0001": PriceData(
                ticker="T0001",
                prices=[
                    {"date": (date(2024, 1, 2) + timedelta(days=i)).isoformat(), "close": 1000.0}
                    for i in range(30)
                ],
            ),
        }
        with pytest.raises(ValueError, match="Need at least 2"):
            compute_decay(pd, "mom_3m")


# ---------------------------------------------------------------------------
# Empty IC at high lags
# ---------------------------------------------------------------------------


class TestComputeDecayEdgeCases:
    def test_high_lag_may_have_zero_obs(self) -> None:
        # Short data, high max_lag → later lags may have n_obs=0
        pd = _make_universe(n_tickers=10, n_days=200)
        result = compute_decay(pd, "mom_3m", max_lag=10)
        # Some later lags should have 0 observations
        n_obs_list = [dl.n_obs for dl in result.lags]
        # At minimum, last lag should have fewer obs than first
        assert n_obs_list[-1] <= n_obs_list[0]

    def test_zero_obs_has_none_ic(self) -> None:
        pd = _make_universe(n_tickers=10, n_days=200)
        result = compute_decay(pd, "mom_3m", max_lag=10)
        for dl in result.lags:
            if dl.n_obs == 0:
                assert dl.mean_ic is None
                assert dl.std_ic is None

    def test_date_range_filter(self) -> None:
        pd = _make_universe()
        result = compute_decay(
            pd,
            "mom_3m",
            max_lag=3,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert result.max_lag == 3
        # Should have fewer observations than full range
        full = compute_decay(pd, "mom_3m", max_lag=3)
        assert result.lags[0].n_obs <= full.lags[0].n_obs


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestDecaySerialization:
    def test_model_dump_json_round_trip(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        data = result.model_dump()
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["factor"] == "mom_3m"
        assert len(restored["lags"]) == 3

    def test_model_dump_has_required_fields(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        data = result.model_dump()
        assert "factor" in data
        assert "max_lag" in data
        assert "lags" in data
        assert "half_life_months" in data
        lag0 = data["lags"][0]
        assert "lag" in lag0
        assert "mean_ic" in lag0
        assert "std_ic" in lag0
        assert "n_obs" in lag0


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatDecayTable:
    def test_header_present(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=3)
        output = format_decay_table(result)
        assert "Factor Decay" in output
        assert "mom_3m" in output
        assert "Max Lag: 3" in output

    def test_lag_rows_present(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=4)
        output = format_decay_table(result)
        # Should contain lag numbers 1-4
        for lag in range(1, 5):
            assert f"  {lag}" in output or f" {lag} " in output

    def test_half_life_shown(self) -> None:
        pd = _make_universe()
        result = compute_decay(pd, "mom_3m", max_lag=6)
        output = format_decay_table(result)
        assert "Half-life" in output


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestDecayCLI:
    def test_help(self) -> None:
        from click.testing import CliRunner

        from jpfin.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["decay", "--help"])
        assert result.exit_code == 0
        assert "--factor" in result.output
        assert "--max-lag" in result.output

    def test_no_data_source(self) -> None:
        from click.testing import CliRunner

        from jpfin.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["decay"])
        assert result.exit_code != 0
        assert "specify --csv or --db" in result.output
