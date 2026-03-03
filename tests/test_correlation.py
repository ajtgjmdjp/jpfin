"""Tests for factor correlation analysis."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from japan_finance_factors._models import PriceData

from jpfin.correlation import compute_factor_correlation
from jpfin.formatters import format_correlation_table
from jpfin.models import FactorCorrelationResult

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


def _make_universe(n_tickers: int = 20, n_days: int = 800) -> dict[str, PriceData]:
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


class TestComputeCorrelationBasic:
    def test_returns_result(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        assert isinstance(result, FactorCorrelationResult)

    def test_all_factors_included(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        # Should include all registered factors
        assert len(result.factors) >= 2

    def test_matrix_is_square(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        n = len(result.factors)
        assert len(result.correlation_matrix) == n
        for row in result.correlation_matrix:
            assert len(row) == n

    def test_diagonal_is_one(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        for i in range(len(result.factors)):
            assert result.correlation_matrix[i][i] == 1.0

    def test_symmetric(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        n = len(result.factors)
        for i in range(n):
            for j in range(n):
                assert result.correlation_matrix[i][j] == result.correlation_matrix[j][i]

    def test_values_in_range(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        for row in result.correlation_matrix:
            for val in row:
                if val is not None:
                    assert -1.0 <= val <= 1.0

    def test_n_dates_positive(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        assert result.n_dates > 0

    def test_method_stored(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        assert result.method == "spearman"


# ---------------------------------------------------------------------------
# Factor selection
# ---------------------------------------------------------------------------


class TestCorrelationFactorSelection:
    def test_specific_factors(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd, ["mom_3m", "mom_12m"])
        assert result.factors == ["mom_3m", "mom_12m"]
        assert len(result.correlation_matrix) == 2

    def test_unknown_factor_raises(self) -> None:
        pd = _make_universe()
        with pytest.raises(ValueError, match="Unknown factor"):
            compute_factor_correlation(pd, ["mom_3m", "nonexistent"])

    def test_single_factor_raises(self) -> None:
        pd = _make_universe()
        with pytest.raises(ValueError, match="at least 2"):
            compute_factor_correlation(pd, ["mom_3m"])


# ---------------------------------------------------------------------------
# n_obs_matrix
# ---------------------------------------------------------------------------


class TestCorrelationNObsMatrix:
    def test_n_obs_matrix_shape(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        n = len(result.factors)
        assert len(result.n_obs_matrix) == n
        for row in result.n_obs_matrix:
            assert len(row) == n

    def test_diagonal_n_obs_equals_n_dates(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        for i in range(len(result.factors)):
            assert result.n_obs_matrix[i][i] == result.n_dates

    def test_n_obs_nonnegative(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        for row in result.n_obs_matrix:
            for val in row:
                assert val >= 0


# ---------------------------------------------------------------------------
# Mean absolute correlation
# ---------------------------------------------------------------------------


class TestMeanAbsCorrelation:
    def test_length_matches_factors(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        assert len(result.mean_abs_correlation) == len(result.factors)

    def test_values_nonnegative(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        for val in result.mean_abs_correlation:
            assert val >= 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCorrelationErrors:
    def test_unsupported_method_raises(self) -> None:
        pd = _make_universe()
        with pytest.raises(ValueError, match="Unsupported method"):
            compute_factor_correlation(pd, method="pearson")

    def test_insufficient_data_raises(self) -> None:
        # Only 10 days, not enough for rebalance dates
        pd = {
            "T0001": PriceData(
                ticker="T0001",
                prices=[
                    {"date": (date(2024, 1, 2) + timedelta(days=i)).isoformat(), "close": 1000.0}
                    for i in range(10)
                ],
            ),
        }
        with pytest.raises(ValueError):
            compute_factor_correlation(pd)


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------


class TestCorrelationDateFilter:
    def test_date_range_filter(self) -> None:
        pd = _make_universe()
        result_full = compute_factor_correlation(pd)
        result_filtered = compute_factor_correlation(
            pd,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert result_filtered.n_dates <= result_full.n_dates


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestCorrelationSerialization:
    def test_model_dump_json_round_trip(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        data = result.model_dump()
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["method"] == "spearman"
        assert len(restored["factors"]) == len(result.factors)

    def test_model_dump_has_required_fields(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        data = result.model_dump()
        assert "factors" in data
        assert "correlation_matrix" in data
        assert "n_obs_matrix" in data
        assert "mean_abs_correlation" in data
        assert "n_dates" in data
        assert "method" in data


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatCorrelationTable:
    def test_header_present(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        output = format_correlation_table(result)
        assert "Factor Correlation Matrix" in output
        assert "spearman" in output

    def test_factor_names_in_output(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd, ["mom_3m", "mom_12m"])
        output = format_correlation_table(result)
        assert "mom_3m" in output
        assert "mom_12m" in output

    def test_redundancy_ranking_present(self) -> None:
        pd = _make_universe()
        result = compute_factor_correlation(pd)
        output = format_correlation_table(result)
        assert "redundancy" in output


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCorrelationCLI:
    def test_help(self) -> None:
        from click.testing import CliRunner

        from jpfin.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["correlation", "--help"])
        assert result.exit_code == 0
        assert "--factor" in result.output
        assert "--db" in result.output

    def test_no_data_source(self) -> None:
        from click.testing import CliRunner

        from jpfin.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["correlation"])
        assert result.exit_code != 0
        assert "specify --csv or --db" in result.output
