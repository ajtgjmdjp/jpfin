"""Tests for rolling window analysis."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from japan_finance_factors._models import PriceData

from jpfin.formatters import format_rolling_table
from jpfin.models import (
    BacktestResult,
    DataQuality,
    FactorMetrics,
    MonthlyReturn,
    PerformanceMetrics,
)
from jpfin.rolling import compute_rolling

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    n_months: int,
    ic_series: list[float] | None = None,
    factor: str = "mom_3m",
) -> BacktestResult:
    """Build a synthetic BacktestResult with n_months of 1%/month returns."""
    base = date(2024, 1, 28)
    monthly_returns = []
    for i in range(n_months):
        start = base + timedelta(days=30 * i)
        end = base + timedelta(days=30 * (i + 1))
        monthly_returns.append(
            MonthlyReturn(
                period_start=start.isoformat(),
                period_end=end.isoformat(),
                monthly_return=0.01,
                cumulative=1.01 ** (i + 1),
            )
        )

    factor_metrics = None
    if ic_series is not None:
        factor_metrics = FactorMetrics(
            mean_ic=sum(ic_series) / len(ic_series) if ic_series else None,
            ic_series=ic_series,
            mean_turnover=None,
            turnover_series=[],
        )

    return BacktestResult(
        factor=factor,
        top_n=5,
        period=f"{monthly_returns[0].period_start} ~ {monthly_returns[-1].period_end}",
        months=n_months,
        performance=PerformanceMetrics(
            total_return=1.01**n_months - 1,
            cagr=0.12,
            annualized_vol=0.15,
            sharpe_ratio=0.80,
            max_drawdown=-0.05,
        ),
        monthly_returns=monthly_returns,
        holdings_history=[],
        data_quality=DataQuality(
            total_rebalances=n_months,
            skipped_rebalances=0,
            total_ticker_slots=n_months * 5,
            ffill_count=0,
            skip_count=0,
        ),
        factor_metrics=factor_metrics,
    )


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


# ---------------------------------------------------------------------------
# Basic window count
# ---------------------------------------------------------------------------


class TestComputeRollingBasic:
    def test_24m_window12_step1_gives_13_windows(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.total_months == 24
        assert ra.window_months == 12
        assert len(ra.windows) == 13

    def test_factor_name_preserved(self) -> None:
        result = _make_result(24, factor="realized_vol_60d")
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.factor == "realized_vol_60d"

    def test_window_start_end_are_correct(self) -> None:
        result = _make_result(12)
        ra = compute_rolling(result, window_months=12, step=1)
        assert len(ra.windows) == 1
        w = ra.windows[0]
        assert w.window_start == result.monthly_returns[0].period_start
        assert w.window_end == result.monthly_returns[-1].period_end


# ---------------------------------------------------------------------------
# Step sizes
# ---------------------------------------------------------------------------


class TestComputeRollingStep:
    def test_step6_gives_3_windows(self) -> None:
        # start=0 → end=12, start=6 → end=18, start=12 → end=24; start=18 > 24-12
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=6)
        assert len(ra.windows) == 3

    def test_step3_gives_5_windows(self) -> None:
        # start = 0,3,6,9,12 → 5 windows
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=3)
        assert len(ra.windows) == 5

    def test_step_equals_window_non_overlapping(self) -> None:
        # step=12, window=12, total=24 → 2 windows
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=12)
        assert len(ra.windows) == 2


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestComputeRollingBoundary:
    def test_window_equals_total_gives_1_window(self) -> None:
        result = _make_result(12)
        ra = compute_rolling(result, window_months=12, step=1)
        assert len(ra.windows) == 1

    def test_minimum_window_2_months(self) -> None:
        result = _make_result(4)
        ra = compute_rolling(result, window_months=2, step=1)
        assert len(ra.windows) == 3  # start=0,1,2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestComputeRollingErrors:
    def test_window_less_than_2_raises(self) -> None:
        result = _make_result(12)
        with pytest.raises(ValueError, match="window_months must be >= 2"):
            compute_rolling(result, window_months=1)

    def test_window_greater_than_total_raises(self) -> None:
        result = _make_result(6)
        with pytest.raises(ValueError, match="exceeds total months"):
            compute_rolling(result, window_months=12)

    def test_step_less_than_1_raises(self) -> None:
        result = _make_result(12)
        with pytest.raises(ValueError, match="step must be >= 1"):
            compute_rolling(result, window_months=6, step=0)


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------


class TestComputeRollingIC:
    def test_ic_present_computes_per_window(self) -> None:
        ic_series = [0.1 * (i % 3 + 1) for i in range(24)]  # [0.1, 0.2, 0.3, ...]
        result = _make_result(24, ic_series=ic_series)
        ra = compute_rolling(result, window_months=12, step=1)
        # First window IC = mean(ic_series[0:12])
        expected_ic = sum(ic_series[0:12]) / 12
        assert ra.windows[0].mean_ic == pytest.approx(expected_ic)

    def test_ic_absent_factor_metrics_none(self) -> None:
        result = _make_result(24, ic_series=None)
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.ic_mean is None
        assert ra.ic_std is None
        for w in ra.windows:
            assert w.mean_ic is None

    def test_ic_empty_series(self) -> None:
        result = _make_result(24, ic_series=[])
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.ic_mean is None
        for w in ra.windows:
            assert w.mean_ic is None

    def test_ic_summary_stats(self) -> None:
        # Uniform IC → std = 0
        ic_series = [0.15] * 24
        result = _make_result(24, ic_series=ic_series)
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.ic_mean == pytest.approx(0.15)
        assert ra.ic_std == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


class TestComputeRollingSummary:
    def test_sharpe_mean_between_min_and_max(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.sharpe_min <= ra.sharpe_mean <= ra.sharpe_max

    def test_sharpe_std_nonnegative(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        assert ra.sharpe_std >= 0.0

    def test_single_window_sharpe_std_zero(self) -> None:
        result = _make_result(12)
        ra = compute_rolling(result, window_months=12, step=1)
        assert len(ra.windows) == 1
        assert ra.sharpe_std == 0.0
        assert ra.sharpe_min == ra.sharpe_max == ra.sharpe_mean

    def test_performance_metrics_populated(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        for w in ra.windows:
            assert isinstance(w.performance.sharpe_ratio, float)


# ---------------------------------------------------------------------------
# Integration: run_backtest → compute_rolling
# ---------------------------------------------------------------------------


class TestComputeRollingIntegration:
    def test_run_backtest_then_compute_rolling(self) -> None:
        from jpfin.backtest import run_backtest

        price_data = {f"T{i:04d}": _make_price_data(f"T{i:04d}") for i in range(10)}
        result = run_backtest(price_data, factor_fn="mom_3m", top_n=3)
        # Should have many months; rolling analysis must succeed
        assert result.months >= 12
        ra = compute_rolling(result, window_months=12, step=1)
        assert len(ra.windows) >= 1
        assert ra.total_months == result.months


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestRollingAnalysisSerialization:
    def test_model_dump_json_round_trip(self) -> None:
        ic_series = [0.1] * 24
        result = _make_result(24, ic_series=ic_series)
        ra = compute_rolling(result, window_months=12, step=1)
        data = ra.model_dump()
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["factor"] == ra.factor
        assert restored["window_months"] == ra.window_months
        assert len(restored["windows"]) == len(ra.windows)

    def test_model_dump_has_required_fields(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        data = ra.model_dump()
        assert "sharpe_mean" in data
        assert "sharpe_std" in data
        assert "sharpe_min" in data
        assert "sharpe_max" in data
        assert "windows" in data
        w0 = data["windows"][0]
        assert "window_start" in w0
        assert "window_end" in w0
        assert "performance" in w0


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatRollingTable:
    def test_header_present(self) -> None:
        result = _make_result(24, ic_series=[0.1] * 24)
        ra = compute_rolling(result, window_months=12, step=1)
        output = format_rolling_table(ra)
        assert "Rolling Analysis" in output
        assert "mom_3m" in output
        assert "Window: 12M" in output
        assert "Windows: 13" in output

    def test_summary_present(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        output = format_rolling_table(ra)
        assert "Sharpe:" in output
        assert "mean=" in output

    def test_ic_column_present_when_ic_available(self) -> None:
        result = _make_result(24, ic_series=[0.1] * 24)
        ra = compute_rolling(result, window_months=12, step=1)
        output = format_rolling_table(ra)
        assert "IC" in output
        assert "IC:" in output

    def test_ic_column_absent_when_no_ic(self) -> None:
        result = _make_result(24, ic_series=None)
        ra = compute_rolling(result, window_months=12, step=1)
        output = format_rolling_table(ra)
        # Should not mention IC summary line when no IC
        assert "IC:" not in output

    def test_correct_row_count(self) -> None:
        result = _make_result(24)
        ra = compute_rolling(result, window_months=12, step=1)
        output = format_rolling_table(ra)
        # Each window row starts with "  2024-" (date pattern)
        rows = [line for line in output.splitlines() if "2024-" in line or "2025-" in line]
        assert len(rows) == 13


class TestComputeRollingEdgeCases:
    """Edge-case tests for rolling window analysis."""

    def test_one_month_result_raises(self) -> None:
        result = _make_result(1)
        with pytest.raises(ValueError, match=r"window_months.*exceeds total"):
            compute_rolling(result, window_months=2)
