"""Tests for portfolio analytics."""

from __future__ import annotations

import json

import pytest

from jpfin.models import (
    BacktestResult,
    DataQuality,
    FactorMetrics,
    HoldingsPeriod,
    MonthlyReturn,
    PerformanceMetrics,
    PortfolioAnalytics,
)
from jpfin.portfolio import compute_portfolio_analytics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    n_periods: int = 12,
    top_n: int = 5,
    factor: str = "mom_3m",
) -> BacktestResult:
    """Build a synthetic BacktestResult for testing."""
    holdings_history = []
    monthly_returns = []
    turnover_series = []
    ic_series = []
    cum = 1.0

    for i in range(n_periods):
        tickers = [f"{7200 + i * top_n + j}" for j in range(top_n)]
        holdings_history.append(
            HoldingsPeriod(
                date=f"2024-{i + 1:02d}-28",
                holdings=tickers,
                factor_values={t: 0.1 * j for j, t in enumerate(tickers)},
            )
        )
        ret = 0.01 + 0.001 * i
        cum *= 1 + ret
        monthly_returns.append(
            MonthlyReturn(
                period_start=f"2024-{i + 1:02d}-28",
                period_end=f"2024-{(i + 2) if i + 2 <= 12 else 1:02d}-28",
                monthly_return=ret,
                cumulative=cum,
            )
        )
        ic_series.append(0.05 + 0.01 * i)
        if i > 0:
            turnover_series.append(0.2)

    return BacktestResult(
        factor=factor,
        top_n=top_n,
        period="2024-01-28 ~ 2024-12-28",
        months=n_periods,
        performance=PerformanceMetrics(
            total_return=cum - 1,
            cagr=0.12,
            annualized_vol=0.15,
            sharpe_ratio=0.8,
            max_drawdown=-0.05,
        ),
        monthly_returns=monthly_returns,
        holdings_history=holdings_history,
        data_quality=DataQuality(
            total_rebalances=n_periods,
            skipped_rebalances=0,
            total_ticker_slots=n_periods * top_n,
            ffill_count=0,
            skip_count=0,
        ),
        factor_metrics=FactorMetrics(
            mean_ic=sum(ic_series) / len(ic_series),
            ic_series=ic_series,
            mean_turnover=sum(turnover_series) / len(turnover_series) if turnover_series else None,
            turnover_series=turnover_series,
        ),
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestComputePortfolioAnalytics:
    def test_returns_result(self) -> None:
        result = _make_result()
        pa = compute_portfolio_analytics(result)
        assert isinstance(pa, PortfolioAnalytics)

    def test_hhi_equal_weight(self) -> None:
        result = _make_result(top_n=5)
        pa = compute_portfolio_analytics(result)
        # Equal-weight HHI for 5 holdings: 5 * (1/5)^2 = 0.2
        assert abs(pa.mean_hhi - 0.2) < 1e-10

    def test_hhi_equal_weight_10(self) -> None:
        result = _make_result(top_n=10)
        pa = compute_portfolio_analytics(result)
        # Equal-weight HHI for 10 holdings: 10 * (1/10)^2 = 0.1
        assert abs(pa.mean_hhi - 0.1) < 1e-10

    def test_effective_n(self) -> None:
        result = _make_result(top_n=5)
        pa = compute_portfolio_analytics(result)
        # effective_n = 1/HHI = 1/0.2 = 5.0
        assert abs(pa.mean_effective_n - 5.0) < 1e-10

    def test_hhi_range(self) -> None:
        result = _make_result()
        pa = compute_portfolio_analytics(result)
        assert pa.min_hhi <= pa.mean_hhi <= pa.max_hhi

    def test_sector_weights_count(self) -> None:
        result = _make_result(n_periods=6)
        pa = compute_portfolio_analytics(result)
        assert len(pa.sector_weights) == 6

    def test_sector_weights_sum_to_one(self) -> None:
        result = _make_result()
        pa = compute_portfolio_analytics(result)
        for sw in pa.sector_weights:
            total = sum(sw.weights.values())
            assert abs(total - 1.0) < 1e-10

    def test_turnover_from_factor_metrics(self) -> None:
        result = _make_result()
        pa = compute_portfolio_analytics(result)
        assert pa.turnover_series == result.factor_metrics.turnover_series  # type: ignore[union-attr]
        assert pa.mean_turnover == result.factor_metrics.mean_turnover  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPortfolioAnalyticsEdgeCases:
    def test_no_holdings_raises(self) -> None:
        result = _make_result()
        result.holdings_history = []
        with pytest.raises(ValueError, match="No holdings history"):
            compute_portfolio_analytics(result)

    def test_no_factor_metrics(self) -> None:
        result = _make_result()
        result.factor_metrics = None
        pa = compute_portfolio_analytics(result)
        assert pa.turnover_series == []
        assert pa.mean_turnover is None

    def test_single_holding(self) -> None:
        result = _make_result(top_n=1)
        pa = compute_portfolio_analytics(result)
        # HHI for 1 holding: 1 * 1^2 = 1.0
        assert abs(pa.mean_hhi - 1.0) < 1e-10
        assert abs(pa.mean_effective_n - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestPortfolioSerialization:
    def test_model_dump_json_round_trip(self) -> None:
        result = _make_result()
        pa = compute_portfolio_analytics(result)
        data = pa.model_dump()
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert "mean_hhi" in restored
        assert "sector_weights" in restored

    def test_model_dump_has_required_fields(self) -> None:
        result = _make_result()
        pa = compute_portfolio_analytics(result)
        data = pa.model_dump()
        assert "mean_hhi" in data
        assert "min_hhi" in data
        assert "max_hhi" in data
        assert "mean_effective_n" in data
        assert "sector_weights" in data
        assert "turnover_series" in data
        assert "mean_turnover" in data


# ---------------------------------------------------------------------------
# IC Stats
# ---------------------------------------------------------------------------


class TestICStatsInBacktest:
    def test_ic_stats_present(self) -> None:
        """IC stats should be populated when factor_metrics is present."""
        from jpfin.metrics import compute_ic_stats

        series = [0.05, 0.10, -0.02, 0.08, 0.15]
        stats = compute_ic_stats(series)
        assert stats.mean_ic is not None
        assert stats.std_ic is not None
        assert stats.ic_ir is not None
        assert stats.ic_ir_annualized is not None
        assert stats.ic_hit_rate is not None
        assert stats.n_obs == 5

    def test_ic_stats_empty(self) -> None:
        from jpfin.metrics import compute_ic_stats

        stats = compute_ic_stats([])
        assert stats.mean_ic is None
        assert stats.std_ic is None
        assert stats.ic_ir is None
        assert stats.n_obs == 0

    def test_ic_stats_single(self) -> None:
        from jpfin.metrics import compute_ic_stats

        stats = compute_ic_stats([0.1])
        assert stats.mean_ic is not None
        assert stats.std_ic is None
        assert stats.ic_ir is None
        assert stats.n_obs == 1

    def test_ic_ir_formula(self) -> None:
        from jpfin.metrics import compute_ic_stats

        series = [0.1, 0.2, 0.3]
        stats = compute_ic_stats(series)
        assert stats.mean_ic is not None
        assert stats.std_ic is not None
        assert stats.ic_ir is not None
        expected_ir = stats.mean_ic / stats.std_ic
        assert abs(stats.ic_ir - expected_ir) < 1e-10

    def test_ic_ir_annualized_formula(self) -> None:
        from jpfin.metrics import compute_ic_stats

        series = [0.1, 0.2, 0.3]
        stats = compute_ic_stats(series)
        assert stats.ic_ir is not None
        assert stats.ic_ir_annualized is not None
        expected = stats.ic_ir * 12**0.5
        assert abs(stats.ic_ir_annualized - expected) < 1e-10

    def test_ic_hit_rate(self) -> None:
        from jpfin.metrics import compute_ic_stats

        series = [0.1, -0.05, 0.2, 0.0, 0.3]
        stats = compute_ic_stats(series)
        # 3 positive out of 5 (0.0 is not positive)
        assert stats.ic_hit_rate is not None
        assert abs(stats.ic_hit_rate - 0.6) < 1e-10

    def test_ic_stats_zero_std(self) -> None:
        from jpfin.metrics import compute_ic_stats

        stats = compute_ic_stats([0.1, 0.1, 0.1])
        assert stats.mean_ic is not None
        assert stats.std_ic == 0.0
        assert stats.ic_ir is None


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatPortfolioTable:
    def test_header_present(self) -> None:
        from jpfin.formatters import format_portfolio_table

        result = _make_result()
        pa = compute_portfolio_analytics(result)
        output = format_portfolio_table(pa)
        assert "Portfolio Analytics" in output

    def test_hhi_shown(self) -> None:
        from jpfin.formatters import format_portfolio_table

        result = _make_result()
        pa = compute_portfolio_analytics(result)
        output = format_portfolio_table(pa)
        assert "HHI" in output
        assert "Effective N" in output

    def test_turnover_shown(self) -> None:
        from jpfin.formatters import format_portfolio_table

        result = _make_result()
        pa = compute_portfolio_analytics(result)
        output = format_portfolio_table(pa)
        assert "Turnover" in output

    def test_sector_shown(self) -> None:
        from jpfin.formatters import format_portfolio_table

        result = _make_result()
        pa = compute_portfolio_analytics(result)
        output = format_portfolio_table(pa)
        assert "Sector Allocation" in output


# ---------------------------------------------------------------------------
# Backtest table IC display
# ---------------------------------------------------------------------------


class TestFormatBacktestTableIC:
    def test_ic_stats_in_backtest_table(self) -> None:
        """Backtest table should show IC stats when available."""
        from jpfin.formatters import format_backtest_table
        from jpfin.metrics import compute_ic_stats

        result = _make_result()
        # Ensure ic_stats is computed
        result.factor_metrics.ic_stats = compute_ic_stats(  # type: ignore[union-attr]
            result.factor_metrics.ic_series  # type: ignore[union-attr]
        )
        output = format_backtest_table(result)
        assert "Mean IC" in output
        assert "IC IR" in output


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestPortfolioCLI:
    def test_backtest_help_has_portfolio_flag(self) -> None:
        from click.testing import CliRunner

        from jpfin.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--portfolio-analytics" in result.output

    def test_run_help_has_portfolio_flag(self) -> None:
        from click.testing import CliRunner

        from jpfin.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--portfolio-analytics" in result.output
