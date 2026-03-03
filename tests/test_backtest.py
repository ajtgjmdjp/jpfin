"""Tests for backtest module."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from japan_finance_factors._models import PriceData

from jpfin.backtest import (
    _compute_benchmark_metrics,
    _ffill_close,
    _month_end_dates,
    _rebalance_dates,
    _spearman_rank_corr,
    load_prices_csv,
    run_backtest,
)
from jpfin.models import BacktestError, BacktestResult


def _make_price_data(
    ticker: str,
    n_days: int = 300,
    start_price: float = 1000,
    step: float = 1.0,
) -> PriceData:
    return PriceData(
        ticker=ticker,
        prices=[
            {
                "date": (date(2024, 1, 2) + timedelta(days=i)).isoformat(),
                "close": start_price + step * i,
            }
            for i in range(n_days)
        ],
    )


class TestMonthEndDates:
    def test_basic(self) -> None:
        dates = [date(2024, 1, i) for i in range(1, 32)]
        dates += [date(2024, 2, i) for i in range(1, 29)]
        ends = _month_end_dates(dates)
        assert date(2024, 1, 31) in ends
        assert date(2024, 2, 28) in ends

    def test_empty(self) -> None:
        assert _month_end_dates([]) == []

    def test_single_month(self) -> None:
        dates = [
            date(2024, 3, 1),
            date(2024, 3, 15),
            date(2024, 3, 31),
        ]
        ends = _month_end_dates(dates)
        assert len(ends) == 1
        assert ends[0] == date(2024, 3, 31)


class TestRebalanceDates:
    def test_monthly(self) -> None:
        dates = [date(2024, 1, i) for i in range(1, 32)]
        dates += [date(2024, 2, i) for i in range(1, 29)]
        ends = _rebalance_dates(dates, "monthly")
        assert date(2024, 1, 31) in ends
        assert date(2024, 2, 28) in ends

    def test_weekly(self) -> None:
        # 2024-01-01 is Monday, generate 3 weeks of dates
        dates = [date(2024, 1, i) for i in range(1, 22)]
        ends = _rebalance_dates(dates, "weekly")
        assert len(ends) >= 2  # at least 2 week boundaries

    def test_quarterly(self) -> None:
        # Jan through June
        dates = []
        for m in range(1, 7):
            for d in range(1, 29):
                dates.append(date(2024, m, d))
        dates.sort()
        ends = _rebalance_dates(dates, "quarterly")
        # March and June should have boundaries
        assert any(d.month == 3 for d in ends)
        assert any(d.month == 6 for d in ends)

    def test_empty(self) -> None:
        assert _rebalance_dates([], "monthly") == []

    def test_invalid_freq(self) -> None:
        with pytest.raises(ValueError, match="Unsupported rebalance frequency"):
            _rebalance_dates([date(2024, 1, 1)], "daily")

    def test_weekly_backtest(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1, rebalance_freq="weekly")
        assert isinstance(result, BacktestResult)
        # Weekly should have more rebalances than monthly
        monthly = run_backtest(price_data, "mom_3m", top_n=1, rebalance_freq="monthly")
        assert result.months >= monthly.months

    def test_quarterly_backtest(self) -> None:
        price_data = {
            "A": _make_price_data("A", 400, step=2.0),
            "B": _make_price_data("B", 400, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1, rebalance_freq="quarterly")
        assert isinstance(result, BacktestResult)
        assert result.months > 0


class TestLoadPricesCsv:
    def test_basic(self) -> None:
        csv_content = (
            "date,ticker,close\n2024-01-01,7203,2000\n2024-01-02,7203,2010\n2024-01-01,6758,1500\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as f:
            f.write(csv_content)
            f.flush()
            data = load_prices_csv(f.name)
        assert "7203" in data
        assert "6758" in data
        assert len(data["7203"].prices) == 2

    def test_extra_columns(self) -> None:
        csv_content = (
            "date,ticker,open,high,low,close,volume\n2024-01-01,7203,1990,2010,1980,2000,100000\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as f:
            f.write(csv_content)
            f.flush()
            data = load_prices_csv(f.name)
        assert data["7203"].prices[0]["open"] == 1990.0


class TestRunBacktest:
    def test_momentum_backtest(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, 1000, step=2.0),
            "B": _make_price_data("B", 300, 1000, step=1.0),
            "C": _make_price_data("C", 300, 1000, step=0.5),
        }
        result = run_backtest(price_data, "mom_3m", top_n=2)
        assert isinstance(result, BacktestResult)
        assert result.factor == "mom_3m"
        assert result.top_n == 2
        assert result.months > 0
        assert result.performance.total_return > 0  # All uptrending
        assert result.performance.max_drawdown <= 0

    def test_insufficient_data(self) -> None:
        price_data = {
            "A": PriceData(
                ticker="A",
                prices=[{"date": "2024-01-01", "close": 100}],
            ),
        }
        with pytest.raises(BacktestError):
            run_backtest(price_data, "mom_3m", top_n=1)

    def test_unsupported_factor(self) -> None:
        price_data = {"A": _make_price_data("A")}
        with pytest.raises(ValueError, match="Unsupported factor"):
            run_backtest(price_data, "roe", top_n=1)

    def test_holdings_history(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1)
        assert len(result.holdings_history) > 0
        for h in result.holdings_history:
            assert "A" in h.holdings

    def test_top_n_validation(self) -> None:
        price_data = {"A": _make_price_data("A")}
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            run_backtest(price_data, "mom_3m", top_n=0)
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            run_backtest(price_data, "mom_3m", top_n=-1)

    def test_model_dump_is_serializable(self) -> None:
        """BacktestResult.model_dump() should produce JSON-serializable dict."""
        import json

        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1)
        dumped = result.model_dump()
        json.dumps(dumped)  # should not raise

    def test_data_quality_present(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1)
        assert result.data_quality is not None
        assert result.data_quality.total_rebalances > 0
        assert result.data_quality.coverage > 0

    def test_ffill_limit_zero_strict(self) -> None:
        """With ffill_limit=0, only exact-date prices are used."""
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1, ffill_limit=0)
        assert isinstance(result, BacktestResult)
        # Should still produce results (continuous data)
        assert result.months > 0


class TestFfillClose:
    def test_exact_match(self) -> None:
        closes = {date(2024, 1, 10): 100.0, date(2024, 1, 11): 101.0}
        sorted_dates = [date(2024, 1, 10), date(2024, 1, 11)]
        assert _ffill_close(closes, date(2024, 1, 10), sorted_dates) == 100.0

    def test_forward_fill(self) -> None:
        closes = {date(2024, 1, 10): 100.0}
        sorted_dates = [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)]
        # date(2024, 1, 12) not in closes → fill from 2024-01-10
        assert _ffill_close(closes, date(2024, 1, 12), sorted_dates) == 100.0

    def test_beyond_limit(self) -> None:
        closes = {date(2024, 1, 1): 100.0}
        sorted_dates = [date(2024, 1, 1)] + [date(2024, 1, i) for i in range(2, 20)]
        # 15 days gap, limit=5 → None
        assert _ffill_close(closes, date(2024, 1, 15), sorted_dates, limit=5) is None

    def test_empty_closes(self) -> None:
        sorted_dates = [date(2024, 1, 10)]
        assert _ffill_close({}, date(2024, 1, 10), sorted_dates) is None


class TestSpearmanRankCorr:
    def test_perfect_positive(self) -> None:
        assert _spearman_rank_corr([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        assert _spearman_rank_corr([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_too_few_pairs(self) -> None:
        assert _spearman_rank_corr([1, 2], [3, 4]) is None

    def test_tied_values(self) -> None:
        result = _spearman_rank_corr([1, 1, 2, 3], [10, 20, 30, 40])
        assert result is not None
        assert -1.0 <= result <= 1.0


class TestFactorMetrics:
    def test_ic_present(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
            "C": _make_price_data("C", 300, step=0.5),
        }
        result = run_backtest(price_data, "mom_3m", top_n=2)
        assert result.factor_metrics is not None
        assert len(result.factor_metrics.ic_series) > 0
        assert result.factor_metrics.mean_ic is not None

    def test_turnover_present(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
            "C": _make_price_data("C", 300, step=0.5),
        }
        result = run_backtest(price_data, "mom_3m", top_n=2)
        assert result.factor_metrics is not None
        assert len(result.factor_metrics.turnover_series) > 0
        assert result.factor_metrics.mean_turnover is not None
        # Turnover is between 0 and 1
        for t in result.factor_metrics.turnover_series:
            assert 0.0 <= t <= 1.0

    def test_stable_holdings_low_turnover(self) -> None:
        """With consistent uptrend ordering, turnover should be low."""
        price_data = {
            "A": _make_price_data("A", 300, step=3.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1)
        assert result.factor_metrics is not None
        if result.factor_metrics.turnover_series:
            # A always has higher momentum, so holdings shouldn't change
            assert result.factor_metrics.mean_turnover == pytest.approx(0.0)


class TestBenchmarkMetrics:
    def test_compute_basic(self) -> None:
        portfolio = [0.05, 0.03, -0.02, 0.04]
        bench = [0.02, 0.01, -0.01, 0.02]
        bm = _compute_benchmark_metrics(portfolio, bench, "test")
        assert bm.benchmark_name == "test"
        assert bm.excess_return != 0.0
        assert bm.tracking_error >= 0.0

    def test_compute_empty(self) -> None:
        bm = _compute_benchmark_metrics([], [], "test")
        assert bm.excess_return == 0.0
        assert bm.tracking_error == 0.0

    def test_no_benchmark_by_default(self) -> None:
        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1)
        assert result.benchmark is None

    @patch("jpfin.backtest._fetch_benchmark_prices")
    def test_with_benchmark(self, mock_bm: MagicMock) -> None:
        # Create benchmark prices matching the backtest period
        bm_prices: dict[date, float] = {}
        base = 1000.0
        for i in range(400):
            d = date(2024, 1, 2) + timedelta(days=i)
            base *= 1.001
            bm_prices[d] = base
        mock_bm.return_value = bm_prices

        price_data = {
            "A": _make_price_data("A", 300, step=2.0),
            "B": _make_price_data("B", 300, step=1.0),
        }
        result = run_backtest(price_data, "mom_3m", top_n=1, benchmark="topix")
        assert result.benchmark is not None
        assert result.benchmark.benchmark_name == "topix"
        assert isinstance(result.benchmark.excess_return, float)

    def test_unknown_benchmark(self) -> None:
        from jpfin.backtest import _fetch_benchmark_prices

        with pytest.raises(ValueError, match="Unknown benchmark"):
            _fetch_benchmark_prices("nonexistent", date(2024, 1, 1), date(2024, 6, 1))
