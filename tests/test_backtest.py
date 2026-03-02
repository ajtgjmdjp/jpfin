"""Tests for backtest module."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta

import pytest
from japan_finance_factors._models import PriceData

from jpfin.backtest import _ffill_close, _month_end_dates, load_prices_csv, run_backtest
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
