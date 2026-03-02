"""Simple factor-based backtesting.

Long-only, monthly rebalance, equal-weight, top N by factor.
Price data sourced from yfinance or user-provided CSV.
"""

from __future__ import annotations

import bisect
import contextlib
import csv
import statistics
from datetime import date
from pathlib import Path
from typing import Any

from japan_finance_factors._models import PriceData

from jpfin._utils import parse_date
from jpfin.factor_registry import HIGHER_IS_BETTER, PRICE_FACTOR_FNS


def load_prices_csv(path: str | Path) -> dict[str, PriceData]:
    """Load price data from CSV file.

    Expected CSV format: date,ticker,close (at minimum).
    Additional columns (open, high, low, volume) are optional.

    Returns:
        Dict mapping ticker → PriceData.
    """
    path = Path(path)
    ticker_prices: dict[str, list[dict[str, Any]]] = {}

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            if not ticker:
                continue
            price_row: dict[str, Any] = {
                "date": row.get("date", "").strip(),
            }
            for field in ("close", "open", "high", "low", "volume"):
                val = row.get(field)
                if val is not None:
                    with contextlib.suppress(ValueError):
                        price_row[field] = float(val)
            ticker_prices.setdefault(ticker, []).append(price_row)

    return {
        ticker: PriceData(ticker=ticker, prices=prices) for ticker, prices in ticker_prices.items()
    }


def _month_end_dates(dates: list[date]) -> list[date]:
    """Extract month-end dates from a sorted list."""
    if not dates:
        return []
    ends: list[date] = []
    for i in range(len(dates) - 1):
        if dates[i].month != dates[i + 1].month:
            ends.append(dates[i])
    ends.append(dates[-1])
    return ends


def _next_trading_day(
    d: date,
    sorted_dates: list[date],
) -> date | None:
    """Return the first trading day strictly after d using bisect."""
    idx = bisect.bisect_right(sorted_dates, d)
    if idx < len(sorted_dates):
        return sorted_dates[idx]
    return None


def _compute_performance(
    returns: list[float],
) -> dict[str, float]:
    """Compute summary statistics from monthly return series."""
    portfolio_value = 1.0
    for r in returns:
        portfolio_value *= 1 + r

    n_months = len(returns)
    total_return = portfolio_value - 1.0

    if n_months == 0 or portfolio_value <= 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_vol": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    cagr = portfolio_value ** (12.0 / n_months) - 1.0

    if n_months >= 2:
        monthly_vol = statistics.stdev(returns)
        ann_vol = monthly_vol * (12**0.5)
        sharpe = cagr / ann_vol if ann_vol > 0 else 0.0
    else:
        ann_vol = 0.0
        sharpe = 0.0

    # Max drawdown from cumulative series
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))
    peak = cumulative[0]
    max_dd = 0.0
    for c in cumulative:
        peak = max(peak, c)
        dd = (c - peak) / peak
        max_dd = min(max_dd, dd)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def run_backtest(
    price_data: dict[str, PriceData],
    factor_fn: str = "mom_3m",
    *,
    top_n: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """Run a simple long-only monthly rebalance backtest.

    Strategy: At each month-end, compute the specified price-based
    factor for all tickers, select top N, hold equal-weight for the
    next month.

    Args:
        price_data: Dict mapping ticker → PriceData.
        factor_fn: Factor to rank by. Must be a price-based factor.
        top_n: Number of top tickers to hold (>= 1).
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        Dict with performance metrics and return series.

    Raises:
        ValueError: If factor_fn is unsupported or top_n < 1.
    """
    if factor_fn not in PRICE_FACTOR_FNS:
        supported = list(PRICE_FACTOR_FNS.keys())
        raise ValueError(f"Unsupported factor: {factor_fn}. Use one of {supported}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    compute_factor = PRICE_FACTOR_FNS[factor_fn]

    # Build ticker → {date → close} lookup
    ticker_close: dict[str, dict[date, float]] = {}
    all_dates: set[date] = set()
    for ticker, pd in price_data.items():
        closes_by_date: dict[date, float] = {}
        for p in pd._sorted_prices():
            d = parse_date(p.get("date"))
            c = p.get("close")
            if d is None or c is None:
                continue
            closes_by_date[d] = float(c)
            all_dates.add(d)
        ticker_close[ticker] = closes_by_date

    sorted_dates = sorted(all_dates)
    if start_date:
        sorted_dates = [d for d in sorted_dates if d >= start_date]
    if end_date:
        sorted_dates = [d for d in sorted_dates if d <= end_date]

    if len(sorted_dates) < 2:
        return {"error": "Insufficient date range for backtest"}

    rebalance_dates = _month_end_dates(sorted_dates)
    if len(rebalance_dates) < 2:
        return {"error": "Need at least 2 months of data"}

    # Run backtest
    holdings_history: list[dict[str, Any]] = []
    monthly_returns: list[dict[str, Any]] = []
    portfolio_value = 1.0
    higher_is_better = HIGHER_IS_BETTER.get(factor_fn, True)

    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]

        # Compute factors for all tickers at rebalance date
        factor_values: list[tuple[str, float]] = []
        for ticker, pd in price_data.items():
            filtered_prices = [
                p
                for p in pd._sorted_prices()
                if (d := parse_date(p.get("date"))) is not None and d <= rebal_date
            ]
            if not filtered_prices:
                continue
            filtered_pd = PriceData(
                ticker=ticker,
                prices=filtered_prices,
            )
            val = compute_factor(filtered_pd)
            if val is not None:
                factor_values.append((ticker, val))

        if not factor_values:
            continue

        # Rank and select top N
        factor_values.sort(
            key=lambda x: x[1],
            reverse=higher_is_better,
        )
        selected = [t for t, _ in factor_values[:top_n]]

        # Use next trading day after signal for execution
        exec_date = _next_trading_day(rebal_date, sorted_dates)
        next_exec = _next_trading_day(next_rebal, sorted_dates)
        if exec_date is None or next_exec is None:
            continue

        # Calculate equal-weight return for the holding period
        period_returns: list[float] = []
        skipped_tickers: list[str] = []
        for ticker in selected:
            closes = ticker_close.get(ticker, {})
            start_price = closes.get(exec_date)
            end_price = closes.get(next_exec)
            if start_price and end_price and start_price > 0:
                ret = (end_price - start_price) / start_price
                period_returns.append(ret)
            else:
                skipped_tickers.append(ticker)

        avg_return = sum(period_returns) / len(period_returns) if period_returns else 0.0
        portfolio_value *= 1 + avg_return

        period_info: dict[str, Any] = {
            "date": rebal_date.isoformat(),
            "holdings": selected,
            "factor_values": {t: v for t, v in factor_values[:top_n]},
        }
        if skipped_tickers:
            period_info["skipped"] = skipped_tickers
        holdings_history.append(period_info)

        monthly_returns.append(
            {
                "period_start": rebal_date.isoformat(),
                "period_end": next_rebal.isoformat(),
                "return": avg_return,
                "cumulative": portfolio_value,
            }
        )

    returns = [m["return"] for m in monthly_returns]
    performance = _compute_performance(returns)

    return {
        "factor": factor_fn,
        "top_n": top_n,
        "period": (f"{rebalance_dates[0].isoformat()} ~ {rebalance_dates[-1].isoformat()}"),
        "months": len(returns),
        "performance": performance,
        "monthly_returns": monthly_returns,
        "holdings_history": holdings_history,
    }
