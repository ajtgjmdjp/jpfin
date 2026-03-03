"""Simple factor-based backtesting.

Long-only, monthly rebalance, equal-weight, top N by factor.
Price data sourced from yfinance or user-provided CSV.
"""

from __future__ import annotations

import bisect
import contextlib
import csv
from datetime import date
from pathlib import Path
from typing import Any

from japan_finance_factors._models import PriceData

from jpfin._utils import parse_date
from jpfin.factor_registry import HIGHER_IS_BETTER, PRICE_FACTOR_FNS
from jpfin.metrics import (
    compute_benchmark_metrics,
    compute_ic_stats,
    compute_performance,
    spearman_rank_corr,
)
from jpfin.models import (
    BacktestError,
    BacktestResult,
    BenchmarkMetrics,
    DataQuality,
    FactorMetrics,
    HoldingsPeriod,
    MonthlyReturn,
)

# Well-known benchmark tickers for yfinance
BENCHMARK_TICKERS: dict[str, str] = {
    "topix": "1306.T",
    "nikkei225": "^N225",
    "nk225": "^N225",
}


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


def month_end_dates(dates: list[date]) -> list[date]:
    """Extract month-end dates from a sorted list."""
    if not dates:
        return []
    ends: list[date] = []
    for i in range(len(dates) - 1):
        if dates[i].month != dates[i + 1].month:
            ends.append(dates[i])
    ends.append(dates[-1])
    return ends


def rebalance_dates(
    dates: list[date],
    freq: str = "monthly",
) -> list[date]:
    """Extract rebalance dates from a sorted list of trading dates.

    Args:
        dates: Sorted trading dates.
        freq: ``"weekly"``, ``"monthly"``, or ``"quarterly"``.

    Returns:
        List of rebalance dates.
    """
    if not dates:
        return []
    if freq == "monthly":
        return month_end_dates(dates)
    if freq == "quarterly":
        ends: list[date] = []
        for i in range(len(dates) - 1):
            d0, d1 = dates[i], dates[i + 1]
            if d0.month != d1.month and d0.month % 3 == 0:
                ends.append(d0)
        ends.append(dates[-1])
        return ends
    if freq == "weekly":
        ends = []
        for i in range(len(dates) - 1):
            if dates[i].isocalendar()[1] != dates[i + 1].isocalendar()[1]:
                ends.append(dates[i])
        ends.append(dates[-1])
        return ends
    raise ValueError(f"Unsupported rebalance frequency: {freq}. Use weekly/monthly/quarterly")


def next_trading_day(
    d: date,
    sorted_dates: list[date],
) -> date | None:
    """Return the first trading day strictly after d using bisect."""
    idx = bisect.bisect_right(sorted_dates, d)
    if idx < len(sorted_dates):
        return sorted_dates[idx]
    return None


def ffill_close(
    closes: dict[date, float],
    target: date,
    sorted_dates: list[date],
    limit: int = 5,
) -> float | None:
    """Return close on *target* or forward-fill from the most recent prior day.

    Looks back up to *limit* trading days before *target* for a valid close.
    Returns ``None`` if no close found within the window.
    """
    if target in closes:
        return closes[target]
    idx = bisect.bisect_left(sorted_dates, target)
    # Walk backwards from idx-1
    for j in range(idx - 1, max(idx - 1 - limit, -1), -1):
        if j < 0:
            break
        d = sorted_dates[j]
        if d in closes:
            return closes[d]
    return None


# ---------------------------------------------------------------------------
# Pre-indexed price data for efficient date-based filtering
# ---------------------------------------------------------------------------


class _TickerPriceIndex:
    """Pre-parsed and sorted price data for a single ticker.

    Enables O(log n) date-based filtering via bisect instead of
    O(n) list scans with repeated ``parse_date()`` calls.
    """

    __slots__ = ("closes", "dates", "prices", "ticker")

    def __init__(self, ticker: str, pd: PriceData) -> None:
        self.ticker = ticker
        pairs: list[tuple[date, dict[str, Any]]] = []
        closes: dict[date, float] = {}
        for p in pd._sorted_prices():
            d = parse_date(p.get("date"))
            if d is None:
                continue
            pairs.append((d, p))
            c = p.get("close")
            if c is not None:
                closes[d] = float(c)
        pairs.sort(key=lambda x: x[0])
        self.dates: list[date] = [d for d, _ in pairs]
        self.prices: list[dict[str, Any]] = [p for _, p in pairs]
        self.closes = closes

    def filter_up_to(self, cutoff: date) -> PriceData | None:
        """Return PriceData with prices up to *cutoff* (inclusive).

        Uses bisect for O(log n) cutoff instead of O(n) list scan.
        """
        idx = bisect.bisect_right(self.dates, cutoff)
        if idx == 0:
            return None
        return PriceData(ticker=self.ticker, prices=self.prices[:idx])


def _build_price_index(
    price_data: dict[str, PriceData],
) -> tuple[dict[str, _TickerPriceIndex], dict[str, dict[date, float]], list[date]]:
    """Pre-process price data into indexed structures.

    Returns:
        Tuple of (ticker_index, ticker_close, sorted_all_dates).
    """
    ticker_index: dict[str, _TickerPriceIndex] = {}
    ticker_close: dict[str, dict[date, float]] = {}
    all_dates: set[date] = set()

    for ticker, pd in price_data.items():
        idx = _TickerPriceIndex(ticker, pd)
        ticker_index[ticker] = idx
        ticker_close[ticker] = idx.closes
        all_dates.update(idx.dates)

    return ticker_index, ticker_close, sorted(all_dates)


def _fetch_benchmark_prices(
    benchmark: str,
    start: date,
    end: date,
) -> dict[date, float]:
    """Fetch benchmark daily close prices from yfinance.

    Args:
        benchmark: Benchmark name (e.g. ``"topix"``, ``"nikkei225"``).
        start: Start date.
        end: End date.

    Returns:
        Dict mapping date to close price.

    Raises:
        ValueError: If benchmark name is unknown.
    """
    import yfinance as yf

    yf_ticker = BENCHMARK_TICKERS.get(benchmark.lower())
    if yf_ticker is None:
        available = list(BENCHMARK_TICKERS.keys())
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {available}")

    from datetime import timedelta

    df = yf.download(
        yf_ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=5)).isoformat(),
        interval="1d",
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No benchmark data for {benchmark} ({yf_ticker})")

    # Extract Close series — handle both MultiIndex and flat columns
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        close_series = df[("Close", yf_ticker)]
    elif "Close" in df.columns:
        close_series = df["Close"]
    else:
        raise ValueError(f"No 'Close' column in benchmark data for {benchmark}")

    closes: dict[date, float] = {}
    for idx, val in close_series.items():
        d = idx.date() if hasattr(idx, "date") else idx
        if pd.notna(val):
            closes[d] = float(val)
    return closes


# ---------------------------------------------------------------------------
# Extracted helpers for run_backtest (pure functions, no side effects)
# ---------------------------------------------------------------------------


def _compute_factor_rankings(
    filtered_price_data: dict[str, PriceData],
    *,
    multi_factor: bool,
    factors: list[str] | None = None,
    validated_weights: list[float] | None = None,
    compute_factor: Any = None,
    higher_is_better: bool = True,
) -> list[tuple[str, float]]:
    """Compute factor values and return ranked list of (ticker, score).

    Returns empty list if no tickers have valid factor values.
    """
    if multi_factor:
        from jpfin.composite import compute_composite_scores
        from jpfin.factor_registry import compute_price_factors

        assert factors is not None
        assert validated_weights is not None

        ticker_factor_vals: dict[str, dict[str, float | None]] = {}
        for ticker, fpd in filtered_price_data.items():
            ticker_factor_vals[ticker] = compute_price_factors(fpd, factors)

        return compute_composite_scores(ticker_factor_vals, factors, validated_weights)

    # Single-factor: compute and rank directly
    factor_values_raw: list[tuple[str, float]] = []
    for ticker, fpd in filtered_price_data.items():
        val = compute_factor(fpd)
        if val is not None:
            factor_values_raw.append((ticker, val))

    factor_values_raw.sort(key=lambda x: x[1], reverse=higher_is_better)
    return factor_values_raw


def _compute_period_returns(
    selected: list[str],
    ticker_close: dict[str, dict[date, float]],
    exec_date: date,
    next_exec: date,
    sorted_dates: list[date],
    ffill_limit: int,
) -> tuple[list[float], list[str], int, int]:
    """Compute equal-weight returns for a holding period.

    Returns:
        Tuple of (period_returns, skipped_tickers, ffill_count, skip_count).
    """
    period_returns: list[float] = []
    skipped_tickers: list[str] = []
    ffill_count = 0
    skip_count = 0

    for ticker in selected:
        closes = ticker_close.get(ticker, {})
        start_price = ffill_close(closes, exec_date, sorted_dates, ffill_limit)
        end_price = ffill_close(closes, next_exec, sorted_dates, ffill_limit)
        if start_price is not None and end_price is not None and start_price > 0:
            if exec_date not in closes or next_exec not in closes:
                ffill_count += 1
            ret = (end_price - start_price) / start_price
            period_returns.append(ret)
        else:
            skip_count += 1
            skipped_tickers.append(ticker)

    return period_returns, skipped_tickers, ffill_count, skip_count


def _compute_period_ic(
    factor_values: list[tuple[str, float]],
    ticker_close: dict[str, dict[date, float]],
    exec_date: date,
    next_exec: date,
    sorted_dates: list[date],
    ffill_limit: int,
) -> float | None:
    """Compute IC (rank correlation of factor vs forward return) for one period."""
    fwd_factors: list[float] = []
    fwd_returns: list[float] = []
    for ticker, fval in factor_values:
        closes = ticker_close.get(ticker, {})
        sp = ffill_close(closes, exec_date, sorted_dates, ffill_limit)
        ep = ffill_close(closes, next_exec, sorted_dates, ffill_limit)
        if sp is not None and ep is not None and sp > 0:
            fwd_factors.append(fval)
            fwd_returns.append((ep - sp) / sp)
    return spearman_rank_corr(fwd_factors, fwd_returns)


def run_backtest(
    price_data: dict[str, PriceData],
    factor_fn: str | list[str] = "mom_3m",
    *,
    weights: list[float] | None = None,
    benchmark: str | None = None,
    top_n: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
    ffill_limit: int = 5,
    rebalance_freq: str = "monthly",
) -> BacktestResult:
    """Run a simple long-only rebalance backtest.

    Strategy: At each rebalance date, compute the specified price-based
    factor(s) for all tickers, select top N, hold equal-weight until the
    next rebalance.

    Args:
        price_data: Dict mapping ticker → PriceData.
        factor_fn: Factor to rank by. A single string for single-factor,
            or a list of strings for multi-factor composite scoring.
        weights: Weights for multi-factor composite. ``None`` for equal.
            Ignored when ``factor_fn`` is a string.
        benchmark: Benchmark name for comparison (e.g. ``"topix"``).
            Fetches benchmark prices via yfinance and computes
            excess return, tracking error, and information ratio.
        top_n: Number of top tickers to hold (>= 1).
        start_date: Backtest start date.
        end_date: Backtest end date.
        ffill_limit: Max trading days to look back for execution
            prices when no close exists on the exact date. Set to 0
            to disable forward-fill (strict mode).
        rebalance_freq: ``"weekly"``, ``"monthly"``, or ``"quarterly"``.

    Returns:
        BacktestResult with performance metrics and return series.

    Raises:
        ValueError: If factor_fn is unsupported or top_n < 1.
        BacktestError: If insufficient data to run backtest.
    """
    # Determine single vs multi-factor mode
    factors: list[str] | None = None
    validated_weights: list[float] | None = None
    compute_factor: Any = None
    higher_is_better = True

    if isinstance(factor_fn, list):
        from jpfin.composite import validate_composite_args

        factors, validated_weights = validate_composite_args(factor_fn, weights)
        multi_factor = True
        factor_label = "+".join(factors)
    else:
        if factor_fn not in PRICE_FACTOR_FNS:
            supported = list(PRICE_FACTOR_FNS.keys())
            raise ValueError(f"Unsupported factor: {factor_fn}. Use one of {supported}")
        multi_factor = False
        factor_label = factor_fn
        compute_factor = PRICE_FACTOR_FNS[factor_fn]

    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    if not price_data:
        raise BacktestError("price_data cannot be empty")

    # Build pre-indexed price data for efficient lookups
    ticker_index, ticker_close, sorted_dates = _build_price_index(price_data)
    if start_date:
        sorted_dates = [d for d in sorted_dates if d >= start_date]
    if end_date:
        sorted_dates = [d for d in sorted_dates if d <= end_date]

    if len(sorted_dates) < 2:
        raise BacktestError("Insufficient date range for backtest")

    rebal_schedule = rebalance_dates(sorted_dates, rebalance_freq)
    if len(rebal_schedule) < 2:
        raise BacktestError("Need at least 2 rebalance periods")

    # Run backtest
    holdings_history: list[HoldingsPeriod] = []
    monthly_returns: list[MonthlyReturn] = []
    portfolio_value = 1.0
    if not multi_factor:
        higher_is_better = HIGHER_IS_BETTER.get(factor_label, True)

    # Data quality counters
    total_rebalances = len(rebal_schedule) - 1
    skipped_rebalances = 0
    total_ticker_slots = 0
    ffill_count = 0
    skip_count = 0

    # Factor metrics accumulators
    ic_series: list[float] = []
    turnover_series: list[float] = []
    prev_holdings: set[str] = set()

    for i in range(total_rebalances):
        rebal_date = rebal_schedule[i]
        next_rebal = rebal_schedule[i + 1]

        # Filter prices up to rebalance date (O(log n) via bisect)
        filtered_price_data: dict[str, PriceData] = {}
        for ticker, tidx in ticker_index.items():
            fpd = tidx.filter_up_to(rebal_date)
            if fpd is not None:
                filtered_price_data[ticker] = fpd

        factor_values = _compute_factor_rankings(
            filtered_price_data,
            multi_factor=multi_factor,
            factors=factors if multi_factor else None,
            validated_weights=validated_weights if multi_factor else None,
            compute_factor=compute_factor if not multi_factor else None,
            higher_is_better=higher_is_better if not multi_factor else True,
        )

        if not factor_values:
            skipped_rebalances += 1
            continue

        # Select top N
        selected = [t for t, _ in factor_values[:top_n]]

        # Use next trading day after signal for execution
        exec_date = next_trading_day(rebal_date, sorted_dates)
        next_exec = next_trading_day(next_rebal, sorted_dates)
        if exec_date is None or next_exec is None:
            skipped_rebalances += 1
            continue

        # Calculate equal-weight return for the holding period
        period_returns, skipped_tickers, period_ffill, period_skip = _compute_period_returns(
            selected, ticker_close, exec_date, next_exec, sorted_dates, ffill_limit
        )
        total_ticker_slots += len(selected)
        ffill_count += period_ffill
        skip_count += period_skip

        avg_return = sum(period_returns) / len(period_returns) if period_returns else 0.0
        portfolio_value *= 1 + avg_return

        holdings_history.append(
            HoldingsPeriod(
                date=rebal_date.isoformat(),
                holdings=selected,
                factor_values={t: v for t, v in factor_values[:top_n]},
                skipped=skipped_tickers or None,
            )
        )

        monthly_returns.append(
            MonthlyReturn(
                period_start=rebal_date.isoformat(),
                period_end=next_rebal.isoformat(),
                monthly_return=avg_return,
                cumulative=portfolio_value,
            )
        )

        # --- IC: rank corr(factor, forward return) across ALL tickers ---
        ic = _compute_period_ic(
            factor_values, ticker_close, exec_date, next_exec, sorted_dates, ffill_limit
        )
        if ic is not None:
            ic_series.append(ic)

        # --- Turnover ---
        current_set = set(selected)
        if prev_holdings:
            changed = len(current_set.symmetric_difference(prev_holdings))
            turnover_series.append(changed / (2 * top_n))
        prev_holdings = current_set

    returns = [m.monthly_return for m in monthly_returns]
    performance = compute_performance(returns)

    # --- Benchmark comparison ---
    benchmark_metrics: BenchmarkMetrics | None = None
    if benchmark and monthly_returns:
        bm_closes = _fetch_benchmark_prices(
            benchmark,
            rebal_schedule[0],
            rebal_schedule[-1],
        )
        bm_sorted = sorted(bm_closes.keys())

        bm_returns: list[float] = []
        for mr in monthly_returns:
            start_d = parse_date(mr.period_start)
            end_d = parse_date(mr.period_end)
            if start_d is None or end_d is None:
                bm_returns.append(0.0)
                continue
            exec_d = next_trading_day(start_d, bm_sorted)
            next_d = next_trading_day(end_d, bm_sorted)
            if exec_d is None or next_d is None:
                bm_returns.append(0.0)
                continue
            sp = ffill_close(bm_closes, exec_d, bm_sorted, ffill_limit)
            ep = ffill_close(bm_closes, next_d, bm_sorted, ffill_limit)
            if sp and ep and sp > 0:
                bm_returns.append((ep - sp) / sp)
            else:
                bm_returns.append(0.0)

        benchmark_metrics = compute_benchmark_metrics(returns, bm_returns, benchmark)

    return BacktestResult(
        factor=factor_label,
        top_n=top_n,
        period=f"{rebal_schedule[0].isoformat()} ~ {rebal_schedule[-1].isoformat()}",
        months=len(returns),
        performance=performance,
        monthly_returns=monthly_returns,
        holdings_history=holdings_history,
        data_quality=DataQuality(
            total_rebalances=total_rebalances,
            skipped_rebalances=skipped_rebalances,
            total_ticker_slots=total_ticker_slots,
            ffill_count=ffill_count,
            skip_count=skip_count,
        ),
        factor_metrics=FactorMetrics(
            mean_ic=sum(ic_series) / len(ic_series) if ic_series else None,
            ic_series=ic_series,
            ic_stats=compute_ic_stats(ic_series),
            mean_turnover=(
                sum(turnover_series) / len(turnover_series) if turnover_series else None
            ),
            turnover_series=turnover_series,
        ),
        benchmark=benchmark_metrics,
    )
