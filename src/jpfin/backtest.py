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
from jpfin.models import (
    BacktestError,
    BacktestResult,
    BenchmarkMetrics,
    DataQuality,
    FactorMetrics,
    HoldingsPeriod,
    MonthlyReturn,
    PerformanceMetrics,
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


def _rebalance_dates(
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
        return _month_end_dates(dates)
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


def _next_trading_day(
    d: date,
    sorted_dates: list[date],
) -> date | None:
    """Return the first trading day strictly after d using bisect."""
    idx = bisect.bisect_right(sorted_dates, d)
    if idx < len(sorted_dates):
        return sorted_dates[idx]
    return None


def _ffill_close(
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


def _spearman_rank_corr(x: list[float], y: list[float]) -> float | None:
    """Compute Spearman rank correlation between two equal-length lists.

    Returns None if fewer than 3 pairs or zero variance in ranks.
    """
    n = len(x)
    if n < 3 or len(y) != n:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j]] == vals[indexed[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov: float = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    var_x: float = sum((a - mean_rx) ** 2 for a in rx)
    var_y: float = sum((b - mean_ry) ** 2 for b in ry)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return float(cov / denom)


def _compute_performance(
    returns: list[float],
) -> PerformanceMetrics:
    """Compute summary statistics from monthly return series."""
    portfolio_value = 1.0
    for r in returns:
        portfolio_value *= 1 + r

    n_months = len(returns)

    if n_months == 0 or portfolio_value <= 0:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_vol=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
        )

    total_return = portfolio_value - 1.0
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

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
    )


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

    closes: dict[date, float] = {}
    for idx, row in df.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        close_val = row["Close"]
        if close_val is not None and close_val == close_val:  # not NaN
            closes[d] = float(close_val)
    return closes


def _compute_benchmark_metrics(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    name: str,
) -> BenchmarkMetrics:
    """Compute excess return and information ratio vs benchmark.

    Args:
        portfolio_returns: Monthly portfolio returns.
        benchmark_returns: Monthly benchmark returns (same length).
        name: Benchmark name for display.

    Returns:
        BenchmarkMetrics with excess return, tracking error, IR.
    """
    n = min(len(portfolio_returns), len(benchmark_returns))
    if n == 0:
        return BenchmarkMetrics(
            benchmark_name=name,
            benchmark_return=0.0,
            excess_return=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
        )

    # Compute cumulative benchmark return
    bm_value = 1.0
    for r in benchmark_returns[:n]:
        bm_value *= 1 + r
    benchmark_total = bm_value - 1.0

    # Compute cumulative portfolio return
    pf_value = 1.0
    for r in portfolio_returns[:n]:
        pf_value *= 1 + r
    portfolio_total = pf_value - 1.0

    excess = portfolio_total - benchmark_total

    # Tracking error = std of monthly excess returns
    excess_series = [
        p - b for p, b in zip(portfolio_returns[:n], benchmark_returns[:n], strict=True)
    ]
    te = statistics.stdev(excess_series) * 12**0.5 if n >= 2 else 0.0

    ir = excess / te if te > 0 else 0.0

    return BenchmarkMetrics(
        benchmark_name=name,
        benchmark_return=benchmark_total,
        excess_return=excess,
        tracking_error=te,
        information_ratio=ir,
    )


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
    if isinstance(factor_fn, list):
        from jpfin.composite import compute_composite_scores, validate_composite_args

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
        raise BacktestError("Insufficient date range for backtest")

    rebalance_dates = _rebalance_dates(sorted_dates, rebalance_freq)
    if len(rebalance_dates) < 2:
        raise BacktestError("Need at least 2 rebalance periods")

    # Run backtest
    holdings_history: list[HoldingsPeriod] = []
    monthly_returns: list[MonthlyReturn] = []
    portfolio_value = 1.0
    if not multi_factor:
        higher_is_better = HIGHER_IS_BETTER.get(factor_fn, True)  # type: ignore[arg-type]

    # Data quality counters
    total_rebalances = len(rebalance_dates) - 1
    skipped_rebalances = 0
    total_ticker_slots = 0
    ffill_count = 0
    skip_count = 0

    # Factor metrics accumulators
    ic_series: list[float] = []
    turnover_series: list[float] = []
    prev_holdings: set[str] = set()

    for i in range(total_rebalances):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]

        # Filter prices up to rebalance date
        filtered_price_data: dict[str, PriceData] = {}
        for ticker, pd in price_data.items():
            filtered_prices = [
                p
                for p in pd._sorted_prices()
                if (d := parse_date(p.get("date"))) is not None and d <= rebal_date
            ]
            if filtered_prices:
                filtered_price_data[ticker] = PriceData(
                    ticker=ticker,
                    prices=filtered_prices,
                )

        if multi_factor:
            # Multi-factor: compute all factors, then composite z-score
            from jpfin.factor_registry import compute_price_factors

            ticker_factor_vals: dict[str, dict[str, float | None]] = {}
            for ticker, fpd in filtered_price_data.items():
                ticker_factor_vals[ticker] = compute_price_factors(fpd, factors)

            factor_values = compute_composite_scores(
                ticker_factor_vals, factors, validated_weights
            )
        else:
            # Single-factor: compute and rank directly
            factor_values_raw: list[tuple[str, float]] = []
            for ticker, fpd in filtered_price_data.items():
                val = compute_factor(fpd)
                if val is not None:
                    factor_values_raw.append((ticker, val))

            factor_values_raw.sort(key=lambda x: x[1], reverse=higher_is_better)
            factor_values = factor_values_raw

        if not factor_values:
            skipped_rebalances += 1
            continue

        # Select top N
        selected = [t for t, _ in factor_values[:top_n]]

        # Use next trading day after signal for execution
        exec_date = _next_trading_day(rebal_date, sorted_dates)
        next_exec = _next_trading_day(next_rebal, sorted_dates)
        if exec_date is None or next_exec is None:
            skipped_rebalances += 1
            continue

        # Calculate equal-weight return for the holding period
        period_returns: list[float] = []
        skipped_tickers: list[str] = []
        for ticker in selected:
            total_ticker_slots += 1
            closes = ticker_close.get(ticker, {})
            start_price = _ffill_close(closes, exec_date, sorted_dates, ffill_limit)
            end_price = _ffill_close(closes, next_exec, sorted_dates, ffill_limit)
            if start_price is not None and end_price is not None and start_price > 0:
                # Track whether forward-fill was used
                if exec_date not in closes or next_exec not in closes:
                    ffill_count += 1
                ret = (end_price - start_price) / start_price
                period_returns.append(ret)
            else:
                skip_count += 1
                skipped_tickers.append(ticker)

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
        fwd_factors: list[float] = []
        fwd_returns: list[float] = []
        for ticker, fval in factor_values:
            closes = ticker_close.get(ticker, {})
            sp = _ffill_close(closes, exec_date, sorted_dates, ffill_limit)
            ep = _ffill_close(closes, next_exec, sorted_dates, ffill_limit)
            if sp is not None and ep is not None and sp > 0:
                fwd_factors.append(fval)
                fwd_returns.append((ep - sp) / sp)
        ic = _spearman_rank_corr(fwd_factors, fwd_returns)
        if ic is not None:
            ic_series.append(ic)

        # --- Turnover ---
        current_set = set(selected)
        if prev_holdings:
            changed = len(current_set.symmetric_difference(prev_holdings))
            turnover_series.append(changed / (2 * top_n))
        prev_holdings = current_set

    returns = [m.monthly_return for m in monthly_returns]
    performance = _compute_performance(returns)

    # --- Benchmark comparison ---
    benchmark_metrics: BenchmarkMetrics | None = None
    if benchmark and monthly_returns:
        bm_closes = _fetch_benchmark_prices(
            benchmark,
            rebalance_dates[0],
            rebalance_dates[-1],
        )
        bm_sorted = sorted(bm_closes.keys())

        bm_returns: list[float] = []
        for mr in monthly_returns:
            start_d = parse_date(mr.period_start)
            end_d = parse_date(mr.period_end)
            if start_d is None or end_d is None:
                bm_returns.append(0.0)
                continue
            exec_d = _next_trading_day(start_d, bm_sorted)
            next_d = _next_trading_day(end_d, bm_sorted)
            if exec_d is None or next_d is None:
                bm_returns.append(0.0)
                continue
            sp = _ffill_close(bm_closes, exec_d, bm_sorted, ffill_limit)
            ep = _ffill_close(bm_closes, next_d, bm_sorted, ffill_limit)
            if sp and ep and sp > 0:
                bm_returns.append((ep - sp) / sp)
            else:
                bm_returns.append(0.0)

        benchmark_metrics = _compute_benchmark_metrics(returns, bm_returns, benchmark)

    return BacktestResult(
        factor=factor_label,
        top_n=top_n,
        period=f"{rebalance_dates[0].isoformat()} ~ {rebalance_dates[-1].isoformat()}",
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
            mean_turnover=(
                sum(turnover_series) / len(turnover_series) if turnover_series else None
            ),
            turnover_series=turnover_series,
        ),
        benchmark=benchmark_metrics,
    )
