"""Factor decay analysis — IC term structure across lags.

Computes Spearman rank IC between factor values at time *t* and forward
returns at lag 1, 2, ..., *max_lag* months.  This quantifies how quickly
a factor's predictive power decays, answering: "Is this short-term alpha
or does it persist?"

Standalone module — does NOT modify or depend on the backtest loop.
"""

from __future__ import annotations

import statistics
from datetime import date

from japan_finance_factors._models import PriceData

from jpfin._utils import parse_date
from jpfin.backtest import ffill_close, next_trading_day, rebalance_dates
from jpfin.factor_registry import PRICE_FACTOR_FNS
from jpfin.metrics import spearman_rank_corr
from jpfin.models import DecayLag, FactorDecayResult


def compute_decay(
    price_data: dict[str, PriceData],
    factor_fn: str = "mom_3m",
    *,
    max_lag: int = 6,
    rebalance_freq: str = "monthly",
    start_date: date | None = None,
    end_date: date | None = None,
    ffill_limit: int = 5,
) -> FactorDecayResult:
    """Compute factor IC at multiple forward lags.

    Args:
        price_data: Dict mapping ticker to PriceData.
        factor_fn: Factor name from the registry.
        max_lag: Maximum forward lag in rebalance periods (>= 1).
        rebalance_freq: ``"weekly"``, ``"monthly"``, or ``"quarterly"``.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        ffill_limit: Max trading days to look back for close prices.

    Returns:
        FactorDecayResult with IC statistics per lag.

    Raises:
        ValueError: If factor_fn is unsupported or max_lag < 1.
    """
    if factor_fn not in PRICE_FACTOR_FNS:
        supported = list(PRICE_FACTOR_FNS.keys())
        raise ValueError(f"Unsupported factor: {factor_fn}. Use one of {supported}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    compute_factor = PRICE_FACTOR_FNS[factor_fn]

    # Build ticker → {date → close} lookup and collect all trading dates
    ticker_close: dict[str, dict[date, float]] = {}
    all_dates: set[date] = set()
    for ticker, pd_obj in price_data.items():
        closes_by_date: dict[date, float] = {}
        for p in pd_obj._sorted_prices():
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

    rebal_dates = rebalance_dates(sorted_dates, rebalance_freq)
    if len(rebal_dates) < 2:
        raise ValueError("Need at least 2 rebalance periods for decay analysis")

    total_rebalances = len(rebal_dates) - 1

    # Collect per-lag IC values
    ic_by_lag: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}

    for i in range(total_rebalances):
        rebal_date = rebal_dates[i]

        # Filter prices up to rebalance date and compute factor values
        factor_values: list[tuple[str, float]] = []
        for ticker, pd_obj in price_data.items():
            filtered_prices = [
                p
                for p in pd_obj._sorted_prices()
                if (d := parse_date(p.get("date"))) is not None and d <= rebal_date
            ]
            if filtered_prices:
                filtered_pd = PriceData(ticker=ticker, prices=filtered_prices)
                val = compute_factor(filtered_pd)
                if val is not None:
                    factor_values.append((ticker, val))

        if len(factor_values) < 3:
            continue

        # Execution date for this rebalance
        exec_date = next_trading_day(rebal_date, sorted_dates)
        if exec_date is None:
            continue

        # For each lag, compute forward return and IC
        for lag in range(1, max_lag + 1):
            future_idx = i + lag
            if future_idx >= len(rebal_dates):
                break

            future_rebal = rebal_dates[future_idx]
            future_exec = next_trading_day(future_rebal, sorted_dates)
            if future_exec is None:
                continue

            fwd_factors: list[float] = []
            fwd_returns: list[float] = []
            for ticker, fval in factor_values:
                closes = ticker_close.get(ticker, {})
                sp = ffill_close(closes, exec_date, sorted_dates, ffill_limit)
                ep = ffill_close(closes, future_exec, sorted_dates, ffill_limit)
                if sp is not None and ep is not None and sp > 0:
                    fwd_factors.append(fval)
                    fwd_returns.append((ep - sp) / sp)

            ic = spearman_rank_corr(fwd_factors, fwd_returns)
            if ic is not None:
                ic_by_lag[lag].append(ic)

    # Aggregate per-lag statistics
    lags: list[DecayLag] = []
    for lag in range(1, max_lag + 1):
        values = ic_by_lag[lag]
        n_obs = len(values)
        if n_obs == 0:
            lags.append(DecayLag(lag=lag, mean_ic=None, std_ic=None, n_obs=0))
        else:
            mean_ic = sum(values) / n_obs
            std_ic = statistics.stdev(values) if n_obs >= 2 else 0.0
            lags.append(DecayLag(lag=lag, mean_ic=mean_ic, std_ic=std_ic, n_obs=n_obs))

    # Half-life: first lag where abs(IC) drops below 50% of lag-1 abs(IC)
    half_life: float | None = None
    lag1_ic = lags[0].mean_ic if lags else None
    if lag1_ic is not None and abs(lag1_ic) > 0:
        threshold = abs(lag1_ic) * 0.5
        for dl in lags[1:]:
            if dl.mean_ic is not None and abs(dl.mean_ic) < threshold:
                half_life = float(dl.lag)
                break

    return FactorDecayResult(
        factor=factor_fn,
        max_lag=max_lag,
        lags=lags,
        half_life_months=half_life,
    )
