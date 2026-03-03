"""Factor cross-sectional correlation analysis.

Computes the average cross-sectional rank correlation between
all pairs of price-based factors across rebalance dates.
Used for multi-factor portfolio construction — identifying
redundant factors and diversification opportunities.
"""

from __future__ import annotations

import statistics
from datetime import date

from japan_finance_factors._models import PriceData

from jpfin._utils import parse_date
from jpfin.backtest import month_end_dates
from jpfin.factor_registry import PRICE_FACTOR_FNS, compute_price_factors
from jpfin.metrics import spearman_rank_corr
from jpfin.models import FactorCorrelationResult


def compute_factor_correlation(
    price_data: dict[str, PriceData],
    factors: list[str] | None = None,
    *,
    method: str = "spearman",
    start_date: date | None = None,
    end_date: date | None = None,
    min_cross_section: int = 5,
) -> FactorCorrelationResult:
    """Compute average cross-sectional correlation between factor pairs.

    At each rebalance date, computes factor values for all tickers,
    then calculates pairwise rank correlations across the cross-section.
    Returns the time-averaged correlation matrix.

    Args:
        price_data: Dict mapping ticker → PriceData.
        factors: Factor names to include. ``None`` for all available.
        method: Correlation method (``"spearman"`` only for now).
        start_date: Start date filter.
        end_date: End date filter.
        min_cross_section: Minimum tickers with valid values for a
            correlation to be computed at a given date.

    Returns:
        FactorCorrelationResult with correlation matrix and metadata.

    Raises:
        ValueError: If method is unsupported, factors are unknown, or
            insufficient data.
    """
    if method != "spearman":
        raise ValueError(f"Unsupported method: {method}. Use 'spearman'.")

    factor_names = factors or list(PRICE_FACTOR_FNS.keys())
    for f in factor_names:
        if f not in PRICE_FACTOR_FNS:
            supported = list(PRICE_FACTOR_FNS.keys())
            raise ValueError(f"Unknown factor: {f}. Available: {supported}")

    n_factors = len(factor_names)
    if n_factors < 2:
        raise ValueError("Need at least 2 factors for correlation analysis")

    # Build date universe from all tickers
    all_dates: set[date] = set()
    for pd in price_data.values():
        for p in pd._sorted_prices():
            d = parse_date(p.get("date"))
            if d is not None:
                all_dates.add(d)

    sorted_dates = sorted(all_dates)
    if start_date:
        sorted_dates = [d for d in sorted_dates if d >= start_date]
    if end_date:
        sorted_dates = [d for d in sorted_dates if d <= end_date]

    rebal_dates = month_end_dates(sorted_dates)
    if len(rebal_dates) < 1:
        raise ValueError("Insufficient data for correlation analysis")

    # Accumulate pairwise correlations across dates
    # corr_accum[i][j] = list of correlations across dates
    corr_accum: list[list[list[float]]] = [
        [[] for _ in range(n_factors)] for _ in range(n_factors)
    ]

    n_dates_used = 0

    for rebal_date in rebal_dates:
        # Compute factor values for all tickers at this date
        factor_vals: dict[str, dict[str, float | None]] = {}
        for ticker, pd in price_data.items():
            filtered_prices = [
                p
                for p in pd._sorted_prices()
                if (d := parse_date(p.get("date"))) is not None and d <= rebal_date
            ]
            if not filtered_prices:
                continue
            filtered_pd = PriceData(ticker=ticker, prices=filtered_prices)
            factor_vals[ticker] = compute_price_factors(filtered_pd, factor_names)

        if len(factor_vals) < min_cross_section:
            continue

        n_dates_used += 1

        # For each pair (i, j), compute cross-sectional rank correlation
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                fi_name = factor_names[i]
                fj_name = factor_names[j]

                # Collect paired values (both non-None)
                xs: list[float] = []
                ys: list[float] = []
                for ticker_vals in factor_vals.values():
                    vi = ticker_vals.get(fi_name)
                    vj = ticker_vals.get(fj_name)
                    if vi is not None and vj is not None:
                        xs.append(vi)
                        ys.append(vj)

                if len(xs) < min_cross_section:
                    continue

                rho = spearman_rank_corr(xs, ys)
                if rho is not None:
                    corr_accum[i][j].append(rho)
                    corr_accum[j][i].append(rho)

    if n_dates_used == 0:
        raise ValueError("No rebalance dates with sufficient data")

    # Build result matrices
    corr_matrix: list[list[float | None]] = []
    n_obs_matrix: list[list[int]] = []

    for i in range(n_factors):
        row_corr: list[float | None] = []
        row_obs: list[int] = []
        for j in range(n_factors):
            if i == j:
                row_corr.append(1.0)
                row_obs.append(n_dates_used)
            else:
                obs = corr_accum[i][j]
                row_obs.append(len(obs))
                if obs:
                    row_corr.append(statistics.mean(obs))
                else:
                    row_corr.append(None)
        corr_matrix.append(row_corr)
        n_obs_matrix.append(row_obs)

    # Mean absolute correlation per factor (excluding diagonal)
    mean_abs_corr: list[float] = []
    for i in range(n_factors):
        abs_vals: list[float] = []
        for j in range(n_factors):
            if i != j:
                v = corr_matrix[i][j]
                if v is not None:
                    abs_vals.append(abs(v))
        mean_abs_corr.append(statistics.mean(abs_vals) if abs_vals else 0.0)

    return FactorCorrelationResult(
        factors=factor_names,
        correlation_matrix=corr_matrix,
        n_obs_matrix=n_obs_matrix,
        mean_abs_correlation=mean_abs_corr,
        n_dates=n_dates_used,
        method=method,
    )
