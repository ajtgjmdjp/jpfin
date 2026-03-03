"""Shared statistical functions for backtest analysis.

Extracted from backtest.py to enable reuse across rolling.py, decay.py,
and other post-processing modules without coupling to the backtest loop.
"""

from __future__ import annotations

import statistics

from jpfin.models import BenchmarkMetrics, PerformanceMetrics


def spearman_rank_corr(x: list[float], y: list[float]) -> float | None:
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


def compute_performance(
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


def compute_benchmark_metrics(
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
