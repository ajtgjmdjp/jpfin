"""Rolling window analysis for backtest results.

Slices the monthly return series from a completed backtest into
overlapping (or non-overlapping) windows and recomputes performance
metrics for each window.  This is a pure post-processing step —
the backtest loop is never re-run.
"""

from __future__ import annotations

import statistics

from jpfin.backtest import _compute_performance
from jpfin.models import BacktestResult, RollingAnalysis, RollingWindow


def compute_rolling(
    result: BacktestResult,
    window_months: int = 12,
    step: int = 1,
) -> RollingAnalysis:
    """Compute rolling window performance analysis.

    Args:
        result: Completed BacktestResult from run_backtest().
        window_months: Length of each rolling window in months (>= 2).
        step: Step size in months between window starts (>= 1).

    Returns:
        RollingAnalysis with per-window metrics and summary statistics.

    Raises:
        ValueError: If window_months < 2, step < 1, or
            window_months > total months.
    """
    total = len(result.monthly_returns)

    if window_months < 2:
        raise ValueError(f"window_months must be >= 2, got {window_months}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if window_months > total:
        raise ValueError(f"window_months ({window_months}) exceeds total months ({total})")

    ic_series = result.factor_metrics.ic_series if result.factor_metrics is not None else []

    windows: list[RollingWindow] = []
    start = 0
    while start + window_months <= total:
        end = start + window_months
        window_returns = result.monthly_returns[start:end]
        returns = [m.monthly_return for m in window_returns]

        perf = _compute_performance(returns)

        # IC slice — best-effort alignment with monthly_returns indices.
        # ic_series may be shorter than monthly_returns when some periods
        # produced too few cross-sectional pairs for a valid rank correlation.
        ic_slice = ic_series[start:end]
        mean_ic: float | None = sum(ic_slice) / len(ic_slice) if ic_slice else None

        windows.append(
            RollingWindow(
                window_start=window_returns[0].period_start,
                window_end=window_returns[-1].period_end,
                performance=perf,
                mean_ic=mean_ic,
            )
        )
        start += step

    sharpe_values = [w.performance.sharpe_ratio for w in windows]
    sharpe_mean = sum(sharpe_values) / len(sharpe_values)
    sharpe_std = statistics.stdev(sharpe_values) if len(sharpe_values) >= 2 else 0.0
    sharpe_min = min(sharpe_values)
    sharpe_max = max(sharpe_values)

    ic_values = [w.mean_ic for w in windows if w.mean_ic is not None]
    if not ic_values:
        ic_mean: float | None = None
        ic_std: float | None = None
    elif len(ic_values) == 1:
        ic_mean = ic_values[0]
        ic_std = 0.0
    else:
        ic_mean = sum(ic_values) / len(ic_values)
        ic_std = statistics.stdev(ic_values)

    return RollingAnalysis(
        factor=result.factor,
        window_months=window_months,
        total_months=total,
        windows=windows,
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        sharpe_min=sharpe_min,
        sharpe_max=sharpe_max,
        ic_mean=ic_mean,
        ic_std=ic_std,
    )
