"""Output formatters for analysis results."""

from __future__ import annotations

import json
from typing import Any


def format_table(result: dict[str, Any]) -> str:
    """Format analysis result as a human-readable table."""
    lines: list[str] = []
    ticker = result["ticker"]
    edinet = result.get("edinet_code") or "N/A"
    as_of = result["as_of"]
    period = result.get("period_end") or "N/A"
    ds = result.get("data_sources", {})

    lines.append(f"{'=' * 60}")
    lines.append(f"  {ticker}  (EDINET: {edinet})")
    lines.append(f"  as_of: {as_of}  period_end: {period}")

    sources = []
    if ds.get("financials"):
        sources.append("EDINET")
    if ds.get("prices"):
        sources.append(f"yfinance ({ds.get('price_points', 0)}d)")
    if ds.get("market_cap"):
        mcap = ds["market_cap"]
        if mcap >= 1e12:
            sources.append(f"market_cap: {mcap / 1e12:.1f}T JPY")
        else:
            sources.append(f"market_cap: {mcap / 1e9:.0f}B JPY")
    lines.append(f"  sources: {', '.join(sources) if sources else 'none'}")
    lines.append(f"{'=' * 60}")

    # Group observations by category
    obs_list = result.get("observations", [])
    if not obs_list:
        lines.append("  No factors computed (insufficient data)")
        return "\n".join(lines)

    categories: dict[str, list[dict[str, Any]]] = {}
    for obs in obs_list:
        cat = obs["category"]
        categories.setdefault(cat, []).append(obs)

    for cat, observations in categories.items():
        lines.append(f"\n  [{cat.upper()}]")
        for obs in observations:
            val = obs["value"]
            if val is None:
                val_str = "N/A"
            elif abs(val) >= 100:
                val_str = f"{val:>12,.1f}"
            elif abs(val) >= 1:
                val_str = f"{val:>12.2f}"
            else:
                val_str = f"{val:>12.4f}"

            stale = ""
            if obs.get("staleness_days") is not None:
                stale = f"  ({obs['staleness_days']}d stale)"

            lines.append(f"    {obs['factor_id']:22s} {val_str}{stale}")

    lines.append("")
    return "\n".join(lines)


def format_backtest_table(result: Any) -> str:
    """Format BacktestResult as a human-readable table.

    Args:
        result: BacktestResult instance (from jpfin.models).

    Returns:
        Formatted string with performance and optional benchmark metrics.
    """
    perf = result.performance
    lines = [
        f"\n  {'=' * 50}",
        f"  Backtest: Top {result.top_n} by {result.factor}",
        f"  Period: {result.period}",
        f"  Months: {result.months}",
        f"  {'=' * 50}",
        f"  Total Return:    {perf.total_return:>8.1%}",
        f"  CAGR:            {perf.cagr:>8.1%}",
        f"  Annualized Vol:  {perf.annualized_vol:>8.1%}",
        f"  Sharpe Ratio:    {perf.sharpe_ratio:>8.2f}",
        f"  Max Drawdown:    {perf.max_drawdown:>8.1%}",
    ]
    if result.benchmark:
        bm = result.benchmark
        lines.extend(
            [
                f"  {'-' * 50}",
                f"  Benchmark:       {bm.benchmark_name}",
                f"  Benchmark Ret:   {bm.benchmark_return:>8.1%}",
                f"  Excess Return:   {bm.excess_return:>8.1%}",
                f"  Tracking Error:  {bm.tracking_error:>8.1%}",
                f"  Info Ratio:      {bm.information_ratio:>8.2f}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def format_rolling_table(analysis: Any) -> str:
    """Format RollingAnalysis as a human-readable table.

    Args:
        analysis: RollingAnalysis instance (from jpfin.models).

    Returns:
        Formatted string with per-window metrics and summary statistics.
    """
    n_windows = len(analysis.windows)
    has_ic = any(w.mean_ic is not None for w in analysis.windows)

    lines = [
        f"\n  {'=' * 72}",
        f"  Rolling Analysis: {analysis.factor}",
        f"  Window: {analysis.window_months}M  |  Total: {analysis.total_months}M"
        f"  |  Windows: {n_windows}",
        f"  {'=' * 72}",
    ]

    col_header = (
        f"  {'Start':>10s}  {'End':>10s}  {'Return':>8s}"
        f"  {'CAGR':>6s}  {'Vol':>6s}  {'Sharpe':>6s}  {'MaxDD':>6s}"
    )
    sep = f"  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}"
    if has_ic:
        col_header += f"  {'IC':>6s}"
        sep += f"  {'-' * 6}"
    lines.append(col_header)
    lines.append(sep)

    for w in analysis.windows:
        p = w.performance
        row = (
            f"  {w.window_start:>10s}  {w.window_end:>10s}"
            f"  {p.total_return:>8.1%}"
            f"  {p.cagr:>6.1%}"
            f"  {p.annualized_vol:>6.1%}"
            f"  {p.sharpe_ratio:>6.2f}"
            f"  {p.max_drawdown:>6.1%}"
        )
        if has_ic:
            ic_str = f"{w.mean_ic:>6.3f}" if w.mean_ic is not None else f"{'N/A':>6s}"
            row += f"  {ic_str}"
        lines.append(row)

    lines.append(f"  {'-' * 72}")
    lines.append(
        f"  Sharpe:  mean={analysis.sharpe_mean:.2f}"
        f"  std={analysis.sharpe_std:.2f}"
        f"  [{analysis.sharpe_min:.2f}, {analysis.sharpe_max:.2f}]"
    )
    if analysis.ic_mean is not None:
        ic_std_str = f"{analysis.ic_std:.3f}" if analysis.ic_std is not None else "N/A"
        lines.append(f"  IC:      mean={analysis.ic_mean:.3f}  std={ic_std_str}")
    lines.append("")
    return "\n".join(lines)


def format_decay_table(result: Any) -> str:
    """Format FactorDecayResult as a human-readable table.

    Args:
        result: FactorDecayResult instance (from jpfin.models).

    Returns:
        Formatted string with IC term structure and half-life.
    """
    lines = [
        f"\n  {'=' * 50}",
        f"  Factor Decay: {result.factor}",
        f"  Max Lag: {result.max_lag}",
        f"  {'=' * 50}",
        f"  {'Lag':>4s}  {'Mean IC':>8s}  {'Std IC':>8s}  {'N obs':>6s}",
        f"  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 6}",
    ]
    for dl in result.lags:
        ic_str = f"{dl.mean_ic:>8.4f}" if dl.mean_ic is not None else f"{'N/A':>8s}"
        std_str = f"{dl.std_ic:>8.4f}" if dl.std_ic is not None else f"{'N/A':>8s}"
        lines.append(f"  {dl.lag:>4d}  {ic_str}  {std_str}  {dl.n_obs:>6d}")
    lines.append(f"  {'-' * 50}")
    if result.half_life_months is not None:
        lines.append(f"  Half-life: {result.half_life_months:.0f} months")
    else:
        lines.append("  Half-life: N/A (no decay detected)")
    lines.append("")
    return "\n".join(lines)


def format_json(results: list[dict[str, Any]]) -> str:
    """Format results as JSON."""
    if len(results) == 1:
        return json.dumps(results[0], indent=2, ensure_ascii=False)
    return json.dumps(results, indent=2, ensure_ascii=False)
