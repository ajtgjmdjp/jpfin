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
            sources.append(f"market_cap: {mcap/1e12:.1f}T JPY")
        else:
            sources.append(f"market_cap: {mcap/1e9:.0f}B JPY")
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


def format_json(results: list[dict[str, Any]]) -> str:
    """Format results as JSON."""
    if len(results) == 1:
        return json.dumps(results[0], indent=2, ensure_ascii=False)
    return json.dumps(results, indent=2, ensure_ascii=False)
