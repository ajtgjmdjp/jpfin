"""Factor screening: rank tickers by a specified factor."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from jpfin.analyze import analyze_ticker_sync

logger = logging.getLogger(__name__)


def screen_tickers(
    tickers: list[str],
    factor: str,
    *,
    year: int | None = None,
    as_of: datetime | None = None,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """Compute factors for multiple tickers and rank by factor.

    Args:
        tickers: List of TSE stock codes.
        factor: Factor ID to rank by (e.g., "roe", "mom_3m").
        year: Fiscal year for EDINET data.
        as_of: Point-in-time cutoff.
        ascending: If True, lower is better (e.g., EV/EBITDA).

    Returns:
        Sorted list of dicts with ticker, factor_value, rank.
    """
    as_of = as_of or datetime.now()
    results: list[dict[str, Any]] = []

    for ticker in tickers:
        try:
            analysis = analyze_ticker_sync(
                ticker, year=year, as_of=as_of,
            )
            results.append({
                "ticker": ticker,
                "factor_value": analysis["factors"].get(factor),
                "factors": analysis["factors"],
                "data_sources": analysis["data_sources"],
            })
        except Exception:
            logger.warning("Failed to analyze %s", ticker, exc_info=True)
            results.append({
                "ticker": ticker,
                "factor_value": None,
                "factors": {},
                "data_sources": {},
                "error": "analysis failed",
            })

    # Sort: None values go to the end
    def sort_key(r: dict) -> tuple[int, float]:
        v = r["factor_value"]
        if v is None:
            return (1, 0.0)
        return (0, v if ascending else -v)

    results.sort(key=sort_key)

    # Add ranks (only for non-None values)
    rank = 1
    for r in results:
        if r["factor_value"] is not None:
            r["rank"] = rank
            rank += 1
        else:
            r["rank"] = None

    return results
