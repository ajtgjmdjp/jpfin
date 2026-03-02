"""Event-study analysis: factor snapshots around a corporate event.

Provides a ``FactorSnapshotProvider`` implementation backed by
``japan-finance-factors`` price data, and a sync helper to run the
event-factor context pipeline from the CLI.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from japan_finance_events import Direction, Event, EventStore, EventType
from japan_finance_factors._models import PriceData
from japan_finance_factors.fetch import fetch_price_data

from jpfin._utils import parse_date
from jpfin.fusion import EventFactorFusion


class PriceFactorProvider:
    """FactorSnapshotProvider backed by pre-loaded yfinance price history.

    Price data must be loaded before use via ``preload()`` or by passing
    pre-fetched data to the constructor. The ``factors_at`` method is
    fully synchronous.
    """

    def __init__(self, cache: dict[str, PriceData | None] | None = None) -> None:
        self._cache: dict[str, PriceData | None] = dict(cache) if cache else {}

    def factors_at(
        self,
        *,
        ticker: str,
        as_of: datetime,
        factors: list[str] | None = None,
    ) -> dict[str, float | None] | None:
        pd = self._cache.get(ticker)
        if pd is None:
            return None

        cutoff = as_of.date() if isinstance(as_of, datetime) else as_of
        filtered_prices = [
            p for p in pd.prices if (d := parse_date(p.get("date"))) is None or d <= cutoff
        ]
        if not filtered_prices:
            return None

        filtered = PriceData(ticker=pd.ticker, prices=filtered_prices)

        from japan_finance_factors.factors import momentum, risk

        all_factors: dict[str, Any] = {
            "mom_3m": momentum.mom_3m,
            "mom_12m": momentum.mom_12m,
            "realized_vol_60d": risk.realized_vol_60d,
            "max_drawdown_252d": risk.max_drawdown_252d,
        }

        requested = factors or list(all_factors.keys())
        result: dict[str, float | None] = {}
        for name in requested:
            fn = all_factors.get(name)
            if fn is not None:
                result[name] = fn(filtered)
            else:
                result[name] = None
        return result if any(v is not None for v in result.values()) else None


async def _fetch_prices(ticker: str) -> PriceData | None:
    """Fetch price data for a single ticker."""
    try:
        return await fetch_price_data(ticker, lookback_days=500)
    except Exception:
        return None


def run_event_study(
    ticker: str,
    event_date: str,
    *,
    before_days: list[int] | None = None,
    after_days: list[int] | None = None,
    factors: list[str] | None = None,
) -> dict[str, Any]:
    """Run an event-study analysis for a ticker at a given event date.

    Args:
        ticker: Stock code (e.g., "7203").
        event_date: ISO date string of the event (e.g., "2025-05-12").
        before_days: Days before event to sample.
        after_days: Days after event to sample.
        factors: Specific factors to compute.

    Returns:
        Dict with event info and factor context at each time window.
    """
    # Fetch price data (async → sync bridge)
    price_data = asyncio.run(_fetch_prices(ticker))

    pit = datetime.fromisoformat(event_date)
    if pit.tzinfo is None:
        pit = pit.replace(hour=15, minute=30, tzinfo=timezone(timedelta(hours=9)))

    event = Event(
        event_type=EventType.OTHER,
        direction=Direction.UNKNOWN,
        company_ticker=ticker,
        company_name=ticker,
        pit_published_at=pit,
        title=f"Event study: {ticker} @ {event_date}",
    )

    provider = PriceFactorProvider({ticker: price_data})
    fusion = EventFactorFusion(
        events=EventStore(),
        snapshot_provider=provider,
    )

    ctx = fusion.event_factor_context(
        event,
        before_days=before_days,
        after_days=after_days,
        factors=factors,
    )

    windows: list[dict[str, Any]] = []
    for label, obs in ctx.items():
        entry: dict[str, Any] = {"window": label}
        if obs is not None:
            entry["as_of"] = obs.as_of.isoformat()
            entry["factors"] = obs.factors
        else:
            entry["as_of"] = None
            entry["factors"] = {}
        windows.append(entry)

    return {
        "ticker": ticker,
        "event_date": event_date,
        "windows": windows,
    }
