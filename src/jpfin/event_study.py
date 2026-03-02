"""Event-study analysis: factor snapshots around a corporate event.

Provides a ``FactorSnapshotProvider`` implementation backed by
``japan-finance-factors`` price data, and a sync helper to run the
event-factor context pipeline from the CLI.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from japan_finance_events import Direction, Event, EventStore, EventType
from japan_finance_factors._models import PriceData
from japan_finance_factors.fetch import fetch_price_data

from jpfin._utils import parse_date
from jpfin.factor_registry import compute_price_factors
from jpfin.fusion import EventFactorFusion
from jpfin.models import EventStudyResult, EventStudyWindow


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
            p for p in pd.prices if (d := parse_date(p.get("date"))) is not None and d <= cutoff
        ]
        if not filtered_prices:
            return None

        filtered = PriceData(ticker=pd.ticker, prices=filtered_prices)
        result = compute_price_factors(filtered, factors=factors)
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
) -> EventStudyResult:
    """Run an event-study analysis for a ticker at a given event date.

    Args:
        ticker: Stock code (e.g., "7203").
        event_date: ISO date string of the event (e.g., "2025-05-12").
        before_days: Days before event to sample.
        after_days: Days after event to sample.
        factors: Specific factors to compute.

    Returns:
        Typed EventStudyResult with factor context at each time window.
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

    windows: list[EventStudyWindow] = []
    for label, obs in ctx.items():
        if obs is not None:
            windows.append(
                EventStudyWindow(
                    window=label,
                    as_of=obs.as_of.isoformat(),
                    factors=obs.factors,
                )
            )
        else:
            windows.append(EventStudyWindow(window=label, as_of=None, factors={}))

    return EventStudyResult(ticker=ticker, event_date=event_date, windows=windows)
