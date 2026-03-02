"""Event-Factor Fusion: PIT-safe join of corporate events and factor values.

Provides ``EventFactorFusion`` for combining events from
``japan-finance-events`` with factor snapshots from
``japan-finance-factors``, aligned at precise point-in-time timestamps.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from japan_finance_events import Event, EventStore, EventType


@runtime_checkable
class FactorSnapshotProvider(Protocol):
    """Protocol for providing factor snapshots at a given point in time.

    Callers inject an implementation that knows how to compute or look up
    factor values for a ticker at a specific ``as_of`` timestamp.
    """

    def factors_at(
        self,
        *,
        ticker: str,
        as_of: datetime,
        factors: list[str] | None = None,
    ) -> dict[str, float | None] | None:
        """Return factor values for *ticker* at *as_of*, or ``None`` if unavailable."""
        ...


@runtime_checkable
class CompanyResolver(Protocol):
    """Protocol for resolving company identifiers to ticker."""

    def resolve_ticker(self, identifier: str) -> str | None:
        """Resolve any identifier (EDINET code, ticker, etc.) to a 4-digit ticker."""
        ...


class EventFactorObservation(BaseModel):
    """A single event paired with factor values at a point in time."""

    event_id: str
    event_type: str
    direction: str
    company_ticker: str
    company_name: str
    as_of: datetime
    offset_days: int
    factors: dict[str, float | None]

    model_config = {"frozen": True}


class EventFactorFusion:
    """Join events and factor snapshots with PIT safety.

    Args:
        events: An ``EventStore`` instance for querying events.
        snapshot_provider: Provides factor values at a given point in time.
        resolver: Resolves company identifiers to tickers.
    """

    def __init__(
        self,
        *,
        events: EventStore,
        snapshot_provider: FactorSnapshotProvider,
        resolver: CompanyResolver | None = None,
    ) -> None:
        self._events = events
        self._provider = snapshot_provider
        self._resolver = resolver

    def _resolve_ticker(self, event: Event) -> str | None:
        """Extract ticker from event, falling back to resolver."""
        if event.company_ticker:
            return event.company_ticker
        if self._resolver and event.edinet_code:
            return self._resolver.resolve_ticker(event.edinet_code)
        return None

    def factors_at_event(
        self,
        event: Event,
        *,
        offset_days: int = 0,
        factors: list[str] | None = None,
    ) -> EventFactorObservation | None:
        """Get factor values at an event's PIT timestamp (+ optional offset).

        Args:
            event: The corporate event.
            offset_days: Days to add to ``pit_published_at`` (negative for before).
            factors: Specific factors to compute. ``None`` for all available.

        Returns:
            ``EventFactorObservation`` or ``None`` if ticker is unresolvable
            or no factor data is available.
        """
        ticker = self._resolve_ticker(event)
        if not ticker:
            return None

        as_of = event.pit_published_at + timedelta(days=offset_days)
        result = self._provider.factors_at(ticker=ticker, as_of=as_of, factors=factors)
        if result is None:
            return None

        return EventFactorObservation(
            event_id=event.event_id,
            event_type=event.event_type.value,
            direction=event.direction.value,
            company_ticker=ticker,
            company_name=event.company_name,
            as_of=as_of,
            offset_days=offset_days,
            factors=result,
        )

    def aligned_observations(
        self,
        *,
        start: datetime,
        end: datetime,
        companies: list[str] | None = None,
        event_types: set[EventType] | None = None,
        factors: list[str] | None = None,
        offset_days: int = 0,
    ) -> Iterator[EventFactorObservation]:
        """Yield event-factor pairs in PIT order within a time window.

        Args:
            start: Window start (inclusive).
            end: Window end (inclusive).
            companies: Filter by ticker or EDINET code. ``None`` for all.
            event_types: Filter by event type. ``None`` for all.
            factors: Specific factors to compute.
            offset_days: Days offset from each event's PIT timestamp.

        Yields:
            ``EventFactorObservation`` for each event with available factor data.
        """
        for event in self._events.iter_pit(start=start, end=end, event_types=event_types):
            if companies:
                ids = {event.company_ticker, event.edinet_code}
                if not any(c in ids for c in companies):
                    continue
            obs = self.factors_at_event(event, offset_days=offset_days, factors=factors)
            if obs is not None:
                yield obs

    def event_factor_context(
        self,
        event: Event,
        *,
        before_days: list[int] | None = None,
        after_days: list[int] | None = None,
        factors: list[str] | None = None,
    ) -> dict[str, EventFactorObservation | None]:
        """Get factor snapshots at multiple points around an event.

        Useful for event-study analysis: compare factors before and after
        a corporate event to assess impact.

        Args:
            event: The corporate event to analyze.
            before_days: Days before event to sample (e.g., ``[1, 5, 20]``).
            after_days: Days after event to sample.
            factors: Specific factors to compute.

        Returns:
            Dict mapping labels (``"T-5"``, ``"T0"``, ``"T+20"``) to observations.
        """
        if before_days is None:
            before_days = [1, 5, 20]
        if after_days is None:
            after_days = [1, 5, 20]

        out: dict[str, EventFactorObservation | None] = {}
        for d in sorted(before_days, reverse=True):
            out[f"T-{d}"] = self.factors_at_event(event, offset_days=-d, factors=factors)
        out["T0"] = self.factors_at_event(event, offset_days=0, factors=factors)
        for d in sorted(after_days):
            out[f"T+{d}"] = self.factors_at_event(event, offset_days=d, factors=factors)
        return out
