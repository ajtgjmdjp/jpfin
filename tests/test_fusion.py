"""Tests for jpfin.fusion — Event-Factor Fusion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from japan_finance_events import Direction, Event, EventStore, EventType
from pydantic import ValidationError

from jpfin.fusion import (
    CompanyResolver,
    EventFactorFusion,
    FactorSnapshotProvider,
)


def _make_event(
    *,
    ticker: str = "7203",
    edinet_code: str | None = "E02144",
    event_type: EventType = EventType.EARNINGS,
    direction: Direction = Direction.UP,
    pit: str = "2025-05-12T15:30:00+09:00",
    title: str = "FY2024 決算発表",
) -> Event:
    return Event(
        event_type=event_type,
        direction=direction,
        company_ticker=ticker,
        edinet_code=edinet_code,
        company_name="トヨタ自動車",
        pit_published_at=datetime.fromisoformat(pit),
        title=title,
    )


class StubProvider:
    """Stub that returns fixed factor values."""

    def __init__(self, data: dict[str, dict[str, float | None]] | None = None) -> None:
        self._data = data or {}

    def factors_at(
        self,
        *,
        ticker: str,
        as_of: datetime,
        factors: list[str] | None = None,
    ) -> dict[str, float | None] | None:
        result = self._data.get(ticker)
        if result is None:
            return None
        if factors:
            return {k: result.get(k) for k in factors}
        return dict(result)


class StubResolver:
    """Stub that resolves EDINET code to ticker."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = mapping or {}

    def resolve_ticker(self, identifier: str) -> str | None:
        return self._mapping.get(identifier)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_stub_provider_is_protocol_compliant() -> None:
    assert isinstance(StubProvider(), FactorSnapshotProvider)


def test_stub_resolver_is_protocol_compliant() -> None:
    assert isinstance(StubResolver(), CompanyResolver)


# ---------------------------------------------------------------------------
# factors_at_event
# ---------------------------------------------------------------------------


class TestFactorsAtEvent:
    def test_basic(self) -> None:
        event = _make_event()
        provider = StubProvider({"7203": {"roe": 0.12, "mom_3m": 0.05}})
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
        )
        obs = fusion.factors_at_event(event)
        assert obs is not None
        assert obs.event_id == event.event_id
        assert obs.company_ticker == "7203"
        assert obs.offset_days == 0
        assert obs.factors == {"roe": 0.12, "mom_3m": 0.05}

    def test_with_offset(self) -> None:
        event = _make_event(pit="2025-05-12T15:30:00+09:00")
        provider = StubProvider({"7203": {"roe": 0.10}})
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
        )
        obs = fusion.factors_at_event(event, offset_days=5)
        assert obs is not None
        assert obs.offset_days == 5
        expected = event.pit_published_at + timedelta(days=5)
        assert obs.as_of == expected

    def test_no_ticker_returns_none(self) -> None:
        event = _make_event(ticker=None, edinet_code=None)
        provider = StubProvider({"7203": {"roe": 0.10}})
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
        )
        assert fusion.factors_at_event(event) is None

    def test_ticker_from_resolver(self) -> None:
        event = _make_event(ticker=None, edinet_code="E02144")
        provider = StubProvider({"7203": {"roe": 0.12}})
        resolver = StubResolver({"E02144": "7203"})
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
            resolver=resolver,
        )
        obs = fusion.factors_at_event(event)
        assert obs is not None
        assert obs.company_ticker == "7203"

    def test_no_factor_data_returns_none(self) -> None:
        event = _make_event()
        provider = StubProvider({})  # No data
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
        )
        assert fusion.factors_at_event(event) is None

    def test_specific_factors(self) -> None:
        event = _make_event()
        provider = StubProvider({"7203": {"roe": 0.12, "mom_3m": 0.05, "ev_ebitda": 8.5}})
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
        )
        obs = fusion.factors_at_event(event, factors=["roe", "mom_3m"])
        assert obs is not None
        assert set(obs.factors.keys()) == {"roe", "mom_3m"}

    def test_observation_is_frozen(self) -> None:
        event = _make_event()
        provider = StubProvider({"7203": {"roe": 0.12}})
        fusion = EventFactorFusion(
            events=EventStore(),
            snapshot_provider=provider,
        )
        obs = fusion.factors_at_event(event)
        assert obs is not None
        with pytest.raises(ValidationError):
            obs.offset_days = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# aligned_observations
# ---------------------------------------------------------------------------


class TestAlignedObservations:
    def _build_store(self) -> EventStore:
        store = EventStore()
        store.upsert_many(
            [
                _make_event(
                    ticker="7203",
                    pit="2025-05-12T15:30:00+09:00",
                    title="Toyota FY2024",
                ),
                _make_event(
                    ticker="6758",
                    edinet_code="E01777",
                    pit="2025-05-13T16:00:00+09:00",
                    title="Sony FY2024",
                ),
                _make_event(
                    ticker="9984",
                    edinet_code="E05739",
                    event_type=EventType.FORECAST_REVISION,
                    pit="2025-05-14T17:00:00+09:00",
                    title="SoftBank 業績修正",
                ),
            ]
        )
        return store

    def test_all_events(self) -> None:
        store = self._build_store()
        provider = StubProvider(
            {
                "7203": {"roe": 0.12},
                "6758": {"roe": 0.08},
                "9984": {"roe": 0.05},
            }
        )
        fusion = EventFactorFusion(events=store, snapshot_provider=provider)
        results = list(
            fusion.aligned_observations(
                start=datetime(2025, 5, 1, tzinfo=timezone.utc),
                end=datetime(2025, 5, 31, tzinfo=timezone.utc),
            )
        )
        assert len(results) == 3

    def test_filter_by_event_type(self) -> None:
        store = self._build_store()
        provider = StubProvider(
            {
                "7203": {"roe": 0.12},
                "6758": {"roe": 0.08},
                "9984": {"roe": 0.05},
            }
        )
        fusion = EventFactorFusion(events=store, snapshot_provider=provider)
        results = list(
            fusion.aligned_observations(
                start=datetime(2025, 5, 1, tzinfo=timezone.utc),
                end=datetime(2025, 5, 31, tzinfo=timezone.utc),
                event_types={EventType.EARNINGS},
            )
        )
        assert len(results) == 2
        assert all(r.event_type == "earnings" for r in results)

    def test_filter_by_company(self) -> None:
        store = self._build_store()
        provider = StubProvider(
            {
                "7203": {"roe": 0.12},
                "6758": {"roe": 0.08},
                "9984": {"roe": 0.05},
            }
        )
        fusion = EventFactorFusion(events=store, snapshot_provider=provider)
        results = list(
            fusion.aligned_observations(
                start=datetime(2025, 5, 1, tzinfo=timezone.utc),
                end=datetime(2025, 5, 31, tzinfo=timezone.utc),
                companies=["7203"],
            )
        )
        assert len(results) == 1
        assert results[0].company_ticker == "7203"

    def test_skips_events_without_factor_data(self) -> None:
        store = self._build_store()
        provider = StubProvider({"7203": {"roe": 0.12}})  # Only Toyota
        fusion = EventFactorFusion(events=store, snapshot_provider=provider)
        results = list(
            fusion.aligned_observations(
                start=datetime(2025, 5, 1, tzinfo=timezone.utc),
                end=datetime(2025, 5, 31, tzinfo=timezone.utc),
            )
        )
        assert len(results) == 1

    def test_empty_window(self) -> None:
        store = self._build_store()
        provider = StubProvider({"7203": {"roe": 0.12}})
        fusion = EventFactorFusion(events=store, snapshot_provider=provider)
        results = list(
            fusion.aligned_observations(
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 31, tzinfo=timezone.utc),
            )
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# event_factor_context
# ---------------------------------------------------------------------------


class TestEventFactorContext:
    def test_default_windows(self) -> None:
        event = _make_event()
        provider = StubProvider({"7203": {"mom_3m": 0.05}})
        fusion = EventFactorFusion(events=EventStore(), snapshot_provider=provider)
        ctx = fusion.event_factor_context(event, factors=["mom_3m"])
        assert "T-20" in ctx
        assert "T-5" in ctx
        assert "T-1" in ctx
        assert "T0" in ctx
        assert "T+1" in ctx
        assert "T+5" in ctx
        assert "T+20" in ctx
        assert len(ctx) == 7

    def test_custom_windows(self) -> None:
        event = _make_event()
        provider = StubProvider({"7203": {"roe": 0.12}})
        fusion = EventFactorFusion(events=EventStore(), snapshot_provider=provider)
        ctx = fusion.event_factor_context(
            event,
            before_days=[30],
            after_days=[60, 90],
        )
        assert set(ctx.keys()) == {"T-30", "T0", "T+60", "T+90"}

    def test_context_preserves_offsets(self) -> None:
        event = _make_event()
        provider = StubProvider({"7203": {"roe": 0.12}})
        fusion = EventFactorFusion(events=EventStore(), snapshot_provider=provider)
        ctx = fusion.event_factor_context(event, before_days=[5], after_days=[5])
        assert ctx["T-5"] is not None
        assert ctx["T-5"].offset_days == -5
        assert ctx["T+5"] is not None
        assert ctx["T+5"].offset_days == 5
        assert ctx["T0"] is not None
        assert ctx["T0"].offset_days == 0
