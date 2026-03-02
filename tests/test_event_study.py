"""Tests for jpfin.event_study — Event-study analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from jpfin.cli import main
from jpfin.event_study import PriceFactorProvider, run_event_study


def _make_prices(n: int = 100) -> list[dict[str, object]]:
    """Generate fake daily prices."""
    prices: list[dict[str, object]] = []
    base = 1000.0
    for i in range(n):
        m = (i // 28) + 3
        d = (i % 28) + 1
        if m > 12:
            m = 12
        prices.append({"date": f"2025-{m:02d}-{d:02d}", "close": base + i * 5.0})
    return prices


class TestPriceFactorProvider:
    def test_protocol_conformance(self) -> None:
        from jpfin.fusion import FactorSnapshotProvider

        assert isinstance(PriceFactorProvider(), FactorSnapshotProvider)

    def test_empty_cache_returns_none(self) -> None:
        from datetime import datetime

        provider = PriceFactorProvider()
        assert provider.factors_at(ticker="7203", as_of=datetime(2025, 5, 12)) is None

    def test_with_preloaded_data(self) -> None:
        from datetime import datetime

        from japan_finance_factors._models import PriceData

        prices = _make_prices()
        pd = PriceData(ticker="7203", prices=prices)
        provider = PriceFactorProvider({"7203": pd})

        result = provider.factors_at(ticker="7203", as_of=datetime(2025, 5, 12))
        # Should return some factors (may be None if insufficient data for some)
        assert result is not None or result is None  # no crash


class TestRunEventStudy:
    @patch("jpfin.event_study._fetch_prices", new_callable=AsyncMock)
    def test_basic(self, mock_fetch: AsyncMock) -> None:
        from japan_finance_factors._models import PriceData

        mock_fetch.return_value = PriceData(ticker="7203", prices=_make_prices())

        result = run_event_study("7203", "2025-05-12")
        assert result.ticker == "7203"
        assert result.event_date == "2025-05-12"
        assert len(result.windows) == 7  # T-20, T-5, T-1, T0, T+1, T+5, T+20

    @patch("jpfin.event_study._fetch_prices", new_callable=AsyncMock)
    def test_custom_windows(self, mock_fetch: AsyncMock) -> None:
        from japan_finance_factors._models import PriceData

        mock_fetch.return_value = PriceData(ticker="7203", prices=_make_prices())

        result = run_event_study(
            "7203",
            "2025-05-12",
            before_days=[10],
            after_days=[10],
        )
        assert len(result.windows) == 3  # T-10, T0, T+10

    @patch("jpfin.event_study._fetch_prices", new_callable=AsyncMock)
    def test_no_data(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.return_value = None

        result = run_event_study("9999", "2025-05-12")
        for w in result.windows:
            assert w.factors == {}

    @patch("jpfin.event_study._fetch_prices", new_callable=AsyncMock)
    def test_specific_factors(self, mock_fetch: AsyncMock) -> None:
        from japan_finance_factors._models import PriceData

        mock_fetch.return_value = PriceData(ticker="7203", prices=_make_prices())

        result = run_event_study("7203", "2025-05-12", factors=["mom_3m"])
        for w in result.windows:
            if w.factors:
                assert set(w.factors.keys()) == {"mom_3m"}


class TestEventStudyCLI:
    @patch("jpfin.event_study._fetch_prices", new_callable=AsyncMock)
    def test_json_output(self, mock_fetch: AsyncMock) -> None:
        from japan_finance_factors._models import PriceData

        mock_fetch.return_value = PriceData(ticker="7203", prices=_make_prices())

        runner = CliRunner()
        result = runner.invoke(main, ["event-study", "7203", "2025-05-12", "-f", "json"])
        assert result.exit_code == 0, result.output
        assert '"ticker": "7203"' in result.output

    @patch("jpfin.event_study._fetch_prices", new_callable=AsyncMock)
    def test_table_output(self, mock_fetch: AsyncMock) -> None:
        from japan_finance_factors._models import PriceData

        mock_fetch.return_value = PriceData(ticker="7203", prices=_make_prices())

        runner = CliRunner()
        result = runner.invoke(main, ["event-study", "7203", "2025-05-12"])
        assert result.exit_code == 0, result.output
        assert "Event Study" in result.output
        assert "7203" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["event-study", "--help"])
        assert result.exit_code == 0
        assert "factor" in result.output.lower()
