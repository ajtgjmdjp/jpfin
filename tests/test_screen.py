"""Tests for screen module."""

from __future__ import annotations

from unittest.mock import patch

from jpfin.screen import screen_tickers


def _mock_analysis(ticker: str, roe: float | None = 0.1) -> dict:
    return {
        "ticker": ticker,
        "factors": {"roe": roe, "ev_ebitda": 10.0},
        "data_sources": {"financials": True, "prices": True},
    }


class TestScreenTickers:
    @patch("jpfin.screen.analyze_ticker_sync")
    def test_basic_ranking(self, mock_analyze) -> None:
        mock_analyze.side_effect = [
            _mock_analysis("A", roe=0.15),
            _mock_analysis("B", roe=0.08),
            _mock_analysis("C", roe=0.12),
        ]
        results = screen_tickers(["A", "B", "C"], "roe")
        assert results[0]["ticker"] == "A"  # Highest ROE
        assert results[1]["ticker"] == "C"
        assert results[2]["ticker"] == "B"  # Lowest ROE
        assert results[0]["rank"] == 1

    @patch("jpfin.screen.analyze_ticker_sync")
    def test_ascending(self, mock_analyze) -> None:
        mock_analyze.side_effect = [
            _mock_analysis("A", roe=0.15),
            _mock_analysis("B", roe=0.08),
        ]
        results = screen_tickers(["A", "B"], "roe", ascending=True)
        assert results[0]["ticker"] == "B"  # Lowest first

    @patch("jpfin.screen.analyze_ticker_sync")
    def test_none_values_at_end(self, mock_analyze) -> None:
        mock_analyze.side_effect = [
            _mock_analysis("A", roe=0.10),
            _mock_analysis("B", roe=None),
        ]
        results = screen_tickers(["A", "B"], "roe")
        assert results[0]["ticker"] == "A"
        assert results[1]["rank"] is None

    def test_empty_tickers_returns_empty(self) -> None:
        results = screen_tickers([], "roe")
        assert results == []

    @patch("jpfin.screen.analyze_ticker_sync")
    def test_all_analysis_failures(self, mock_analyze) -> None:
        mock_analyze.side_effect = RuntimeError("API down")
        results = screen_tickers(["A", "B"], "roe")
        assert len(results) == 2
        assert all(r["factor_value"] is None for r in results)
        assert all(r["rank"] is None for r in results)
