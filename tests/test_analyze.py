"""Tests for analyze module."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from japan_finance_factors._models import PriceData

from jpfin.analyze import _resolve_edinet_code, analyze_ticker_sync


class TestResolveEdinetCode:
    @patch("jpfin.analyze.resolve", create=True)
    def test_resolve_success(self, mock_resolve: MagicMock) -> None:
        # Patch japan_finance_codes.resolve at the point it's imported
        mock_result = type("R", (), {"edinet_code": "E02144"})()
        mock_resolve.return_value = mock_result
        # Since _resolve_edinet_code does a local import, we mock the module
        with patch.dict("sys.modules", {"japan_finance_codes": MagicMock(resolve=mock_resolve)}):
            result = _resolve_edinet_code("7203")
            assert result == "E02144"

    def test_resolve_returns_none_on_failure(self) -> None:
        # When japan-finance-codes is not installed or returns None
        with patch.dict("sys.modules", {"japan_finance_codes": MagicMock(resolve=MagicMock(return_value=None))}):
            assert _resolve_edinet_code("XXXX") is None


class TestAnalyzeTickerSync:
    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    @patch("jpfin.analyze._resolve_edinet_code")
    def test_price_only(self, mock_resolve: MagicMock, mock_fin: AsyncMock, mock_price: AsyncMock) -> None:
        mock_resolve.return_value = None
        mock_fin.return_value = None
        mock_price.return_value = PriceData(
            ticker="7203",
            prices=[
                {"date": (date(2024, 1, 1) + timedelta(days=i)).isoformat(), "close": 2000 + i}
                for i in range(300)
            ],
        )

        result = analyze_ticker_sync("7203", as_of=datetime(2025, 7, 1))
        assert result["ticker"] == "7203"
        assert result["data_sources"]["prices"] is True
        assert result["data_sources"]["financials"] is False
        assert len(result["observations"]) > 0

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    @patch("jpfin.analyze._resolve_edinet_code")
    def test_no_data(self, mock_resolve: MagicMock, mock_fin: AsyncMock, mock_price: AsyncMock) -> None:
        mock_resolve.return_value = None
        mock_fin.return_value = None
        mock_price.return_value = None

        result = analyze_ticker_sync("9999", as_of=datetime(2025, 7, 1))
        assert result["ticker"] == "9999"
        assert len(result["observations"]) == 0
