"""Cross-package integration smoke tests.

These tests verify the full pipeline from data models through factor
computation to CLI output, using fixture data (no network calls).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner
from japan_finance_factors._models import FinancialData, PriceData

from jpfin.cli import main


def _make_price_data(ticker: str = "7203", days: int = 300) -> PriceData:
    """Generate synthetic price data for testing."""
    base = date(2024, 1, 1)
    return PriceData(
        ticker=ticker,
        prices=[
            {"date": (base + timedelta(days=i)).isoformat(), "close": 2000.0 + i * 2}
            for i in range(days)
        ],
        market_cap=50e12,
    )


def _make_financial_data(edinet_code: str = "E02144") -> FinancialData:
    """Generate minimal financial data for testing."""
    return FinancialData(
        edinet_code=edinet_code,
        period_end=date(2025, 3, 31),
        revenue=37e12,
        operating_income=5.3e12,
        net_income=4.9e12,
        total_assets=90e12,
        total_equity=30e12,
        total_debt=20e12,
        cash_and_equivalents=8e12,
        ebitda=9e12,
        depreciation=3.5e12,
        capital_expenditure=2e12,
        free_cash_flow=5e12,
        market_cap=50e12,
    )


def _mock_registry(ticker: str, edinet_code: str | None):
    """Create a mock CompanyRegistry for japan-finance-codes."""
    mock_company = type("C", (), {"edinet_code": edinet_code})() if edinet_code else None
    mock_reg = MagicMock(by_ticker=MagicMock(return_value=mock_company))
    mock_cls = MagicMock()
    mock_cls.create = MagicMock(return_value=mock_reg)
    return MagicMock(CompanyRegistry=mock_cls)


class TestRealSnapshotResolution:
    """Test 0: real japan-finance-codes snapshot (no mocking)."""

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    def test_real_snapshot_resolves_toyota(self, mock_fin, mock_price) -> None:
        """Use actual CompanyRegistry.create() with bundled snapshot."""
        mock_fin.return_value = _make_financial_data()
        mock_price.return_value = _make_price_data()

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["edinet_code"] == "E02144"
        assert data["ticker"] == "7203"

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    def test_real_snapshot_unknown_ticker(self, mock_fin, mock_price) -> None:
        """Unknown ticker should return edinet_code=None gracefully."""
        mock_fin.return_value = None
        mock_price.return_value = None

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "0000", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["edinet_code"] is None


class TestCodeResolutionPipeline:
    """Test 1: ticker → EDINET code resolution (japan-finance-codes layer)."""

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    def test_ticker_resolves_to_edinet_code(self, mock_fin, mock_price) -> None:
        mock_fin.return_value = _make_financial_data()
        mock_price.return_value = _make_price_data()

        mock_mod = _mock_registry("7203", "E02144")
        with patch.dict("sys.modules", {"japan_finance_codes": mock_mod}):
            runner = CliRunner()
            result = runner.invoke(main, ["analyze", "7203", "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["edinet_code"] == "E02144"


class TestFinancialFactorPipeline:
    """Test 2: EDINET code → financial data → factor computation."""

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    @patch("jpfin.analyze._resolve_edinet_code", new_callable=AsyncMock)
    def test_financial_factors_computed(self, mock_resolve, mock_fin, mock_price) -> None:
        mock_resolve.return_value = "E02144"
        mock_fin.return_value = _make_financial_data()
        mock_price.return_value = None

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        factors = data["factors"]
        # Financial factors should be computed from EDINET data
        assert factors.get("ev_ebitda") is not None or factors.get("roe") is not None
        assert data["data_sources"]["financials"] is True


class TestPriceFactorPipeline:
    """Test 3: ticker → price data → momentum factor."""

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    @patch("jpfin.analyze._resolve_edinet_code", new_callable=AsyncMock)
    def test_price_factors_computed(self, mock_resolve, mock_fin, mock_price) -> None:
        mock_resolve.return_value = None
        mock_fin.return_value = None
        mock_price.return_value = _make_price_data()

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        factors = data["factors"]
        # Price-based factors should include momentum
        assert factors.get("mom_3m") is not None
        assert data["data_sources"]["prices"] is True
        assert data["data_sources"]["price_points"] > 0


class TestFullAnalyzePipeline:
    """Test 4: full jpfin analyze pipeline (codes + EDINET + stockprice + factors)."""

    @patch("jpfin.analyze._fetch_prices", new_callable=AsyncMock)
    @patch("jpfin.analyze._fetch_financials", new_callable=AsyncMock)
    def test_full_pipeline_json(self, mock_fin, mock_price) -> None:
        mock_fin.return_value = _make_financial_data()
        mock_price.return_value = _make_price_data()

        mock_mod = _mock_registry("7203", "E02144")
        with patch.dict("sys.modules", {"japan_finance_codes": mock_mod}):
            runner = CliRunner()
            result = runner.invoke(main, ["analyze", "7203", "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["ticker"] == "7203"
            assert data["edinet_code"] == "E02144"
            assert data["data_sources"]["financials"] is True
            assert data["data_sources"]["prices"] is True
            # Should have both financial and price-based factors
            assert len(data["factors"]) > 5
            assert len(data["observations"]) > 0


class TestScreenPipeline:
    """Test 5: jpfin screen (multiple tickers)."""

    @patch("jpfin.screen.analyze_ticker_sync")
    def test_screen_multiple_tickers(self, mock_analyze) -> None:
        def _result(ticker, **kwargs):
            return {
                "ticker": ticker,
                "factors": {"mom_3m": 0.1 if ticker == "7203" else -0.05},
                "data_sources": {"prices": True},
            }

        mock_analyze.side_effect = _result
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["screen", "7203", "6758", "--factor", "mom_3m"],
        )
        assert result.exit_code == 0
        assert "7203" in result.output
        assert "6758" in result.output


class TestBacktestPipeline:
    """Test 6: jpfin backtest (with fixture CSV)."""

    def test_backtest_with_fixture_csv(self, tmp_path) -> None:
        # Create fixture CSV with 2 tickers, ~6 months of daily data
        csv_path = tmp_path / "prices.csv"
        lines = ["date,ticker,close"]
        base = date(2024, 1, 1)
        for i in range(180):
            d = base + timedelta(days=i)
            if d.weekday() < 5:  # weekdays only
                lines.append(f"{d.isoformat()},7203,{2000 + i * 3}")
                lines.append(f"{d.isoformat()},6758,{1500 - i}")
        csv_path.write_text("\n".join(lines))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["backtest", "--csv", str(csv_path), "--factor", "mom_3m", "--top", "1"],
        )
        assert result.exit_code == 0
        assert "total_return" in result.output.lower() or "Total Return" in result.output
