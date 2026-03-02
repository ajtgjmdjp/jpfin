"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from jpfin.cli import main


def _mock_result(ticker: str = "7203") -> dict:
    return {
        "ticker": ticker,
        "edinet_code": "E02144",
        "as_of": "2026-03-01T00:00:00",
        "period_end": "2025-03-31",
        "data_sources": {
            "financials": True,
            "prices": True,
            "price_points": 266,
            "market_cap": 51e12,
        },
        "factors": {"ev_ebitda": 10.89, "roe": 0.08},
        "observations": [
            {
                "factor_id": "ev_ebitda",
                "category": "value",
                "value": 10.89,
                "staleness_days": 249,
            },
            {
                "factor_id": "roe",
                "category": "quality",
                "value": 0.08,
                "staleness_days": 249,
            },
        ],
    }


class TestAnalyzeCommand:
    @patch("jpfin.cli.analyze_ticker_sync")
    def test_single_ticker_table(self, mock_analyze) -> None:
        mock_analyze.return_value = _mock_result()
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203"])
        assert result.exit_code == 0
        assert "7203" in result.output
        assert "ev_ebitda" in result.output

    @patch("jpfin.cli.analyze_ticker_sync")
    def test_single_ticker_json(self, mock_analyze) -> None:
        mock_analyze.return_value = _mock_result()
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203", "--format", "json"])
        assert result.exit_code == 0
        import json

        parsed = json.loads(result.output)
        assert parsed["ticker"] == "7203"

    @patch("jpfin.cli.analyze_ticker_sync")
    def test_multiple_tickers(self, mock_analyze) -> None:
        mock_analyze.side_effect = [_mock_result("7203"), _mock_result("6758")]
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203", "6758"])
        assert result.exit_code == 0
        assert "7203" in result.output

    @patch("jpfin.cli.analyze_ticker_sync")
    def test_year_option(self, mock_analyze) -> None:
        mock_analyze.return_value = _mock_result()
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "7203", "--year", "2024"])
        assert result.exit_code == 0
        mock_analyze.assert_called_once()
        _, kwargs = mock_analyze.call_args
        assert kwargs["year"] == 2024

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __import__("jpfin").__version__ in result.output

    @patch("jpfin.cli.analyze_ticker_sync")
    def test_error_exits_nonzero(self, mock_analyze) -> None:
        mock_analyze.side_effect = RuntimeError("API error")
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "9999"])
        assert result.exit_code != 0

    def test_no_ticker_shows_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["analyze"])
        assert result.exit_code != 0
