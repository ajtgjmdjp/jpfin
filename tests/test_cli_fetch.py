"""CLI tests for fetch and universe commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from japan_finance_factors._models import PriceData

from jpfin.cli import main
from jpfin.universe import UniverseResult


class TestFetchCommand:
    def test_no_universe_specified(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["fetch"])
        assert result.exit_code != 0
        assert "No universe specified" in result.output

    @patch("jpfin.fetch.save_prices_csv", return_value=10)
    @patch("jpfin.fetch.fetch_prices")
    @patch("jpfin.universe.load_universe")
    def test_fetch_with_tickers(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        mock_load.return_value = UniverseResult(
            tickers=["7203", "6758"],
            source_type="explicit",
            source_label="user-specified tickers",
        )
        mock_fetch.return_value = {
            "7203": PriceData(ticker="7203", prices=[{"date": "2024-01-01", "close": 2050.0}]),
        }

        runner = CliRunner()
        result = runner.invoke(main, ["fetch", "--tickers", "7203,6758", "--out", "/tmp/test.csv"])
        assert result.exit_code == 0

    def test_batch_size_zero_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["fetch", "--tickers", "7203", "--batch-size", "0"])
        assert result.exit_code != 0

    def test_batch_size_negative_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["fetch", "--tickers", "7203", "--batch-size", "-1"])
        assert result.exit_code != 0

    @patch("jpfin.fetch.update_prices_csv", return_value=5)
    def test_update_mode(self, mock_update: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["fetch", "--update", "--out", "/tmp/test.csv"])
        assert result.exit_code == 0
        assert "5 new rows" in result.output
        mock_update.assert_called_once()

    @patch("jpfin.fetch.update_prices_csv", return_value=5)
    @patch("jpfin.universe.load_universe")
    def test_update_with_universe(
        self,
        mock_load: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """--update + --universe resolves tickers from universe."""
        mock_load.return_value = UniverseResult(
            tickers=["7203", "6758"],
            source_type="index_snapshot",
            source_label="nikkei225",
        )

        runner = CliRunner()
        result = runner.invoke(
            main, ["fetch", "--update", "--universe", "nikkei225", "--out", "/tmp/test.csv"]
        )
        assert result.exit_code == 0
        mock_load.assert_called_once()
        # update should receive the resolved tickers
        call_kwargs = mock_update.call_args
        assert call_kwargs[1]["tickers"] == ["7203", "6758"]


class TestUniverseListCommand:
    def test_list(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["universe", "list"])
        assert result.exit_code == 0
        assert "nikkei225" in result.output
        assert "topix_core30" in result.output
