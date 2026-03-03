"""CLI tests for fetch, universe, and db commands."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from japan_finance_factors._models import PriceData

from jpfin.cli import main
from jpfin.store import save_prices_db
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


class TestFetchDbCommand:
    @patch("jpfin.store.save_prices_db", return_value=10)
    @patch("jpfin.fetch.fetch_prices")
    @patch("jpfin.universe.load_universe")
    def test_fetch_to_db(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        mock_load.return_value = UniverseResult(
            tickers=["7203"],
            source_type="explicit",
            source_label="user-specified tickers",
        )
        mock_fetch.return_value = {
            "7203": PriceData(ticker="7203", prices=[{"date": "2024-01-01", "close": 2050.0}]),
        }

        runner = CliRunner()
        result = runner.invoke(main, ["fetch", "--tickers", "7203", "--db", "/tmp/test.db"])
        assert result.exit_code == 0
        mock_save.assert_called_once()

    def test_fetch_out_and_db_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["fetch", "--tickers", "7203", "--out", "a.csv", "--db", "a.db"]
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    @patch("jpfin.store.update_prices_db", return_value=3)
    def test_update_db(self, mock_update: MagicMock) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            result = runner.invoke(main, ["fetch", "--update", "--db", f.name])
        assert result.exit_code == 0
        assert "3 new rows" in result.output


class TestBacktestDbCommand:
    def test_backtest_csv_and_db_error(self) -> None:
        runner = CliRunner()
        with (
            tempfile.NamedTemporaryFile(suffix=".csv") as csv_f,
            tempfile.NamedTemporaryFile(suffix=".db") as db_f,
        ):
            result = runner.invoke(main, ["backtest", "--csv", csv_f.name, "--db", db_f.name])
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_backtest_no_source_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["backtest"])
        assert result.exit_code != 0
        assert "specify --csv or --db" in result.output


class TestDbSubgroup:
    def test_db_info(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_prices_db(
                {
                    "7203": PriceData(
                        ticker="7203",
                        prices=[{"date": "2024-01-01", "close": 2050.0}],
                    ),
                },
                db_path,
            )
            result = runner.invoke(main, ["db", "info", str(db_path)])
            assert result.exit_code == 0
            assert "Tickers: 1" in result.output
            assert "Rows:    1" in result.output
            assert "2024-01-01" in result.output

    def test_db_export(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            csv_path = Path(tmpdir) / "export.csv"
            save_prices_db(
                {
                    "7203": PriceData(
                        ticker="7203",
                        prices=[{"date": "2024-01-01", "close": 2050.0}],
                    ),
                },
                db_path,
            )
            result = runner.invoke(main, ["db", "export", str(db_path), str(csv_path)])
            assert result.exit_code == 0
            assert csv_path.exists()
            assert "1 rows" in result.output

    def test_db_import(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            from jpfin.fetch import save_prices_csv

            csv_path = Path(tmpdir) / "input.csv"
            db_path = Path(tmpdir) / "imported.db"
            save_prices_csv(
                {
                    "7203": PriceData(
                        ticker="7203",
                        prices=[{"date": "2024-01-01", "close": 2050.0}],
                    ),
                },
                csv_path,
            )
            result = runner.invoke(main, ["db", "import", str(csv_path), str(db_path)])
            assert result.exit_code == 0
            assert db_path.exists()
            assert "1 rows" in result.output


class TestUniverseListCommand:
    def test_list(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["universe", "list"])
        assert result.exit_code == 0
        assert "nikkei225" in result.output
        assert "topix_core30" in result.output
