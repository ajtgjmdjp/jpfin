"""Tests for SQLite price data storage."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from japan_finance_factors._models import PriceData

from jpfin.store import (
    db_info,
    export_db_to_csv,
    import_csv_to_db,
    load_prices_db,
    save_prices_db,
    update_prices_db,
)


def _sample_data() -> dict[str, PriceData]:
    return {
        "7203": PriceData(
            ticker="7203",
            prices=[
                {
                    "date": "2024-01-01",
                    "open": 2000.0,
                    "high": 2100.0,
                    "low": 1900.0,
                    "close": 2050.0,
                    "volume": 1000000.0,
                },
                {
                    "date": "2024-01-02",
                    "close": 2060.0,
                },
            ],
        ),
        "6758": PriceData(
            ticker="6758",
            prices=[
                {
                    "date": "2024-01-01",
                    "close": 3000.0,
                    "volume": 500000.0,
                },
            ],
        ),
    }


class TestSavePricesDb:
    def test_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            rows = save_prices_db(_sample_data(), path)
            assert rows == 3
            assert path.exists()

    def test_empty_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            rows = save_prices_db({}, path)
            assert rows == 0

    def test_skips_no_close(self) -> None:
        data = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-01", "open": 2000.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            rows = save_prices_db(data, path)
            assert rows == 0

    def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "test.db"
            rows = save_prices_db(_sample_data(), path)
            assert rows == 3
            assert path.exists()

    def test_upsert(self) -> None:
        """INSERT OR REPLACE updates existing rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            updated = {
                "7203": PriceData(
                    ticker="7203",
                    prices=[{"date": "2024-01-01", "close": 9999.0}],
                ),
            }
            save_prices_db(updated, path)

            loaded = load_prices_db(path, tickers=["7203"])
            prices_jan1 = [p for p in loaded["7203"].prices if p["date"] == "2024-01-01"]
            assert prices_jan1[0]["close"] == 9999.0


class TestLoadPricesDb:
    def test_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            loaded = load_prices_db(path)
            assert set(loaded.keys()) == {"7203", "6758"}
            assert len(loaded["7203"].prices) == 2
            assert len(loaded["6758"].prices) == 1

    def test_filter_tickers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            loaded = load_prices_db(path, tickers=["7203"])
            assert set(loaded.keys()) == {"7203"}

    def test_filter_start_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            loaded = load_prices_db(path, start_date="2024-01-02")
            assert "6758" not in loaded
            assert len(loaded["7203"].prices) == 1
            assert loaded["7203"].prices[0]["date"] == "2024-01-02"

    def test_filter_end_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            loaded = load_prices_db(path, end_date="2024-01-01")
            assert len(loaded["7203"].prices) == 1
            assert loaded["7203"].prices[0]["date"] == "2024-01-01"

    def test_filter_date_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            data = {
                "7203": PriceData(
                    ticker="7203",
                    prices=[
                        {"date": "2024-01-01", "close": 100.0},
                        {"date": "2024-01-02", "close": 101.0},
                        {"date": "2024-01-03", "close": 102.0},
                    ],
                ),
            }
            save_prices_db(data, path)

            loaded = load_prices_db(path, start_date="2024-01-02", end_date="2024-01-02")
            assert len(loaded["7203"].prices) == 1
            assert loaded["7203"].prices[0]["close"] == 101.0

    def test_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prices_db("/nonexistent/path.db")

    def test_empty_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db({}, path)
            loaded = load_prices_db(path)
            assert loaded == {}

    def test_preserves_ohlcv(self) -> None:
        """All OHLCV fields round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            loaded = load_prices_db(path, tickers=["7203"])
            p = loaded["7203"].prices[0]
            assert p["open"] == 2000.0
            assert p["high"] == 2100.0
            assert p["low"] == 1900.0
            assert p["close"] == 2050.0
            assert p["volume"] == 1000000.0


class TestUpdatePricesDb:
    @patch("jpfin.fetch.fetch_prices")
    def test_update_new_db(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-01", "close": 2050.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            new_rows = update_prices_db(path, tickers=["7203"])
            assert new_rows == 1
            assert path.exists()

    @patch("jpfin.fetch.fetch_prices")
    def test_update_existing(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-03", "close": 2070.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            new_rows = update_prices_db(path, tickers=["7203"])
            assert new_rows == 1

            loaded = load_prices_db(path, tickers=["7203"])
            assert len(loaded["7203"].prices) == 3

    @patch("jpfin.fetch.fetch_prices")
    def test_update_no_new_data(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)
            new_rows = update_prices_db(path, tickers=["7203"])
            assert new_rows == 0

    def test_update_no_tickers_empty_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            new_rows = update_prices_db(path)
            assert new_rows == 0

    @patch("jpfin.fetch.fetch_prices")
    def test_update_reads_existing_tickers(self, mock_fetch: MagicMock) -> None:
        """When tickers=None, reads from existing DB."""
        mock_fetch.return_value = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            update_prices_db(path)
            call_args = mock_fetch.call_args[0]
            assert set(call_args[0]) == {"6758", "7203"}

    @patch("jpfin.fetch.fetch_prices")
    def test_update_computes_start_date(self, mock_fetch: MagicMock) -> None:
        """Start date is computed from existing data."""
        mock_fetch.return_value = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            update_prices_db(path, tickers=["7203"])
            call_kwargs = mock_fetch.call_args[1]
            assert call_kwargs["start_date"] == "2024-01-03"


class TestExportImport:
    def test_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            csv_path = Path(tmpdir) / "export.csv"
            save_prices_db(_sample_data(), db_path)

            rows = export_db_to_csv(db_path, csv_path)
            assert rows == 3
            assert csv_path.exists()

    def test_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from jpfin.fetch import save_prices_csv

            csv_path = Path(tmpdir) / "input.csv"
            save_prices_csv(_sample_data(), csv_path)

            db_path = Path(tmpdir) / "imported.db"
            rows = import_csv_to_db(csv_path, db_path)
            assert rows == 3
            assert db_path.exists()

    def test_roundtrip_csv_db_csv(self) -> None:
        """CSV -> DB -> CSV preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from jpfin.backtest import load_prices_csv
            from jpfin.fetch import save_prices_csv

            csv1 = Path(tmpdir) / "original.csv"
            save_prices_csv(_sample_data(), csv1)

            db_path = Path(tmpdir) / "roundtrip.db"
            import_csv_to_db(csv1, db_path)

            csv2 = Path(tmpdir) / "exported.csv"
            export_db_to_csv(db_path, csv2)

            data1 = load_prices_csv(csv1)
            data2 = load_prices_csv(csv2)
            assert set(data1.keys()) == set(data2.keys())
            for ticker in data1:
                assert len(data1[ticker].prices) == len(data2[ticker].prices)

    def test_roundtrip_db_csv_db(self) -> None:
        """DB -> CSV -> DB preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = Path(tmpdir) / "original.db"
            save_prices_db(_sample_data(), db1)

            csv_path = Path(tmpdir) / "temp.csv"
            export_db_to_csv(db1, csv_path)

            db2 = Path(tmpdir) / "reimported.db"
            import_csv_to_db(csv_path, db2)

            data1 = load_prices_db(db1)
            data2 = load_prices_db(db2)
            assert set(data1.keys()) == set(data2.keys())
            for ticker in data1:
                assert len(data1[ticker].prices) == len(data2[ticker].prices)


class TestDbInfo:
    def test_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), path)

            info = db_info(path)
            assert info["ticker_count"] == 2
            assert info["row_count"] == 3
            assert info["date_min"] == "2024-01-01"
            assert info["date_max"] == "2024-01-02"

    def test_empty_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            save_prices_db({}, path)

            info = db_info(path)
            assert info["ticker_count"] == 0
            assert info["row_count"] == 0

    def test_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            db_info("/nonexistent/path.db")


class TestBacktestCompatibility:
    """Verify that backtest produces same results from CSV and DB sources."""

    def test_same_data_loaded(self) -> None:
        """load_prices_db returns same structure as load_prices_csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from jpfin.backtest import load_prices_csv
            from jpfin.fetch import save_prices_csv

            csv_path = Path(tmpdir) / "prices.csv"
            db_path = Path(tmpdir) / "prices.db"

            save_prices_csv(_sample_data(), csv_path)
            save_prices_db(_sample_data(), db_path)

            csv_data = load_prices_csv(csv_path)
            db_data = load_prices_db(db_path)

            assert set(csv_data.keys()) == set(db_data.keys())
            for ticker in csv_data:
                assert len(csv_data[ticker].prices) == len(db_data[ticker].prices)
                for cp, dp in zip(csv_data[ticker].prices, db_data[ticker].prices, strict=True):
                    assert cp["date"] == dp["date"]
                    assert cp["close"] == dp["close"]
