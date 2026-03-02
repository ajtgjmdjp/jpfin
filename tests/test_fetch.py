"""Tests for fetch module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from japan_finance_factors._models import PriceData

from jpfin.fetch import (
    _from_yf_ticker,
    _to_yf_ticker,
    save_prices_csv,
)


class TestYfTickerConversion:
    def test_to_yf_ticker(self) -> None:
        assert _to_yf_ticker("7203") == "7203.T"

    def test_to_yf_ticker_already_suffixed(self) -> None:
        assert _to_yf_ticker("7203.T") == "7203.T"

    def test_to_yf_ticker_strip(self) -> None:
        assert _to_yf_ticker("  7203  ") == "7203.T"

    def test_from_yf_ticker(self) -> None:
        assert _from_yf_ticker("7203.T") == "7203"

    def test_from_yf_ticker_no_suffix(self) -> None:
        assert _from_yf_ticker("7203") == "7203"


class TestSavePricesCsv:
    def test_basic(self) -> None:
        data = {
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
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        rows = save_prices_csv(data, path)
        assert rows == 2

        # Verify CSV content
        content = Path(path).read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "date,ticker,open,high,low,close,volume"
        assert "7203" in lines[1]
        assert "2050.0" in lines[1]

    def test_multiple_tickers(self) -> None:
        data = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-01", "close": 2050.0}],
            ),
            "6758": PriceData(
                ticker="6758",
                prices=[{"date": "2024-01-01", "close": 3000.0}],
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        rows = save_prices_csv(data, path)
        assert rows == 2

        # Sorted by date then ticker
        content = Path(path).read_text()
        lines = content.strip().split("\n")
        assert "6758" in lines[1]  # 6758 before 7203
        assert "7203" in lines[2]

    def test_roundtrip_with_load(self) -> None:
        """CSV written by save_prices_csv can be read by load_prices_csv."""
        from jpfin.backtest import load_prices_csv

        data = {
            "7203": PriceData(
                ticker="7203",
                prices=[
                    {"date": "2024-01-01", "close": 2050.0, "volume": 1000000.0},
                    {"date": "2024-01-02", "close": 2060.0, "volume": 1100000.0},
                ],
            ),
            "6758": PriceData(
                ticker="6758",
                prices=[
                    {"date": "2024-01-01", "close": 3000.0},
                ],
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        save_prices_csv(data, path)
        loaded = load_prices_csv(path)

        assert set(loaded.keys()) == {"7203", "6758"}
        assert len(loaded["7203"].prices) == 2
        assert len(loaded["6758"].prices) == 1

    def test_empty_data(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        rows = save_prices_csv({}, path)
        assert rows == 0

    def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "prices.csv"
            data = {
                "7203": PriceData(
                    ticker="7203",
                    prices=[{"date": "2024-01-01", "close": 2050.0}],
                ),
            }
            rows = save_prices_csv(data, path)
            assert rows == 1
            assert path.exists()
