"""Tests for fetch module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from japan_finance_factors._models import PriceData

from jpfin.fetch import (
    _from_yf_ticker,
    _to_yf_ticker,
    fetch_prices,
    save_prices_csv,
    update_prices_csv,
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        save_prices_csv(data, path)
        loaded = load_prices_csv(path)

        assert set(loaded.keys()) == {"7203", "6758"}
        assert len(loaded["7203"].prices) == 2
        assert len(loaded["6758"].prices) == 1

    def test_empty_data(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
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


def _make_single_ticker_df() -> MagicMock:
    """Create a mock DataFrame mimicking yf.download() for a single ticker."""
    import pandas as pd

    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    df = pd.DataFrame(
        {
            "Open": [2000.0, 2010.0],
            "High": [2100.0, 2110.0],
            "Low": [1900.0, 1910.0],
            "Close": [2050.0, 2060.0],
            "Volume": [1000000.0, 1100000.0],
        },
        index=idx,
    )
    return df


def _make_multi_ticker_df() -> MagicMock:
    """Create a mock MultiIndex DataFrame for multiple tickers."""
    import pandas as pd

    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01")])
    arrays = [
        ["Close", "Close", "Open", "Open"],
        ["7203.T", "6758.T", "7203.T", "6758.T"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
    df = pd.DataFrame(
        [[2050.0, 3000.0, 2000.0, 2900.0]],
        index=idx,
        columns=columns,
    )
    return df


class TestFetchPrices:
    @patch("yfinance.download")
    def test_single_ticker(self, mock_dl: MagicMock) -> None:
        mock_dl.return_value = _make_single_ticker_df()
        result = fetch_prices(["7203"], progress=False)
        assert "7203" in result
        assert len(result["7203"].prices) == 2
        mock_dl.assert_called_once()

    @patch("yfinance.download")
    def test_multi_ticker(self, mock_dl: MagicMock) -> None:
        mock_dl.return_value = _make_multi_ticker_df()
        result = fetch_prices(["7203", "6758"], progress=False)
        assert "7203" in result
        assert "6758" in result

    @patch("yfinance.download")
    def test_empty_result(self, mock_dl: MagicMock) -> None:
        import pandas as pd

        mock_dl.return_value = pd.DataFrame()
        result = fetch_prices(["7203"], progress=False)
        assert result == {}

    @patch("yfinance.download")
    def test_download_error_skips_batch(self, mock_dl: MagicMock) -> None:
        mock_dl.side_effect = Exception("Network error")
        result = fetch_prices(["7203"], progress=False)
        assert result == {}

    def test_empty_tickers(self) -> None:
        result = fetch_prices([], progress=False)
        assert result == {}

    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            fetch_prices(["7203"], batch_size=0, progress=False)

    @patch("yfinance.download")
    def test_batching(self, mock_dl: MagicMock) -> None:
        """Tickers are split into batches of batch_size."""
        mock_dl.return_value = _make_single_ticker_df()
        fetch_prices(["7203", "6758", "9984"], batch_size=1, sleep_seconds=0, progress=False)
        assert mock_dl.call_count == 3

    @patch("yfinance.download")
    def test_start_date_overrides_period(self, mock_dl: MagicMock) -> None:
        mock_dl.return_value = _make_single_ticker_df()
        fetch_prices(["7203"], start_date="2024-01-01", progress=False)
        call_kwargs = mock_dl.call_args[1]
        assert call_kwargs["start"] == "2024-01-01"
        assert "period" not in call_kwargs


class TestUpdatePricesCsv:
    @patch("jpfin.fetch.fetch_prices")
    def test_update_new_file(self, mock_fetch: MagicMock) -> None:
        """Update on a non-existent file creates it."""
        mock_fetch.return_value = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-01", "close": 2050.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.csv"
            new_rows = update_prices_csv(path, tickers=["7203"])
            assert new_rows == 1
            assert path.exists()

    @patch("jpfin.fetch.fetch_prices")
    def test_update_merges_data(self, mock_fetch: MagicMock) -> None:
        """New rows are merged with existing data."""
        mock_fetch.return_value = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-03", "close": 2070.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.csv"
            # Create initial CSV
            save_prices_csv(
                {
                    "7203": PriceData(
                        ticker="7203",
                        prices=[
                            {"date": "2024-01-01", "close": 2050.0},
                            {"date": "2024-01-02", "close": 2060.0},
                        ],
                    ),
                },
                path,
            )
            new_rows = update_prices_csv(path, tickers=["7203"])
            assert new_rows == 1
            # Verify merged content
            from jpfin.backtest import load_prices_csv

            loaded = load_prices_csv(path)
            assert len(loaded["7203"].prices) == 3

    @patch("jpfin.fetch.fetch_prices")
    def test_update_dedup(self, mock_fetch: MagicMock) -> None:
        """Duplicate dates are not added."""
        mock_fetch.return_value = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-01", "close": 2050.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.csv"
            save_prices_csv(
                {
                    "7203": PriceData(
                        ticker="7203",
                        prices=[{"date": "2024-01-01", "close": 2050.0}],
                    ),
                },
                path,
            )
            new_rows = update_prices_csv(path, tickers=["7203"])
            assert new_rows == 0

    def test_update_no_tickers_empty_file(self) -> None:
        """Update with no tickers on empty file returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.csv"
            new_rows = update_prices_csv(path)
            assert new_rows == 0

    @patch("jpfin.fetch.fetch_prices")
    def test_update_subset_tickers(self, mock_fetch: MagicMock) -> None:
        """Update only fetches for requested subset."""
        mock_fetch.return_value = {
            "7203": PriceData(
                ticker="7203",
                prices=[{"date": "2024-01-03", "close": 2070.0}],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prices.csv"
            save_prices_csv(
                {
                    "7203": PriceData(
                        ticker="7203",
                        prices=[{"date": "2024-01-01", "close": 2050.0}],
                    ),
                    "6758": PriceData(
                        ticker="6758",
                        prices=[{"date": "2024-01-01", "close": 3000.0}],
                    ),
                },
                path,
            )
            update_prices_csv(path, tickers=["7203"])
            # fetch_prices should be called with only ["7203"]
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args
            assert call_args[0][0] == ["7203"]
