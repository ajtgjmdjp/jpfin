"""Batch price data fetching via yfinance.

Downloads OHLCV data for multiple Japanese equities and saves to CSV
compatible with ``backtest.load_prices_csv()``.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from japan_finance_factors._models import PriceData


def _to_yf_ticker(code: str) -> str:
    """Append .T suffix for Tokyo Stock Exchange if not present."""
    code = code.strip()
    if code.endswith(".T"):
        return code
    return f"{code}.T"


def _from_yf_ticker(yf_ticker: str) -> str:
    """Strip .T suffix to get bare ticker code."""
    if yf_ticker.endswith(".T"):
        return yf_ticker[:-2]
    return yf_ticker


def fetch_prices(
    tickers: list[str],
    *,
    period: str = "2y",
    start_date: str | None = None,
    end_date: str | None = None,
    batch_size: int = 20,
    sleep_seconds: float = 1.0,
    progress: bool = True,
) -> dict[str, PriceData]:
    """Batch-fetch daily OHLCV from yfinance.

    Uses ``yfinance.download()`` for efficient multi-ticker requests.

    Args:
        tickers: List of bare ticker codes (e.g. ``["7203", "6758"]``).
        period: yfinance period string (e.g. ``"1y"``, ``"2y"``, ``"5y"``).
            Ignored if *start_date* is provided.
        start_date: Fetch start date (ISO format). Overrides *period*.
        end_date: Fetch end date (ISO format). Defaults to today.
        batch_size: Number of tickers per ``yf.download()`` call.
        sleep_seconds: Sleep between batches (rate-limit protection).
        progress: Print progress to stderr.

    Returns:
        Dict mapping bare ticker code to PriceData.
    """
    import yfinance as yf

    if not tickers:
        return {}
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    yf_tickers = [_to_yf_ticker(t) for t in tickers]
    result: dict[str, PriceData] = {}

    for i in range(0, len(yf_tickers), batch_size):
        batch = yf_tickers[i : i + batch_size]
        if progress:
            print(
                f"  Fetching batch {i // batch_size + 1}"
                f"/{(len(yf_tickers) + batch_size - 1) // batch_size}"
                f" ({len(batch)} tickers)...",
                file=sys.stderr,
            )

        kwargs: dict[str, Any] = {
            "tickers": batch,
            "interval": "1d",
            "progress": False,
            "threads": True,
        }
        if start_date:
            kwargs["start"] = start_date
            if end_date:
                kwargs["end"] = end_date
        else:
            kwargs["period"] = period

        try:
            df = yf.download(**kwargs)
        except Exception as e:
            print(f"  Warning: batch download failed: {e}", file=sys.stderr)
            continue

        if df.empty:
            continue

        # yf.download returns MultiIndex columns for multiple tickers:
        # (Price, Ticker) e.g. ("Close", "7203.T")
        # For single ticker, columns are flat: "Close", "Open", etc.
        if len(batch) == 1:
            ticker_code = _from_yf_ticker(batch[0])
            prices: list[dict[str, Any]] = []
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, "date") else idx
                price_row: dict[str, Any] = {"date": d.isoformat()}
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    if col in df.columns:
                        val = row[col]
                        if val is not None and val == val:  # not NaN
                            price_row[col.lower()] = float(val)
                if "close" in price_row:
                    prices.append(price_row)
            if prices:
                result[ticker_code] = PriceData(ticker=ticker_code, prices=prices)
        else:
            # MultiIndex columns: iterate per ticker
            for yf_t in batch:
                ticker_code = _from_yf_ticker(yf_t)
                prices = []
                for idx, row in df.iterrows():
                    d = idx.date() if hasattr(idx, "date") else idx
                    price_row_: dict[str, Any] = {"date": d.isoformat()}
                    for col in ("Open", "High", "Low", "Close", "Volume"):
                        try:
                            val = row[(col, yf_t)]
                            if val is not None and val == val:  # not NaN
                                price_row_[col.lower()] = float(val)
                        except KeyError:
                            pass
                    if "close" in price_row_:
                        prices.append(price_row_)
                if prices:
                    result[ticker_code] = PriceData(ticker=ticker_code, prices=prices)

        # Rate-limit sleep between batches
        if i + batch_size < len(yf_tickers) and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return result


def save_prices_csv(
    data: dict[str, PriceData],
    path: str | Path,
) -> int:
    """Save PriceData dict to CSV compatible with ``load_prices_csv()``.

    Output columns: date, ticker, open, high, low, close, volume.

    Args:
        data: Dict mapping ticker to PriceData.
        path: Output CSV file path.

    Returns:
        Number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ticker, pd in sorted(data.items()):
        for p in pd.prices:
            row: dict[str, Any] = {"date": p.get("date", ""), "ticker": ticker}
            for field in ("open", "high", "low", "close", "volume"):
                val = p.get(field)
                if val is not None:
                    row[field] = val
            rows.append(row)

    rows.sort(key=lambda r: (r.get("date", ""), r.get("ticker", "")))

    fieldnames = ["date", "ticker", "open", "high", "low", "close", "volume"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def _read_existing_max_dates(path: Path) -> dict[str, date]:
    """Read existing CSV and return max date per ticker."""
    max_dates: dict[str, date] = {}
    if not path.exists():
        return max_dates
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            date_str = row.get("date", "").strip()
            if not ticker or not date_str:
                continue
            try:
                d = date.fromisoformat(date_str)
            except ValueError:
                continue
            if ticker not in max_dates or d > max_dates[ticker]:
                max_dates[ticker] = d
    return max_dates


def update_prices_csv(
    path: str | Path,
    tickers: list[str] | None = None,
    **fetch_kwargs: Any,
) -> int:
    """Incremental update of an existing price CSV.

    Reads the existing file, determines the latest date per ticker,
    fetches only newer data, and rewrites the file with merged results.

    Args:
        path: Path to existing CSV (created if not exists).
        tickers: Tickers to update. If None, updates all tickers in the file.
        **fetch_kwargs: Passed to ``fetch_prices()``.

    Returns:
        Number of new rows added.
    """
    from jpfin.backtest import load_prices_csv

    path = Path(path)

    # Read existing data
    existing: dict[str, PriceData] = {}
    if path.exists():
        existing = load_prices_csv(path)

    if tickers is None:
        tickers = list(existing.keys())

    if not tickers:
        return 0

    # Determine start dates per ticker
    max_dates = _read_existing_max_dates(path)

    # Find the earliest "next day" across requested tickers only
    from datetime import timedelta

    relevant_dates = [max_dates[t] for t in tickers if t in max_dates]
    if relevant_dates:
        earliest_next = min(relevant_dates) + timedelta(days=1)
        fetch_kwargs.setdefault("start_date", earliest_next.isoformat())

    # Fetch new data
    new_data = fetch_prices(tickers, **fetch_kwargs)

    # Merge: for each ticker, combine existing + new, dedup by date
    merged: dict[str, PriceData] = dict(existing)
    new_rows = 0

    for ticker, new_pd in new_data.items():
        existing_dates: set[str] = set()
        if ticker in merged:
            existing_dates = {p.get("date", "") for p in merged[ticker].prices}

        new_prices = [
            p for p in new_pd.prices if p.get("date", "") not in existing_dates
        ]
        new_rows += len(new_prices)

        if ticker in merged:
            all_prices = list(merged[ticker].prices) + new_prices
            merged[ticker] = PriceData(ticker=ticker, prices=all_prices)
        else:
            merged[ticker] = new_pd

    save_prices_csv(merged, path)
    return new_rows
