"""SQLite price data storage.

Provides save/load/update operations for price data with the same
``dict[str, PriceData]`` interface used by CSV functions. Uses stdlib
``sqlite3`` — no additional dependencies required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from japan_finance_factors._models import PriceData

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL NOT NULL,
    volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they don't exist."""
    conn.executescript(_SCHEMA)


def save_prices_db(
    data: dict[str, PriceData],
    path: str | Path,
) -> int:
    """Save PriceData dict to SQLite database.

    Uses ``INSERT OR REPLACE`` to handle upserts cleanly.

    Args:
        data: Dict mapping ticker to PriceData.
        path: Database file path (created if not exists).

    Returns:
        Number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, float | None, float | None, float | None, float, float | None]] = []
    for ticker, pd in sorted(data.items()):
        for p in pd.prices:
            close = p.get("close")
            if close is None:
                continue
            rows.append(
                (
                    ticker,
                    p.get("date", ""),
                    p.get("open"),
                    p.get("high"),
                    p.get("low"),
                    float(close),
                    p.get("volume"),
                )
            )

    conn = sqlite3.connect(str(path))
    try:
        _init_db(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    return len(rows)


def load_prices_db(
    path: str | Path,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, PriceData]:
    """Load price data from SQLite database.

    Returns the same ``dict[str, PriceData]`` as ``load_prices_csv()``,
    making it a drop-in replacement for the backtest pipeline.

    Args:
        path: Database file path.
        tickers: Filter by specific tickers. ``None`` for all.
        start_date: Filter by date >= *start_date* (ISO format).
        end_date: Filter by date <= *end_date* (ISO format).

    Returns:
        Dict mapping ticker to PriceData.

    Raises:
        FileNotFoundError: If database file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    where_clauses: list[str] = []
    params: list[str] = []

    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        where_clauses.append(f"ticker IN ({placeholders})")
        params.extend(tickers)

    if start_date:
        where_clauses.append("date >= ?")
        params.append(start_date)

    if end_date:
        where_clauses.append("date <= ?")
        params.append(end_date)

    where = " AND ".join(where_clauses)
    query = "SELECT ticker, date, open, high, low, close, volume FROM prices"
    if where:
        query += f" WHERE {where}"
    query += " ORDER BY ticker, date"

    conn = sqlite3.connect(str(path))
    try:
        _init_db(conn)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        ticker_prices: dict[str, list[dict[str, Any]]] = {}
        for row in cursor:
            ticker = row["ticker"]
            price: dict[str, Any] = {"date": row["date"]}
            for col in ("open", "high", "low", "close", "volume"):
                val = row[col]
                if val is not None:
                    price[col] = float(val)
            if ticker not in ticker_prices:
                ticker_prices[ticker] = []
            ticker_prices[ticker].append(price)
    finally:
        conn.close()

    return {
        ticker: PriceData(ticker=ticker, prices=prices) for ticker, prices in ticker_prices.items()
    }


def update_prices_db(
    path: str | Path,
    tickers: list[str] | None = None,
    **fetch_kwargs: Any,
) -> int:
    """Incremental update of an existing SQLite price database.

    Determines the latest date per ticker and fetches only newer data.

    Args:
        path: Database file path (created if not exists).
        tickers: Tickers to update. ``None`` reads all existing tickers from DB.
        **fetch_kwargs: Passed to ``fetch_prices()``.

    Returns:
        Number of new rows inserted.
    """
    from datetime import timedelta

    from jpfin.fetch import fetch_prices

    path = Path(path)

    # Read existing max dates
    max_dates: dict[str, str] = {}
    if path.exists():
        conn = sqlite3.connect(str(path))
        try:
            _init_db(conn)
            for row in conn.execute("SELECT ticker, MAX(date) FROM prices GROUP BY ticker"):
                if row[1]:
                    max_dates[row[0]] = row[1]
        finally:
            conn.close()

    if tickers is None:
        tickers = list(max_dates.keys())

    if not tickers:
        return 0

    # Compute start date from requested tickers only
    from datetime import date as date_type

    relevant_dates = []
    for t in tickers:
        if t in max_dates:
            relevant_dates.append(date_type.fromisoformat(max_dates[t]))

    if relevant_dates:
        earliest_next = min(relevant_dates) + timedelta(days=1)
        fetch_kwargs.setdefault("start_date", earliest_next.isoformat())

    new_data = fetch_prices(tickers, **fetch_kwargs)
    if not new_data:
        return 0

    # Count existing rows before insert so we can compute truly new rows
    conn = sqlite3.connect(str(path))
    try:
        _init_db(conn)
        row_before = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
        count_before = row_before[0] if row_before else 0
    finally:
        conn.close()

    save_prices_db(new_data, path)

    conn = sqlite3.connect(str(path))
    try:
        row_after = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
        count_after = row_after[0] if row_after else 0
    finally:
        conn.close()

    return count_after - count_before


def export_db_to_csv(
    db_path: str | Path,
    csv_path: str | Path,
    tickers: list[str] | None = None,
) -> int:
    """Export SQLite database to CSV format.

    Args:
        db_path: Source database path.
        csv_path: Destination CSV path.
        tickers: Filter by specific tickers. ``None`` for all.

    Returns:
        Number of rows exported.
    """
    from jpfin.fetch import save_prices_csv

    data = load_prices_db(db_path, tickers=tickers)
    return save_prices_csv(data, csv_path)


def import_csv_to_db(
    csv_path: str | Path,
    db_path: str | Path,
) -> int:
    """Import CSV file into SQLite database.

    Args:
        csv_path: Source CSV path.
        db_path: Destination database path (created if not exists).

    Returns:
        Number of rows imported.
    """
    from jpfin.backtest import load_prices_csv

    data = load_prices_csv(csv_path)
    return save_prices_db(data, db_path)


def db_info(path: str | Path) -> dict[str, Any]:
    """Get database statistics.

    Args:
        path: Database file path.

    Returns:
        Dict with ``ticker_count``, ``row_count``, ``date_min``, ``date_max``.

    Raises:
        FileNotFoundError: If database file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    conn = sqlite3.connect(str(path))
    try:
        _init_db(conn)
        row = conn.execute(
            "SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(date), MAX(date) FROM prices"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"ticker_count": 0, "row_count": 0, "date_min": None, "date_max": None}

    return {
        "ticker_count": row[0],
        "row_count": row[1],
        "date_min": row[2],
        "date_max": row[3],
    }
