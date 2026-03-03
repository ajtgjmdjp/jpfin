"""CLI helper functions to reduce duplication across commands."""

from __future__ import annotations

import sys

import click
from japan_finance_factors._models import PriceData


def validate_csv_or_db(
    csv_path: str | None,
    db_path: str | None,
) -> None:
    """Validate that exactly one of --csv or --db is specified.

    Exits with an error message if both or neither are provided.
    """
    if csv_path and db_path:
        click.echo("Error: specify --csv or --db, not both.", err=True)
        sys.exit(1)
    if not csv_path and not db_path:
        click.echo("Error: specify --csv or --db.", err=True)
        sys.exit(1)


def load_price_data(
    csv_path: str | None,
    db_path: str | None,
    *,
    verbose: bool = True,
) -> tuple[dict[str, PriceData], str]:
    """Validate inputs, load price data, and return (data, source_label).

    Calls :func:`validate_csv_or_db` first, then loads from the appropriate
    backend. Optionally prints a summary line to stderr.

    Returns:
        Tuple of (price_data dict, source file path string).
    """
    validate_csv_or_db(csv_path, db_path)

    if db_path:
        from jpfin.store import load_prices_db

        price_data = load_prices_db(db_path)
        source = db_path
    else:
        from jpfin.backtest import load_prices_csv

        assert csv_path is not None
        price_data = load_prices_csv(csv_path)
        source = csv_path

    if verbose:
        click.echo(
            f"  Loaded {len(price_data)} tickers from {source}",
            err=True,
        )

    return price_data, source
