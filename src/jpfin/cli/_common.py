"""Shared CLI utilities."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from japan_finance_factors._models import PriceData

    from jpfin.models import BacktestResult, PortfolioAnalytics


def _resolve_factors(
    factors: tuple[str, ...],
    weights: tuple[float, ...],
) -> tuple[str | list[str], list[float] | None]:
    """Resolve CLI --factor/--weight tuples into backtest arguments."""
    if not factors:
        return "mom_3m", None
    if len(factors) == 1:
        return factors[0], None
    return list(factors), list(weights) if weights else None


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


def display_backtest_output(
    result: BacktestResult,
    *,
    rolling_window: int | None,
    rolling_step: int,
    pa: PortfolioAnalytics | None,
    fmt: str,
) -> None:
    """Display backtest results with optional rolling/portfolio analytics."""
    from jpfin.formatters import (
        format_backtest_table,
        format_json,
        format_portfolio_table,
        format_rolling_table,
    )

    if rolling_window is not None:
        from jpfin.rolling import compute_rolling

        try:
            ra = compute_rolling(result, window_months=rolling_window, step=rolling_step)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        if fmt == "json":
            data: list[dict[str, object]] = [ra.model_dump()]
            if pa is not None:
                data.append(pa.model_dump())
            click.echo(format_json(data))
        else:
            click.echo(format_backtest_table(result))
            click.echo(format_rolling_table(ra))
            if pa is not None:
                click.echo(format_portfolio_table(pa))
    elif fmt == "json":
        data = [result.model_dump()]
        if pa is not None:
            data.append(pa.model_dump())
        click.echo(format_json(data))
    else:
        click.echo(format_backtest_table(result))
        if pa is not None:
            click.echo(format_portfolio_table(pa))
