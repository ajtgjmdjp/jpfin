"""Analysis commands: event-study, decay, correlation."""

from __future__ import annotations

import sys

import click

from jpfin.cli._common import load_price_data
from jpfin.formatters import (
    format_correlation_table,
    format_decay_table,
    format_event_study_table,
    format_json,
)


@click.command()
@click.argument("ticker")
@click.argument("event_date")
@click.option(
    "--before",
    "-b",
    "before_days",
    default="1,5,20",
    help="Comma-separated days before event (default: 1,5,20).",
)
@click.option(
    "--after",
    "-a",
    "after_days",
    default="1,5,20",
    help="Comma-separated days after event (default: 1,5,20).",
)
@click.option(
    "--factor",
    "-s",
    "factors",
    multiple=True,
    help="Factors to compute (repeatable). Default: all price-based.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def event_study(
    ticker: str,
    event_date: str,
    before_days: str,
    after_days: str,
    factors: tuple[str, ...],
    fmt: str,
) -> None:
    """Factor snapshots around a corporate event.

    Computes price-based factors (momentum, volatility) at multiple
    points before and after an event date for event-study analysis.

    Examples:

      jpfin event-study 7203 2025-05-12

      jpfin event-study 7203 2025-05-12 --before 1,5,20 --after 1,5,20

      jpfin event-study 7203 2025-05-12 --factor mom_3m --factor realized_vol_60d
    """
    from jpfin.event_study import run_event_study

    before = [int(d) for d in before_days.split(",")]
    after = [int(d) for d in after_days.split(",")]
    factor_list = list(factors) if factors else None

    try:
        result = run_event_study(
            ticker,
            event_date,
            before_days=before,
            after_days=after,
            factors=factor_list,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(format_json([result.model_dump()]))
    else:
        click.echo(format_event_study_table(result))


@click.command()
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True),
    default=None,
    help="CSV file with price data.",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True),
    default=None,
    help="SQLite database with price data.",
)
@click.option(
    "--factor",
    "-s",
    default="mom_3m",
    help="Factor to analyze (e.g., mom_3m, mom_12m).",
)
@click.option(
    "--max-lag",
    default=6,
    type=int,
    help="Maximum forward lag in months (default: 6).",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def decay(
    csv_path: str | None,
    db_path: str | None,
    factor: str,
    max_lag: int,
    fmt: str,
) -> None:
    """Analyze factor signal persistence (IC term structure).

    Computes IC at lag 1, 2, ..., max-lag months to show how quickly
    a factor's predictive power decays.

    Examples:

      jpfin decay --db prices.db --factor mom_3m --max-lag 6

      jpfin decay --csv prices.csv --factor realized_vol_60d --format json
    """
    price_data, _source = load_price_data(csv_path, db_path)

    from jpfin.decay import compute_decay

    try:
        result = compute_decay(price_data, factor, max_lag=max_lag)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(format_json([result.model_dump()]))
    else:
        click.echo(format_decay_table(result))


@click.command()
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True),
    default=None,
    help="CSV file with price data.",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True),
    default=None,
    help="SQLite database with price data.",
)
@click.option(
    "--factor",
    "-s",
    "factors",
    multiple=True,
    help="Factor to include (repeatable). Default: all available.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def correlation(
    csv_path: str | None,
    db_path: str | None,
    factors: tuple[str, ...],
    fmt: str,
) -> None:
    """Compute factor cross-sectional correlation matrix.

    Shows pairwise rank correlations between factors to identify
    redundancy and diversification opportunities.

    Examples:

      jpfin correlation --db prices.db

      jpfin correlation --csv prices.csv --factor mom_3m --factor mom_12m

      jpfin correlation --db prices.db --format json
    """
    price_data, _source = load_price_data(csv_path, db_path)

    factor_list = list(factors) if factors else None

    from jpfin.correlation import compute_factor_correlation

    try:
        result = compute_factor_correlation(price_data, factor_list)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(format_json([result.model_dump()]))
    else:
        click.echo(format_correlation_table(result))
