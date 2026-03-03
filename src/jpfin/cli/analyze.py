"""Analyze and screen commands."""

from __future__ import annotations

import sys

import click

from jpfin.analyze import analyze_ticker_sync
from jpfin.formatters import format_json, format_screen_table, format_table


@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option(
    "--year",
    "-y",
    type=int,
    default=None,
    help="Fiscal year (e.g., 2024). Auto-detect if omitted.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
def analyze(tickers: tuple[str, ...], year: int | None, fmt: str) -> None:
    """Analyze one or more tickers.

    Examples:

      jpfin analyze 7203

      jpfin analyze 7203 6758 9984 --format json

      jpfin analyze 7203 --year 2024
    """
    results = []
    errors = 0
    for ticker in tickers:
        try:
            result = analyze_ticker_sync(ticker, year=year)
            results.append(result)

            if fmt == "table":
                click.echo(format_table(result))
        except KeyboardInterrupt:
            click.echo("\nAborted.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error analyzing {ticker}: {e}", err=True)
            errors += 1

    if fmt == "json":
        click.echo(format_json(results))

    if errors > 0:
        sys.exit(1)


@click.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option(
    "--factor",
    "-s",
    default="roe",
    help="Factor to rank by (e.g., roe, ev_ebitda, mom_3m).",
)
@click.option("--year", "-y", type=int, default=None, help="Fiscal year.")
@click.option(
    "--ascending",
    "-a",
    is_flag=True,
    help="Rank ascending (lower is better).",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def screen(
    tickers: tuple[str, ...],
    factor: str,
    year: int | None,
    ascending: bool,
    fmt: str,
) -> None:
    """Screen and rank tickers by a factor.

    Examples:

      jpfin screen 7203 6758 9984 8306 --factor roe

      jpfin screen 7203 6758 9984 --factor ev_ebitda --ascending
    """
    from jpfin.screen import screen_tickers

    results = screen_tickers(
        list(tickers),
        factor,
        year=year,
        ascending=ascending,
    )

    if fmt == "json":
        click.echo(format_json(results))
    else:
        click.echo(format_screen_table(results, factor, ascending))
