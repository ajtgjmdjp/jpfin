"""CLI entry point for jpfin."""

from __future__ import annotations

import sys

import click

from jpfin import __version__
from jpfin.analyze import analyze_ticker_sync
from jpfin.formatters import format_json, format_table


@click.group()
@click.version_option(__version__, prog_name="jpfin")
def main() -> None:
    """Japanese equity factor analysis CLI."""


@main.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--year", "-y", type=int, default=None, help="Fiscal year (e.g., 2024). Auto-detect if omitted.")
@click.option("--format", "-f", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format.")
def analyze(tickers: tuple[str, ...], year: int | None, fmt: str) -> None:
    """Analyze one or more tickers.

    Examples:

      jpfin analyze 7203

      jpfin analyze 7203 6758 9984 --format json

      jpfin analyze 7203 --year 2024
    """
    results = []
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

    if fmt == "json":
        click.echo(format_json(results))


if __name__ == "__main__":
    main()
