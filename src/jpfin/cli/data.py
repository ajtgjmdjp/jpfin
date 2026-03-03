"""Data commands: fetch, db, universe."""

from __future__ import annotations

import sys

import click


@click.command()
@click.option(
    "--tickers",
    "-t",
    default=None,
    help="Comma-separated ticker codes (e.g., 7203,6758,9984).",
)
@click.option(
    "--universe",
    "-u",
    "universe_name",
    default=None,
    help="Built-in index snapshot (nikkei225, topix_core30).",
)
@click.option(
    "--universe-file",
    type=click.Path(exists=True),
    default=None,
    help="Text file with one ticker per line.",
)
@click.option(
    "--sector",
    default=None,
    help="Industry sector from EDINET (e.g., '\u96fb\u6c17\u6a5f\u5668').",
)
@click.option(
    "--period",
    "-p",
    default="2y",
    help="yfinance period (1y, 2y, 5y, max).",
)
@click.option(
    "--out",
    "-o",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output CSV path.",
)
@click.option(
    "--db",
    "db_path",
    default=None,
    type=click.Path(),
    help="Output SQLite database path.",
)
@click.option(
    "--update",
    is_flag=True,
    help="Incremental update of existing file.",
)
@click.option(
    "--batch-size",
    default=20,
    type=click.IntRange(1),
    help="Tickers per yfinance request.",
)
def fetch(
    tickers: str | None,
    universe_name: str | None,
    universe_file: str | None,
    sector: str | None,
    period: str,
    output_path: str | None,
    db_path: str | None,
    update: bool,
    batch_size: int,
) -> None:
    """Fetch price data from yfinance and save to CSV or SQLite.

    The output is compatible with `jpfin backtest`.

    Examples:

      jpfin fetch --tickers 7203,6758,9984 --period 2y --out prices.csv

      jpfin fetch --universe nikkei225 --db prices.db

      jpfin fetch --update --db prices.db
    """
    from jpfin.universe import load_universe

    use_db = db_path is not None
    if not use_db and output_path is None:
        output_path = "prices.csv"

    if use_db and output_path:
        click.echo("Error: specify --out or --db, not both.", err=True)
        sys.exit(1)

    # Resolve universe (used for both fetch and update)
    ticker_list_parsed = tickers.split(",") if tickers else None
    has_universe_source = any([ticker_list_parsed, universe_name, universe_file, sector])

    if update:
        resolved_tickers: list[str] | None = None
        if has_universe_source:
            try:
                univ = load_universe(
                    name=universe_name,
                    file=universe_file,
                    tickers=ticker_list_parsed,
                    sector=sector,
                )
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
            resolved_tickers = univ.tickers

        try:
            if use_db:
                from jpfin.store import update_prices_db

                assert db_path is not None
                new_rows = update_prices_db(
                    db_path,
                    tickers=resolved_tickers,
                    batch_size=batch_size,
                )
                target = db_path
            else:
                from jpfin.fetch import update_prices_csv

                assert output_path is not None
                new_rows = update_prices_csv(
                    output_path,
                    tickers=resolved_tickers,
                    batch_size=batch_size,
                )
                target = output_path
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        click.echo(f"  Updated {target}: {new_rows} new rows added.", err=True)
        return

    if not has_universe_source:
        click.echo(
            "Error: No universe specified. "
            "Use --tickers, --universe-file, --sector, or --universe.",
            err=True,
        )
        sys.exit(1)

    try:
        universe_result = load_universe(
            name=universe_name,
            file=universe_file,
            tickers=ticker_list_parsed,
            sector=sector,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    for warning in universe_result.warnings:
        click.echo(f"  Warning: {warning}", err=True)

    click.echo(
        f"  Universe: {universe_result.source_label} ({len(universe_result.tickers)} tickers)",
        err=True,
    )

    from jpfin.fetch import fetch_prices

    try:
        data = fetch_prices(
            universe_result.tickers,
            period=period,
            batch_size=batch_size,
        )
    except Exception as e:
        click.echo(f"Error fetching data: {e}", err=True)
        sys.exit(1)

    if not data:
        click.echo("  No data fetched.", err=True)
        sys.exit(1)

    if use_db:
        from jpfin.store import save_prices_db

        assert db_path is not None
        rows = save_prices_db(data, db_path)
        click.echo(
            f"  Saved {len(data)} tickers, {rows} rows to {db_path}",
            err=True,
        )
    else:
        from jpfin.fetch import save_prices_csv

        assert output_path is not None
        rows = save_prices_csv(data, output_path)
        click.echo(
            f"  Saved {len(data)} tickers, {rows} rows to {output_path}",
            err=True,
        )


@click.group()
def db() -> None:
    """SQLite database utilities."""


@db.command("info")
@click.argument("db_path", type=click.Path(exists=True))
def db_info_cmd(db_path: str) -> None:
    """Show database statistics.

    Examples:

      jpfin db info prices.db
    """
    from jpfin.store import db_info

    info = db_info(db_path)
    click.echo(f"\n  Database: {db_path}")
    click.echo(f"  Tickers: {info['ticker_count']}")
    click.echo(f"  Rows:    {info['row_count']}")
    click.echo(f"  From:    {info['date_min'] or 'N/A'}")
    click.echo(f"  To:      {info['date_max'] or 'N/A'}")
    click.echo()


@db.command("export")
@click.argument("db_path", type=click.Path(exists=True))
@click.argument("csv_path", type=click.Path())
def db_export(db_path: str, csv_path: str) -> None:
    """Export SQLite database to CSV.

    Examples:

      jpfin db export prices.db prices.csv
    """
    from jpfin.store import export_db_to_csv

    rows = export_db_to_csv(db_path, csv_path)
    click.echo(f"  Exported {rows} rows to {csv_path}", err=True)


@db.command("import")
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("db_path", type=click.Path())
def db_import(csv_path: str, db_path: str) -> None:
    """Import CSV file into SQLite database.

    Examples:

      jpfin db import prices.csv prices.db
    """
    from jpfin.store import import_csv_to_db

    rows = import_csv_to_db(csv_path, db_path)
    click.echo(f"  Imported {rows} rows to {db_path}", err=True)


@click.group()
def universe() -> None:
    """Manage stock universes."""


@universe.command("list")
def universe_list() -> None:
    """List available built-in universes and sectors.

    Examples:

      jpfin universe list
    """
    from jpfin.universe import list_sectors, list_universes

    indices = list_universes()
    click.echo("\n  Built-in index snapshots:")
    for name in indices:
        click.echo(f"    {name}")

    sectors = list_sectors()
    click.echo(f"\n  Available sectors ({len(sectors)}):")
    for s in sectors:
        click.echo(f"    {s}")
    click.echo()
