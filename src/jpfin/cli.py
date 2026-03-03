"""CLI entry point for jpfin."""

from __future__ import annotations

import sys

import click

from jpfin import __version__
from jpfin.analyze import analyze_ticker_sync
from jpfin.formatters import format_backtest_table, format_json, format_table


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


@click.group()
@click.version_option(__version__, prog_name="jpfin")
def main() -> None:
    """Japanese equity factor analysis CLI."""


@main.command()
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


@main.command()
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
        click.echo(f"\n  Screening by: {factor} ({'asc' if ascending else 'desc'})")
        click.echo(f"  {'Rank':>4s}  {'Ticker':>8s}  {'Value':>12s}")
        click.echo(f"  {'-' * 4}  {'-' * 8}  {'-' * 12}")
        for r in results:
            rank = f"{r['rank']:>4d}" if r["rank"] is not None else "   -"
            val = (
                f"{r['factor_value']:>12.4f}" if r["factor_value"] is not None else "         N/A"
            )
            click.echo(f"  {rank}  {r['ticker']:>8s}  {val}")
        click.echo()


@main.command()
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True),
    default=None,
    help="CSV file with price data (date,ticker,close).",
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
    help="Price-based factor (repeatable for composite). Default: mom_3m.",
)
@click.option(
    "--weight",
    "-w",
    "weights",
    multiple=True,
    type=float,
    help="Weight per factor (repeatable, same order as --factor).",
)
@click.option(
    "--top",
    "top_n",
    default=5,
    type=int,
    help="Number of top tickers to hold.",
)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark for comparison (topix, nikkei225).",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def backtest(
    csv_path: str | None,
    db_path: str | None,
    factors: tuple[str, ...],
    weights: tuple[float, ...],
    top_n: int,
    benchmark: str | None,
    fmt: str,
) -> None:
    """Run a simple factor backtest on historical price data.

    Requires a CSV file or SQLite database with price data.

    Examples:

      jpfin backtest --csv prices.csv --factor mom_3m --top 5

      jpfin backtest --db prices.db --factor mom_3m --top 5

      jpfin backtest --csv p.csv -s mom_3m -s realized_vol_60d -w 0.7 -w 0.3
    """
    if csv_path and db_path:
        click.echo("Error: specify --csv or --db, not both.", err=True)
        sys.exit(1)
    if not csv_path and not db_path:
        click.echo("Error: specify --csv or --db.", err=True)
        sys.exit(1)

    if db_path:
        from jpfin.store import load_prices_db

        price_data = load_prices_db(db_path)
        source = db_path
    else:
        from jpfin.backtest import load_prices_csv

        assert csv_path is not None
        price_data = load_prices_csv(csv_path)
        source = csv_path

    click.echo(
        f"  Loaded {len(price_data)} tickers from {source}",
        err=True,
    )

    factor_arg, weight_arg = _resolve_factors(factors, weights)

    from jpfin.backtest import run_backtest

    try:
        result = run_backtest(
            price_data,
            factor_arg,
            weights=weight_arg,
            benchmark=benchmark,
            top_n=top_n,
        )
    except (ValueError, Exception) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(format_json([result.model_dump()]))
    else:
        click.echo(format_backtest_table(result))


@main.command("event-study")
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
        click.echo(f"\n  {'=' * 60}")
        click.echo(f"  Event Study: {ticker} @ {event_date}")
        click.echo(f"  {'=' * 60}")
        click.echo(f"  {'Window':>8s}  {'as_of':>12s}", nl=False)
        # Collect all factor names from windows
        all_factors: list[str] = []
        for w in result.windows:
            for k in w.factors:
                if k not in all_factors:
                    all_factors.append(k)
        for f in all_factors:
            click.echo(f"  {f:>16s}", nl=False)
        click.echo()
        click.echo(f"  {'-' * 8}  {'-' * 12}", nl=False)
        for _ in all_factors:
            click.echo(f"  {'-' * 16}", nl=False)
        click.echo()

        for w in result.windows:
            as_of = w.as_of[:10] if w.as_of else "N/A"
            click.echo(f"  {w.window:>8s}  {as_of:>12s}", nl=False)
            for f in all_factors:
                val = w.factors.get(f)
                if val is not None:
                    click.echo(f"  {val:>16.4f}", nl=False)
                else:
                    click.echo(f"  {'N/A':>16s}", nl=False)
            click.echo()
        click.echo()


@main.command()
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
    help="Industry sector from EDINET (e.g., '電気機器').",
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


@main.group()
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


@main.command()
@click.option(
    "--tickers",
    "-t",
    default=None,
    help="Comma-separated ticker codes.",
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
@click.option("--sector", default=None, help="Industry sector.")
@click.option(
    "--db",
    "db_path",
    default=None,
    type=click.Path(),
    help="SQLite database for price data (default: in-memory).",
)
@click.option(
    "--period",
    "-p",
    default="2y",
    help="yfinance period (1y, 2y, 5y, max).",
)
@click.option(
    "--factor",
    "-s",
    "factors",
    multiple=True,
    help="Factor (repeatable for composite). Default: mom_3m.",
)
@click.option(
    "--weight",
    "-w",
    "weights",
    multiple=True,
    type=float,
    help="Weight per factor.",
)
@click.option("--top", "top_n", default=5, type=int, help="Top N tickers.")
@click.option("--benchmark", default=None, help="Benchmark (topix, nikkei225).")
@click.option("--no-fetch", is_flag=True, help="Skip data fetch, use existing DB.")
@click.option("--batch-size", default=20, type=click.IntRange(1), help="Tickers per request.")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def run(
    tickers: str | None,
    universe_name: str | None,
    universe_file: str | None,
    sector: str | None,
    db_path: str | None,
    period: str,
    factors: tuple[str, ...],
    weights: tuple[float, ...],
    top_n: int,
    benchmark: str | None,
    no_fetch: bool,
    batch_size: int,
    fmt: str,
) -> None:
    """One-shot: fetch data, run backtest, display results.

    Examples:

      jpfin run --universe nikkei225 --factor mom_3m --top 10

      jpfin run --tickers 7203,6758,9984 --factor mom_3m --benchmark topix

      jpfin run --no-fetch --db prices.db --factor mom_3m
    """
    from jpfin.universe import load_universe

    # Resolve universe
    ticker_list = tickers.split(",") if tickers else None
    has_source = any([ticker_list, universe_name, universe_file, sector])

    if not has_source and not (no_fetch and db_path):
        click.echo(
            "Error: specify tickers/universe/sector, or use --no-fetch --db.",
            err=True,
        )
        sys.exit(1)

    resolved_tickers: list[str] | None = None
    if has_source:
        try:
            univ = load_universe(
                name=universe_name,
                file=universe_file,
                tickers=ticker_list,
                sector=sector,
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        resolved_tickers = univ.tickers
        for w in univ.warnings:
            click.echo(f"  Warning: {w}", err=True)
        click.echo(
            f"  Universe: {univ.source_label} ({len(univ.tickers)} tickers)",
            err=True,
        )

    # Determine DB path
    use_tempdb = db_path is None
    if use_tempdb:
        import tempfile as _tempfile

        _fd, actual_db = _tempfile.mkstemp(suffix=".db")
        import os as _os

        _os.close(_fd)
    else:
        assert db_path is not None
        actual_db = db_path

    # Fetch / update data
    if not no_fetch:
        if resolved_tickers is None:
            click.echo("Error: no tickers to fetch.", err=True)
            sys.exit(1)
        from pathlib import Path

        from jpfin.fetch import fetch_prices
        from jpfin.store import save_prices_db, update_prices_db

        try:
            if not Path(actual_db).exists():
                data = fetch_prices(
                    resolved_tickers,
                    period=period,
                    batch_size=batch_size,
                )
                if data:
                    save_prices_db(data, actual_db)
                    click.echo(
                        f"  Fetched {len(data)} tickers to {actual_db}",
                        err=True,
                    )
            else:
                new_rows = update_prices_db(
                    actual_db,
                    tickers=resolved_tickers,
                    batch_size=batch_size,
                )
                click.echo(
                    f"  Updated {actual_db}: {new_rows} new rows.",
                    err=True,
                )
        except Exception as e:
            click.echo(f"Error fetching data: {e}", err=True)
            sys.exit(1)

    # Load data
    from jpfin.store import load_prices_db

    try:
        price_data = load_prices_db(actual_db)
    except FileNotFoundError:
        click.echo(f"Error: database not found: {actual_db}", err=True)
        sys.exit(1)

    if not price_data:
        click.echo("Error: no price data available.", err=True)
        sys.exit(1)

    click.echo(f"  Loaded {len(price_data)} tickers.", err=True)

    factor_arg, weight_arg = _resolve_factors(factors, weights)

    # Run backtest
    from jpfin.backtest import run_backtest

    try:
        result = run_backtest(
            price_data,
            factor_arg,
            weights=weight_arg,
            benchmark=benchmark,
            top_n=top_n,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(format_json([result.model_dump()]))
    else:
        click.echo(format_backtest_table(result))


@main.group()
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


if __name__ == "__main__":
    main()
