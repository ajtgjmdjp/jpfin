"""Backtest and run commands."""

from __future__ import annotations

import sys

import click

from jpfin.cli._common import _resolve_factors, display_backtest_output, load_price_data


@click.command()
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
    "--rolling",
    "rolling_window",
    default=None,
    type=int,
    help="Rolling window analysis in months (e.g., 12).",
)
@click.option(
    "--step",
    "rolling_step",
    default=1,
    type=int,
    help="Step size for rolling windows (default: 1).",
)
@click.option(
    "--portfolio-analytics",
    "portfolio_analytics",
    is_flag=True,
    help="Show portfolio analytics (HHI, sector allocation, turnover).",
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
    rolling_window: int | None,
    rolling_step: int,
    portfolio_analytics: bool,
    fmt: str,
) -> None:
    """Run a simple factor backtest on historical price data.

    Requires a CSV file or SQLite database with price data.

    Examples:

      jpfin backtest --csv prices.csv --factor mom_3m --top 5

      jpfin backtest --db prices.db --factor mom_3m --top 5

      jpfin backtest --csv p.csv -s mom_3m -s realized_vol_60d -w 0.7 -w 0.3

      jpfin backtest --db prices.db --factor mom_3m --top 5 --rolling 12

      jpfin backtest --db prices.db --factor mom_3m --top 5 --portfolio-analytics
    """
    price_data, _source = load_price_data(csv_path, db_path)

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
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Portfolio analytics (computed before output)
    pa = None
    if portfolio_analytics:
        from jpfin.portfolio import compute_portfolio_analytics

        try:
            pa = compute_portfolio_analytics(result)
        except ValueError as e:
            click.echo(f"Warning: {e}", err=True)

    display_backtest_output(
        result,
        rolling_window=rolling_window,
        rolling_step=rolling_step,
        pa=pa,
        fmt=fmt,
    )


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
    "--rolling",
    "rolling_window",
    default=None,
    type=int,
    help="Rolling window analysis in months (e.g., 12).",
)
@click.option(
    "--step",
    "rolling_step",
    default=1,
    type=int,
    help="Step size for rolling windows (default: 1).",
)
@click.option(
    "--portfolio-analytics",
    "portfolio_analytics",
    is_flag=True,
    help="Show portfolio analytics (HHI, sector allocation, turnover).",
)
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
    rolling_window: int | None,
    rolling_step: int,
    portfolio_analytics: bool,
    fmt: str,
) -> None:
    """One-shot: fetch data, run backtest, display results.

    Examples:

      jpfin run --universe nikkei225 --factor mom_3m --top 10

      jpfin run --tickers 7203,6758,9984 --factor mom_3m --benchmark topix

      jpfin run --no-fetch --db prices.db --factor mom_3m

      jpfin run --no-fetch --db prices.db --factor mom_3m --rolling 12

      jpfin run --no-fetch --db prices.db --factor mom_3m --portfolio-analytics
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

    # Portfolio analytics
    pa = None
    if portfolio_analytics:
        from jpfin.portfolio import compute_portfolio_analytics

        try:
            pa = compute_portfolio_analytics(result)
        except ValueError as e:
            click.echo(f"Warning: {e}", err=True)

    display_backtest_output(
        result,
        rolling_window=rolling_window,
        rolling_step=rolling_step,
        pa=pa,
        fmt=fmt,
    )
