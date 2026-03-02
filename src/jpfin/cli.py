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
    required=True,
    type=click.Path(exists=True),
    help="CSV file with price data (date,ticker,close).",
)
@click.option(
    "--factor",
    "-s",
    default="mom_3m",
    help="Price-based factor (mom_3m, mom_12m, realized_vol_60d).",
)
@click.option(
    "--top",
    "top_n",
    default=5,
    type=int,
    help="Number of top tickers to hold.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def backtest(csv_path: str, factor: str, top_n: int, fmt: str) -> None:
    """Run a simple factor backtest on historical price data.

    Requires a CSV file with columns: date, ticker, close.

    Examples:

      jpfin backtest --csv prices.csv --factor mom_3m --top 5
    """
    from jpfin.backtest import load_prices_csv, run_backtest

    price_data = load_prices_csv(csv_path)
    click.echo(
        f"  Loaded {len(price_data)} tickers from {csv_path}",
        err=True,
    )

    try:
        result = run_backtest(price_data, factor, top_n=top_n)
    except (ValueError, Exception) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(format_json([result.model_dump()]))
    else:
        perf = result.performance
        click.echo(f"\n  {'=' * 50}")
        click.echo(f"  Backtest: Top {top_n} by {factor}")
        click.echo(f"  Period: {result.period}")
        click.echo(f"  Months: {result.months}")
        click.echo(f"  {'=' * 50}")
        click.echo(f"  Total Return:    {perf.total_return:>8.1%}")
        click.echo(f"  CAGR:            {perf.cagr:>8.1%}")
        click.echo(f"  Annualized Vol:  {perf.annualized_vol:>8.1%}")
        click.echo(f"  Sharpe Ratio:    {perf.sharpe_ratio:>8.2f}")
        click.echo(f"  Max Drawdown:    {perf.max_drawdown:>8.1%}")
        click.echo()


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
    default="prices.csv",
    type=click.Path(),
    help="Output CSV path.",
)
@click.option(
    "--update",
    is_flag=True,
    help="Incremental update of existing CSV.",
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
    output_path: str,
    update: bool,
    batch_size: int,
) -> None:
    """Fetch price data from yfinance and save to CSV.

    The output CSV is compatible with `jpfin backtest --csv`.

    Examples:

      jpfin fetch --tickers 7203,6758,9984 --period 2y --out prices.csv

      jpfin fetch --universe nikkei225 --period 1y --out nk225.csv

      jpfin fetch --sector 電気機器 --out electronics.csv

      jpfin fetch --update --out prices.csv
    """
    from jpfin.fetch import fetch_prices, save_prices_csv, update_prices_csv
    from jpfin.universe import load_universe

    # Resolve universe (used for both fetch and update)
    ticker_list_parsed = tickers.split(",") if tickers else None
    has_universe_source = any([ticker_list_parsed, universe_name, universe_file, sector])

    if update:
        # For --update, resolve tickers from universe options if provided,
        # otherwise update all tickers already in the CSV.
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
            new_rows = update_prices_csv(
                output_path,
                tickers=resolved_tickers,
                batch_size=batch_size,
            )
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        click.echo(f"  Updated {output_path}: {new_rows} new rows added.", err=True)
        return

    if not has_universe_source:
        click.echo(
            "Error: No universe specified. "
            "Use --tickers, --universe-file, --sector, or --universe.",
            err=True,
        )
        sys.exit(1)

    try:
        universe = load_universe(
            name=universe_name,
            file=universe_file,
            tickers=ticker_list_parsed,
            sector=sector,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Print warnings
    for warning in universe.warnings:
        click.echo(f"  Warning: {warning}", err=True)

    click.echo(
        f"  Universe: {universe.source_label} ({len(universe.tickers)} tickers)",
        err=True,
    )

    # Fetch
    try:
        data = fetch_prices(
            universe.tickers,
            period=period,
            batch_size=batch_size,
        )
    except Exception as e:
        click.echo(f"Error fetching data: {e}", err=True)
        sys.exit(1)

    if not data:
        click.echo("  No data fetched.", err=True)
        sys.exit(1)

    rows = save_prices_csv(data, output_path)
    click.echo(
        f"  Saved {len(data)} tickers, {rows} rows to {output_path}",
        err=True,
    )


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
