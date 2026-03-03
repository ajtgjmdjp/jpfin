"""CLI entry point for jpfin."""

from __future__ import annotations

import click

from jpfin import __version__


@click.group()
@click.version_option(__version__, prog_name="jpfin")
def main() -> None:
    """Japanese equity factor analysis CLI."""


def _register_commands() -> None:
    from jpfin.cli.analysis import correlation, decay, event_study
    from jpfin.cli.analyze import analyze, screen
    from jpfin.cli.backtest import backtest, run
    from jpfin.cli.data import db, fetch, universe

    for cmd in (analyze, screen, backtest, run, fetch, decay, correlation):
        main.add_command(cmd)
    main.add_command(event_study, "event-study")
    main.add_command(db)
    main.add_command(universe)


_register_commands()

if __name__ == "__main__":
    main()
