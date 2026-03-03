"""CLI tests for the `run` one-shot command."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from japan_finance_factors._models import PriceData

from jpfin.cli import main
from jpfin.store import save_prices_db
from jpfin.universe import UniverseResult


def _sample_data() -> dict[str, PriceData]:
    """Generate price data with enough history for backtesting."""
    prices: dict[str, list[dict[str, float | str]]] = {}
    for ticker in ["A", "B", "C"]:
        ticker_prices = []
        close = 1000.0
        for month in range(1, 13):
            for day in range(1, 29):
                d = f"2023-{month:02d}-{day:02d}"
                close *= 1.001
                ticker_prices.append({"date": d, "close": round(close, 2)})
        prices[ticker] = ticker_prices
    return {t: PriceData(ticker=t, prices=p) for t, p in prices.items()}


class TestRunCommand:
    def test_no_source_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    @patch("jpfin.fetch.fetch_prices")
    @patch("jpfin.universe.load_universe")
    def test_basic_run(
        self,
        mock_load: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_load.return_value = UniverseResult(
            tickers=["A", "B", "C"],
            source_type="explicit",
            source_label="user-specified tickers",
        )
        mock_fetch.return_value = _sample_data()

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--tickers", "A,B,C", "--top", "2"])
        assert result.exit_code == 0
        assert "Total Return" in result.output

    def test_no_fetch_with_db(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), db_path)

            result = runner.invoke(
                main,
                ["run", "--no-fetch", "--db", str(db_path), "--top", "2"],
            )
            assert result.exit_code == 0
            assert "Total Return" in result.output

    def test_no_fetch_without_db_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--no-fetch"])
        assert result.exit_code != 0

    def test_json_output(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_prices_db(_sample_data(), db_path)

            result = runner.invoke(
                main,
                [
                    "run",
                    "--no-fetch",
                    "--db",
                    str(db_path),
                    "--top",
                    "2",
                    "--format",
                    "json",
                ],
            )
            assert result.exit_code == 0
            # Single result: format_json outputs a JSON object
            output = result.output
            json_start = output.find("{")
            assert json_start >= 0
            import json

            parsed = json.loads(output[json_start:])
            assert "performance" in parsed
