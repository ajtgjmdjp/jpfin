"""Tests for output formatters."""

from __future__ import annotations

import json

from jpfin.formatters import format_json, format_table


def _sample_result() -> dict:
    return {
        "ticker": "7203",
        "edinet_code": "E02144",
        "as_of": "2026-03-01T00:00:00",
        "period_end": "2025-03-31",
        "data_sources": {
            "financials": True,
            "prices": True,
            "price_points": 266,
            "market_cap": 51_340_663_324_672,
        },
        "factors": {
            "ev_ebitda": 10.89,
            "roe": 0.08,
            "mom_3m": 0.235,
            "realized_vol_60d": None,
        },
        "observations": [
            {
                "factor_id": "ev_ebitda",
                "category": "value",
                "value": 10.89,
                "staleness_days": 249,
            },
            {
                "factor_id": "roe",
                "category": "quality",
                "value": 0.08,
                "staleness_days": 249,
            },
            {
                "factor_id": "mom_3m",
                "category": "momentum",
                "value": 0.235,
                "staleness_days": None,
            },
            {
                "factor_id": "realized_vol_60d",
                "category": "risk",
                "value": None,
                "staleness_days": None,
            },
        ],
    }


class TestFormatTable:
    def test_contains_ticker(self) -> None:
        output = format_table(_sample_result())
        assert "7203" in output

    def test_contains_edinet(self) -> None:
        output = format_table(_sample_result())
        assert "E02144" in output

    def test_contains_factors(self) -> None:
        output = format_table(_sample_result())
        assert "ev_ebitda" in output
        assert "roe" in output
        assert "mom_3m" in output

    def test_contains_categories(self) -> None:
        output = format_table(_sample_result())
        assert "[VALUE]" in output
        assert "[QUALITY]" in output
        assert "[MOMENTUM]" in output

    def test_na_for_none(self) -> None:
        output = format_table(_sample_result())
        assert "N/A" in output

    def test_market_cap_formatted(self) -> None:
        output = format_table(_sample_result())
        assert "T JPY" in output

    def test_staleness_shown(self) -> None:
        output = format_table(_sample_result())
        assert "249d stale" in output

    def test_no_observations(self) -> None:
        result = _sample_result()
        result["observations"] = []
        output = format_table(result)
        assert "No factors computed" in output


class TestFormatJson:
    def test_single_result(self) -> None:
        output = format_json([_sample_result()])
        parsed = json.loads(output)
        assert parsed["ticker"] == "7203"

    def test_multiple_results(self) -> None:
        r1 = _sample_result()
        r2 = _sample_result()
        r2["ticker"] = "6758"
        output = format_json([r1, r2])
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_valid_json(self) -> None:
        output = format_json([_sample_result()])
        json.loads(output)  # Should not raise
