"""Tests for universe module."""

from __future__ import annotations

import tempfile
import warnings

import pytest

from jpfin.universe import (
    UniverseResult,
    _clean_tickers,
    list_sectors,
    list_universes,
    load_universe,
)


class TestCleanTickers:
    def test_basic(self) -> None:
        assert _clean_tickers(["7203", "6758"]) == ["6758", "7203"]

    def test_dedup(self) -> None:
        assert _clean_tickers(["7203", "7203", "6758"]) == ["6758", "7203"]

    def test_strip_whitespace(self) -> None:
        assert _clean_tickers(["  7203 ", "\t6758\n"]) == ["6758", "7203"]

    def test_skip_comments_and_blanks(self) -> None:
        assert _clean_tickers(["# comment", "", "7203", "  ", "6758"]) == [
            "6758",
            "7203",
        ]

    def test_strip_t_suffix(self) -> None:
        assert _clean_tickers(["7203.T", "6758"]) == ["6758", "7203"]

    def test_empty(self) -> None:
        assert _clean_tickers([]) == []


class TestListUniverses:
    def test_returns_list(self) -> None:
        result = list_universes()
        assert isinstance(result, list)
        assert "nikkei225" in result
        assert "topix_core30" in result


class TestListSectors:
    def test_returns_list(self) -> None:
        sectors = list_sectors()
        assert isinstance(sectors, list)
        assert len(sectors) > 0
        # These should be present for any EDINET-listed universe
        assert "電気機器" in sectors
        assert "銀行業" in sectors


class TestLoadUniverse:
    def test_explicit_tickers(self) -> None:
        result = load_universe(tickers=["7203", "6758"])
        assert isinstance(result, UniverseResult)
        assert result.source_type == "explicit"
        assert "6758" in result.tickers
        assert "7203" in result.tickers
        assert result.survivorship_risk == "low"

    def test_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("7203\n6758\n# comment\n\n9984\n")
            f.flush()
            result = load_universe(file=f.name)
        assert result.source_type == "explicit"
        assert len(result.tickers) == 3
        assert "7203" in result.tickers

    def test_from_file_not_found(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_universe(file="/nonexistent/file.txt")

    def test_from_file_empty(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# only comments\n\n")
            f.flush()
            with pytest.raises(ValueError, match="No valid tickers"):
                load_universe(file=f.name)

    def test_sector(self) -> None:
        result = load_universe(sector="電気機器")
        assert result.source_type == "sector"
        assert len(result.tickers) > 0
        assert result.survivorship_risk == "low"

    def test_sector_unknown(self) -> None:
        with pytest.raises(ValueError, match="No listed companies"):
            load_universe(sector="存在しないセクター")

    def test_index_nikkei225(self) -> None:
        result = load_universe(name="nikkei225")
        assert result.source_type == "index_snapshot"
        assert len(result.tickers) == 225
        assert result.survivorship_risk == "high"
        assert len(result.warnings) > 0

    def test_index_topix_core30(self) -> None:
        result = load_universe(name="topix_core30")
        assert result.source_type == "index_snapshot"
        assert len(result.tickers) == 30

    def test_unknown_index(self) -> None:
        with pytest.raises(ValueError, match="Unknown index"):
            load_universe(name="nonexistent_index")

    def test_no_source(self) -> None:
        with pytest.raises(ValueError, match="No universe specified"):
            load_universe()

    def test_priority_tickers_over_name(self) -> None:
        """Explicit tickers take priority over index name."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_universe(tickers=["7203"], name="nikkei225")
        assert result.source_type == "explicit"
        assert len(result.tickers) == 1
        # Should warn about multiple sources
        assert len(w) == 1
        assert "Multiple universe sources" in str(w[0].message)

    def test_multiple_sources_warns(self) -> None:
        """Warning is emitted when multiple sources are specified."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_universe(tickers=["7203"], sector="電気機器")
        assert result.source_type == "explicit"
        assert len(w) == 1
        assert "tickers" in str(w[0].message)
        assert "sector" in str(w[0].message)
