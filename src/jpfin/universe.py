"""Universe management for factor backtesting.

Three-tier universe resolution:
1. **explicit**: User-provided ticker list or file (most accurate).
2. **sector**: Filter by industry from japan-finance-codes (no index bias).
3. **index_snapshot**: Static index membership snapshot (survivorship bias warning).
"""

from __future__ import annotations

import json
import warnings
from importlib import resources
from pathlib import Path
from typing import Any

from japan_finance_codes import CompanyRegistry

# ---------------------------------------------------------------------------
# Index snapshots — static, with explicit snapshot dates
# ---------------------------------------------------------------------------

_INDEX_SNAPSHOTS: dict[str, dict[str, Any]] | None = None


def _load_index_snapshots() -> dict[str, dict[str, Any]]:
    global _INDEX_SNAPSHOTS
    if _INDEX_SNAPSHOTS is not None:
        return _INDEX_SNAPSHOTS

    _INDEX_SNAPSHOTS = {}
    data_pkg = resources.files("jpfin") / "data"
    for name in ("nikkei225", "topix_core30"):
        ref = data_pkg / f"{name}.json"
        with resources.as_file(ref) as p:
            if p.exists():
                _INDEX_SNAPSHOTS[name] = json.loads(p.read_text())
    return _INDEX_SNAPSHOTS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_universes() -> list[str]:
    """Return names of built-in index snapshots."""
    snapshots = _load_index_snapshots()
    return sorted(snapshots.keys())


def list_sectors() -> list[str]:
    """Return sorted list of available industry names from japan-finance-codes."""
    registry = CompanyRegistry.create()
    industries: set[str] = set()
    for c in registry.companies:
        if c.industry and c.is_listed and c.ticker:
            industries.add(c.industry)
    return sorted(industries)


def load_universe(
    *,
    name: str | None = None,
    file: str | Path | None = None,
    tickers: list[str] | None = None,
    sector: str | None = None,
) -> UniverseResult:
    """Resolve a universe of tickers.

    Priority: *tickers* > *file* > *sector* > *name*.

    Args:
        name: Built-in index snapshot name (e.g. ``"nikkei225"``).
        file: Path to text file with one ticker per line.
        tickers: Explicit list of ticker codes.
        sector: Industry name from japan-finance-codes (e.g. ``"電気機器"``).

    Returns:
        UniverseResult with ticker list and metadata.

    Raises:
        ValueError: If no universe source is specified or name is unknown.
    """
    # Warn if multiple sources are specified
    sources = [
        ("tickers", tickers),
        ("file", file),
        ("sector", sector),
        ("name", name),
    ]
    active = [label for label, val in sources if val]
    if len(active) > 1:
        warnings.warn(
            f"Multiple universe sources specified ({', '.join(active)}). "
            f"Using priority: tickers > file > sector > name.",
            stacklevel=2,
        )

    if tickers:
        return UniverseResult(
            tickers=_clean_tickers(tickers),
            source_type="explicit",
            source_label="user-specified tickers",
        )

    if file:
        path = Path(file)
        if not path.exists():
            raise ValueError(f"Universe file not found: {path}")
        lines = path.read_text().strip().splitlines()
        clean = _clean_tickers(lines)
        if not clean:
            raise ValueError(f"No valid tickers found in {path}")
        return UniverseResult(
            tickers=clean,
            source_type="explicit",
            source_label=f"file: {path.name}",
        )

    if sector:
        registry = CompanyRegistry.create()
        matched = [
            c.ticker
            for c in registry.companies
            if c.industry == sector and c.is_listed and c.ticker
        ]
        if not matched:
            available = list_sectors()
            raise ValueError(
                f"No listed companies found for sector '{sector}'. "
                f"Available sectors: {', '.join(available[:10])}..."
            )
        return UniverseResult(
            tickers=sorted(set(matched)),
            source_type="sector",
            source_label=f"sector: {sector}",
            survivorship_risk="low",
        )

    if name:
        snapshots = _load_index_snapshots()
        if name not in snapshots:
            available = list_universes()
            raise ValueError(
                f"Unknown index '{name}'. Available: {', '.join(available)}"
            )
        snapshot = snapshots[name]
        return UniverseResult(
            tickers=snapshot["tickers"],
            source_type="index_snapshot",
            source_label=f"{name} (as of {snapshot.get('snapshot_date', 'unknown')})",
            snapshot_date=snapshot.get("snapshot_date"),
            survivorship_risk="high",
            warnings=[
                f"Using {name} snapshot from {snapshot.get('snapshot_date', 'unknown')}. "
                "Index membership changes over time — backtests longer than 1 year "
                "may exhibit survivorship bias. Consider --sector for unbiased analysis.",
            ],
        )

    raise ValueError(
        "No universe specified. Use --tickers, --universe-file, --sector, or --universe."
    )


def _clean_tickers(raw: list[str]) -> list[str]:
    """Strip whitespace, remove comments/blanks, deduplicate, sort."""
    result: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = t.strip()
        if not t or t.startswith("#"):
            continue
        # Strip .T suffix if present
        if t.endswith(".T"):
            t = t[:-2]
        if t not in seen:
            seen.add(t)
            result.append(t)
    return sorted(result)


class UniverseResult:
    """Result of universe resolution with metadata."""

    __slots__ = (
        "snapshot_date",
        "source_label",
        "source_type",
        "survivorship_risk",
        "tickers",
        "warnings",
    )

    def __init__(
        self,
        *,
        tickers: list[str],
        source_type: str,
        source_label: str,
        snapshot_date: str | None = None,
        survivorship_risk: str = "low",
        warnings: list[str] | None = None,
    ) -> None:
        self.tickers = tickers
        self.source_type = source_type
        self.source_label = source_label
        self.snapshot_date = snapshot_date
        self.survivorship_risk = survivorship_risk
        self.warnings = warnings or []
