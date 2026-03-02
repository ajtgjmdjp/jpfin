"""Core analysis logic: fetch data → compute factors → format output."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from japan_finance_factors import FactorResult, compute_factors
from japan_finance_factors._models import FinancialData, PriceData
from japan_finance_factors.fetch import fetch_financial_data, fetch_price_data

from jpfin._utils import parse_date


async def _resolve_edinet_code(ticker: str) -> str | None:
    """Resolve ticker to EDINET code using japan-finance-codes."""
    try:
        from japan_finance_codes import CompanyRegistry

        registry = CompanyRegistry.create()
        result = registry.by_ticker(ticker)
        if result and result.edinet_code:
            return result.edinet_code
    except ImportError:
        pass
    except (FileNotFoundError, KeyError, ValueError) as exc:
        import logging

        logging.getLogger(__name__).debug(
            "EDINET code resolution failed for %s: %s",
            ticker,
            exc,
        )
    return None


async def _fetch_financials(
    ticker: str,
    edinet_code: str | None,
    year: int | None,
) -> FinancialData | None:
    """Fetch financial data with auto-detection of period."""
    if edinet_code is None:
        return None

    if year is not None:
        try:
            return await fetch_financial_data(
                edinet_code,
                period=str(year),
            )
        except Exception:
            return None

    # Auto-detect: try current year, then year-1, then year-2
    current_year = datetime.now().year
    for y in [current_year, current_year - 1, current_year - 2]:
        try:
            return await fetch_financial_data(
                edinet_code,
                period=str(y),
            )
        except Exception:
            continue
    return None


async def _fetch_prices(
    ticker: str,
    lookback_days: int = 400,
) -> PriceData | None:
    """Fetch price data from yfinance."""
    try:
        return await fetch_price_data(
            ticker,
            lookback_days=lookback_days,
        )
    except Exception:
        return None


def _filter_prices_by_date(
    pd: PriceData,
    cutoff: datetime,
) -> PriceData:
    """Filter price data to exclude dates after cutoff."""
    cutoff_date = cutoff.date() if isinstance(cutoff, datetime) else cutoff
    filtered = [
        p for p in pd.prices if (d := parse_date(p.get("date"))) is None or d <= cutoff_date
    ]
    return PriceData(
        ticker=pd.ticker,
        prices=filtered,
        market_cap=pd.market_cap,
    )


async def analyze_ticker(
    ticker: str,
    *,
    year: int | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Analyze a single ticker and return structured results.

    Returns:
        Dict with keys: ticker, edinet_code, as_of, factors, metadata
    """
    as_of = as_of or datetime.now()

    # 1. Resolve EDINET code
    edinet_code = await _resolve_edinet_code(ticker)

    # 2. Fetch data in parallel
    fd, pd = await asyncio.gather(
        _fetch_financials(ticker, edinet_code, year),
        _fetch_prices(ticker),
    )

    # 3. Enforce as_of: filter price data to exclude future dates
    if pd is not None:
        pd = _filter_prices_by_date(pd, as_of)

    # 4. Merge market_cap from yfinance into financial data
    if fd is not None and pd is not None and pd.market_cap is not None and fd.market_cap is None:
        fd = fd.model_copy(update={"market_cap": pd.market_cap})

    # 5. Compute factors
    kwargs: dict[str, Any] = {"as_of": as_of}
    if fd is not None:
        kwargs["financial_data"] = fd
    if pd is not None:
        kwargs["price_data"] = pd

    result: FactorResult = compute_factors(**kwargs)

    # 6. Build output
    return {
        "ticker": ticker,
        "edinet_code": edinet_code,
        "as_of": as_of.isoformat(),
        "period_end": (fd.period_end.isoformat() if fd and fd.period_end else None),
        "data_sources": {
            "financials": fd is not None,
            "prices": pd is not None,
            "price_points": len(pd.prices) if pd else 0,
            "market_cap": (pd.market_cap if pd and pd.market_cap else None),
        },
        "factors": result.to_dict(),
        "observations": [
            {
                "factor_id": obs.factor_id,
                "category": obs.category.value,
                "value": obs.value,
                "staleness_days": obs.staleness_days,
            }
            for obs in result.observations
        ],
    }


def analyze_ticker_sync(
    ticker: str,
    *,
    year: int | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for analyze_ticker."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    coro = analyze_ticker(ticker, year=year, as_of=as_of)
    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
        ) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
