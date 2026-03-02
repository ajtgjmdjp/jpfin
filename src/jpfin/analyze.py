"""Core analysis logic: fetch data → compute factors → format output."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from japan_finance_factors import FactorResult, compute_factors
from japan_finance_factors._models import FinancialData, PriceData
from japan_finance_factors.fetch import fetch_financial_data, fetch_price_data


async def _resolve_edinet_code(ticker: str) -> str | None:
    """Resolve ticker to EDINET code using japan-finance-codes."""
    try:
        from japan_finance_codes import CompanyRegistry

        registry = await CompanyRegistry.create()
        result = registry.by_ticker(ticker)
        if result and result.edinet_code:
            return result.edinet_code
    except (ImportError, Exception):
        pass
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
            return await fetch_financial_data(edinet_code, period=str(year))
        except (ValueError, Exception):
            return None

    # Auto-detect: try current year, then year-1, then year-2
    current_year = datetime.now().year
    for y in [current_year, current_year - 1, current_year - 2]:
        try:
            fd = await fetch_financial_data(edinet_code, period=str(y))
            return fd
        except (ValueError, Exception):
            continue
    return None


async def _fetch_prices(ticker: str, lookback_days: int = 400) -> PriceData | None:
    """Fetch price data from yfinance."""
    try:
        return await fetch_price_data(ticker, lookback_days=lookback_days)
    except Exception:
        return None


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
    fd_task = _fetch_financials(ticker, edinet_code, year)
    pd_task = _fetch_prices(ticker)
    fd, pd = await asyncio.gather(fd_task, pd_task)

    # 3. Enforce as_of: filter price data to exclude future dates
    if pd is not None:
        cutoff = as_of.date() if isinstance(as_of, datetime) else as_of
        filtered_prices = []
        for p in pd.prices:
            d = p.get("date")
            if d is None:
                filtered_prices.append(p)
                continue
            if isinstance(d, str):
                try:
                    from datetime import date as _date

                    d_val = _date.fromisoformat(d)
                except ValueError:
                    filtered_prices.append(p)
                    continue
            elif isinstance(d, datetime):
                d_val = d.date()
            else:
                d_val = d
            if d_val <= cutoff:
                filtered_prices.append(p)
        pd = PriceData(ticker=pd.ticker, prices=filtered_prices, market_cap=pd.market_cap)

    # 4. Merge market_cap from yfinance into financial data
    if (
        fd is not None
        and pd is not None
        and pd.market_cap is not None
        and fd.market_cap is None
    ):
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
        "period_end": fd.period_end.isoformat() if fd and fd.period_end else None,
        "data_sources": {
            "financials": fd is not None,
            "prices": pd is not None,
            "price_points": len(pd.prices) if pd else 0,
            "market_cap": pd.market_cap if pd and pd.market_cap else None,
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
