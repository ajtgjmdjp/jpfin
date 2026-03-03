"""Portfolio-level analytics for backtest results.

Post-processing module that computes concentration metrics (HHI),
sector allocation, and turnover analysis from a completed backtest.
"""

from __future__ import annotations

from jpfin.models import BacktestResult, PortfolioAnalytics, SectorWeight


def _get_sector_map(tickers: list[str]) -> dict[str, str]:
    """Look up sector for each ticker via japan-finance-codes.

    Returns a dict mapping ticker → sector name.
    Unknown tickers are mapped to "Unknown".
    """
    try:
        from japan_finance_codes import CompanyRegistry

        registry = CompanyRegistry.create()
    except Exception:
        return {t: "Unknown" for t in tickers}

    sector_map: dict[str, str] = {}
    for ticker in tickers:
        company = registry.by_ticker(ticker)
        if company is not None and company.industry:
            sector_map[ticker] = company.industry
        else:
            sector_map[ticker] = "Unknown"
    return sector_map


def compute_portfolio_analytics(
    result: BacktestResult,
) -> PortfolioAnalytics:
    """Compute portfolio analytics from a completed backtest.

    Analyzes:
    - Concentration (HHI and effective N) per rebalance period
    - Sector allocation at each rebalance date
    - Turnover series (from factor_metrics if available)

    Args:
        result: Completed BacktestResult from run_backtest().

    Returns:
        PortfolioAnalytics with HHI, sector weights, and turnover.

    Raises:
        ValueError: If no holdings history is available.
    """
    if not result.holdings_history:
        raise ValueError("No holdings history available for portfolio analytics")

    # Collect all unique tickers across all periods
    all_tickers: set[str] = set()
    for hp in result.holdings_history:
        all_tickers.update(hp.holdings)

    # Look up sectors
    sector_map = _get_sector_map(sorted(all_tickers))

    # Per-period metrics
    hhi_values: list[float] = []
    sector_weight_list: list[SectorWeight] = []

    for hp in result.holdings_history:
        n = len(hp.holdings)
        if n == 0:
            continue

        # Equal-weight HHI: each holding has weight 1/n
        weight = 1.0 / n
        hhi = sum(weight**2 for _ in hp.holdings)
        hhi_values.append(hhi)

        # Sector allocation
        sector_counts: dict[str, int] = {}
        for ticker in hp.holdings:
            sector = sector_map.get(ticker, "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        weights: dict[str, float] = {
            sector: count / n for sector, count in sorted(sector_counts.items())
        }
        sector_weight_list.append(SectorWeight(date=hp.date, weights=weights))

    if not hhi_values:
        raise ValueError("No valid holding periods for portfolio analytics")

    mean_hhi = sum(hhi_values) / len(hhi_values)
    min_hhi = min(hhi_values)
    max_hhi = max(hhi_values)
    mean_effective_n = 1.0 / mean_hhi if mean_hhi > 0 else 0.0

    # Turnover from factor_metrics
    turnover_series: list[float] = []
    mean_turnover: float | None = None
    if result.factor_metrics is not None:
        turnover_series = result.factor_metrics.turnover_series
        mean_turnover = result.factor_metrics.mean_turnover

    return PortfolioAnalytics(
        mean_hhi=mean_hhi,
        min_hhi=min_hhi,
        max_hhi=max_hhi,
        mean_effective_n=mean_effective_n,
        sector_weights=sector_weight_list,
        turnover_series=turnover_series,
        mean_turnover=mean_turnover,
    )
