"""Typed result models for jpfin output."""

from __future__ import annotations

from pydantic import BaseModel


class BacktestError(Exception):
    """Raised when a backtest cannot run due to insufficient data."""


class PerformanceMetrics(BaseModel):
    """Summary statistics for a backtest."""

    total_return: float
    cagr: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float


class MonthlyReturn(BaseModel):
    """Single month return in a backtest."""

    period_start: str
    period_end: str
    monthly_return: float
    cumulative: float


class HoldingsPeriod(BaseModel):
    """Holdings snapshot at one rebalance date."""

    date: str
    holdings: list[str]
    factor_values: dict[str, float]
    skipped: list[str] | None = None


class DataQuality(BaseModel):
    """Data coverage and gap statistics for a backtest."""

    total_rebalances: int
    skipped_rebalances: int
    total_ticker_slots: int
    ffill_count: int
    skip_count: int

    @property
    def coverage(self) -> float:
        """Fraction of ticker-slots with usable prices."""
        if self.total_ticker_slots == 0:
            return 0.0
        return 1.0 - self.skip_count / self.total_ticker_slots


class BacktestResult(BaseModel):
    """Complete backtest output."""

    factor: str
    top_n: int
    period: str
    months: int
    performance: PerformanceMetrics
    monthly_returns: list[MonthlyReturn]
    holdings_history: list[HoldingsPeriod]
    data_quality: DataQuality | None = None


class EventStudyWindow(BaseModel):
    """A single time window in an event study."""

    window: str
    as_of: str | None
    factors: dict[str, float | None]


class EventStudyResult(BaseModel):
    """Complete event-study output."""

    ticker: str
    event_date: str
    windows: list[EventStudyWindow]
