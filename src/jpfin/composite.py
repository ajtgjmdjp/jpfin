"""Multi-factor composite scoring via cross-sectional z-score.

Algorithm at each rebalance date:
1. Compute each factor for all tickers
2. Cross-sectional z-score: (value - mean) / std
3. Flip sign if HIGHER_IS_BETTER is False (so higher z = always better)
4. Clip z-scores to ±clip
5. Weighted sum: composite = sum(weight_i * z_i)
6. Rank by composite → select top N
"""

from __future__ import annotations

from jpfin.factor_registry import HIGHER_IS_BETTER, PRICE_FACTOR_FNS


def validate_composite_args(
    factors: list[str],
    weights: list[float] | None = None,
) -> tuple[list[str], list[float]]:
    """Validate factor/weight arguments.

    Explicit weights are validated but not normalized (kept as-is).
    When ``weights`` is ``None``, equal weights summing to 1.0 are used.

    Args:
        factors: List of factor names.
        weights: Optional weights. If ``None``, equal weights are used.
            Explicit weights are passed through unchanged.

    Returns:
        Tuple of (factors, weights).

    Raises:
        ValueError: If factors is empty, unknown factor, or weight count mismatch.
    """
    if not factors:
        raise ValueError("At least one factor is required")

    for f in factors:
        if f not in PRICE_FACTOR_FNS:
            supported = list(PRICE_FACTOR_FNS.keys())
            raise ValueError(f"Unknown factor: {f}. Supported: {supported}")

    if weights is None:
        weights = [1.0 / len(factors)] * len(factors)
    elif len(weights) != len(factors):
        raise ValueError(f"Weight count ({len(weights)}) must match factor count ({len(factors)})")

    return factors, weights


def _cross_sectional_zscore(
    values: dict[str, float],
    higher_is_better: bool,
    clip: float = 3.0,
) -> dict[str, float]:
    """Compute cross-sectional z-scores for a single factor.

    Args:
        values: Dict mapping ticker to factor value.
        higher_is_better: If ``False``, z-scores are sign-flipped so that
            higher z always means "better".
        clip: Clip z-scores to ``[-clip, +clip]``.

    Returns:
        Dict mapping ticker to z-score.
    """
    if len(values) < 2:
        return {t: 0.0 for t in values}

    vals = list(values.values())
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)

    if variance == 0:
        return {t: 0.0 for t in values}

    std = variance**0.5
    sign = 1.0 if higher_is_better else -1.0

    result: dict[str, float] = {}
    for ticker, val in values.items():
        z = sign * (val - mean) / std
        z = max(-clip, min(clip, z))
        result[ticker] = z
    return result


def compute_composite_scores(
    ticker_factor_values: dict[str, dict[str, float | None]],
    factors: list[str],
    weights: list[float],
    clip: float = 3.0,
) -> list[tuple[str, float]]:
    """Compute weighted composite scores across multiple factors.

    Args:
        ticker_factor_values: Dict mapping ticker to {factor_name: value}.
        factors: Ordered list of factor names.
        weights: Corresponding weights (same length as factors).
        clip: Z-score clip threshold.

    Returns:
        List of (ticker, composite_score) sorted descending by score.
    """
    # Step 1: For each factor, collect values across tickers
    factor_zscores: dict[str, dict[str, float]] = {}
    for factor_name in factors:
        raw_values: dict[str, float] = {}
        for ticker, fvals in ticker_factor_values.items():
            val = fvals.get(factor_name)
            if val is not None:
                raw_values[ticker] = val

        hib = HIGHER_IS_BETTER.get(factor_name, True)
        factor_zscores[factor_name] = _cross_sectional_zscore(raw_values, hib, clip)

    # Step 2: Weighted sum across factors per ticker
    all_tickers: set[str] = set()
    for zs in factor_zscores.values():
        all_tickers.update(zs.keys())

    scores: list[tuple[str, float]] = []
    for ticker in all_tickers:
        composite = 0.0
        has_any = False
        for factor_name, weight in zip(factors, weights, strict=True):
            z = factor_zscores.get(factor_name, {}).get(ticker)
            if z is not None:
                composite += weight * z
                has_any = True
        if has_any:
            scores.append((ticker, composite))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
