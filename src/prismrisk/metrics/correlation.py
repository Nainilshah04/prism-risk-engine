"""Correlation matrix and rolling pairwise correlation.

Enforces:
- Rule R-T2: Correlation matrix must be symmetric with unit diagonal.
"""

import pandas as pd
from prismrisk.utils.validation import validate_returns


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix across assets.

    Formula:
        Corr_{i,j} = Cov(r_i, r_j) / (sigma_i * sigma_j)

    Args:
        returns: DataFrame [date x asset] of daily simple returns.

    Returns:
        Square symmetric correlation DataFrame [asset x asset] with unit diagonal.
    """
    validate_returns(returns)
    corr = returns.corr(method="pearson")
    return corr


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling pairwise correlation between two return series.

    Args:
        series_a: First daily return Series.
        series_b: Second daily return Series.
        window: Lookback window in trading days.

    Returns:
        Series of rolling correlations. Leading NaNs dropped.
    """
    validate_returns(series_a)
    validate_returns(series_b)
    if window < 2:
        raise ValueError(f"Rolling window must be >= 2, got {window}")

    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    roll_corr = aligned.iloc[:, 0].rolling(window=window).corr(aligned.iloc[:, 1])
    return roll_corr.dropna()
