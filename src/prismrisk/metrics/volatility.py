"""Volatility and dispersion metrics.

Enforces:
- Rule R-F4: Volatility scales by sqrt(252).
- Rule R-C5: No magic numbers; constants parameterised with explicit defaults.
"""

import numpy as np
import pandas as pd
from prismrisk.utils.validation import validate_returns

DEFAULT_TRADING_DAYS: int = 252
DEFAULT_EWMA_LAMBDA: float = 0.94


def annualised_vol(returns: pd.Series, trading_days: int = DEFAULT_TRADING_DAYS) -> float:
    """Compute sample annualized volatility from daily return series.

    Formula:
        sigma_ann = std(r, ddof=1) * sqrt(trading_days)

    Args:
        returns: Daily return series.
        trading_days: Annualization factor (default: 252).

    Returns:
        Annualized standard deviation as float. Always >= 0.0.
    """
    validate_returns(returns)
    if len(returns) < 2:
        return 0.0
    daily_std = float(returns.std(ddof=1))
    return float(daily_std * np.sqrt(trading_days))


def rolling_vol(
    returns: pd.Series,
    window: int,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> pd.Series:
    """Compute rolling annualized volatility over a moving lookback window.

    Formula:
        sigma_{roll, t} = std(r_{t-W+1:t}) * sqrt(trading_days)

    Args:
        returns: Daily return series.
        window: Moving lookback window in trading days (e.g. 30, 90).
        trading_days: Annualization factor (default: 252).

    Returns:
        Series of rolling annualized volatility. Leading NaN entries are dropped.
    """
    validate_returns(returns)
    if window < 2:
        raise ValueError(f"Rolling window must be >= 2, got {window}")

    roll = returns.rolling(window=window).std(ddof=1) * np.sqrt(trading_days)
    return roll.dropna()


def ewma_vol(
    returns: pd.Series,
    lam: float = DEFAULT_EWMA_LAMBDA,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> pd.Series:
    """Compute RiskMetrics Exponentially Weighted Moving Average (EWMA) volatility.

    Formula:
        sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2
        sigma_{EWMA, t} = sqrt(sigma_t^2 * trading_days)

    Args:
        returns: Daily return series.
        lam: Exponential decay factor lambda (default: 0.94).
        trading_days: Annualization factor (default: 252).

    Returns:
        Series of annualized EWMA volatility.
    """
    validate_returns(returns)
    if not (0.0 < lam < 1.0):
        raise ValueError(f"EWMA lambda must be in (0, 1), got {lam}")

    r2 = returns**2
    # In pandas ewm, alpha = 1 - lambda
    alpha = 1.0 - lam
    ewma_var = r2.ewm(alpha=alpha, adjust=False).mean()
    ewma_annual = np.sqrt(ewma_var * trading_days)
    return ewma_annual
