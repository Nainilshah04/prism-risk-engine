"""Drawdown dynamics and Calmar ratio calculation.

Enforces:
- Rule R-F5: Drawdowns are negative numbers (DD <= 0).
- Rule R-T1: Hand-computed known-answer verification (e.g. [100, 110, 99, 121] -> MDD = -10%).
"""

from typing import Optional
import numpy as np
import pandas as pd
from prismrisk.models import DrawdownStats
from prismrisk.utils.validation import validate_returns

DEFAULT_TRADING_DAYS: int = 252


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute underwater drawdown series from daily simple returns.

    Formula:
        V_t = prod_{s=1}^t (1 + r_s)
        M_t = max_{s <= t} V_s
        DD_t = (V_t / M_t) - 1.0

    Args:
        returns: Daily simple returns Series.

    Returns:
        Drawdown Series indexed by date. Values are non-positive floats (<= 0.0).
    """
    validate_returns(returns)
    if returns.empty:
        return pd.Series(dtype=float)

    wealth_index = (1.0 + returns).cumprod()
    # Base starting wealth at 1.0
    running_max = np.maximum.accumulate(wealth_index)
    running_max = np.maximum(running_max, 1.0)
    dd = (wealth_index / running_max) - 1.0
    return pd.Series(dd, index=returns.index, name="drawdown")


def max_drawdown(returns: pd.Series) -> DrawdownStats:
    """Compute peak-to-trough maximum drawdown characteristics.

    Args:
        returns: Daily simple returns Series.

    Returns:
        DrawdownStats containing depth, peak date, trough date,
        recovery date, and duration days.
    """
    validate_returns(returns)
    if returns.empty:
        raise ValueError("Returns Series is empty.")

    dd = drawdown_series(returns)
    min_dd = float(dd.min())

    if min_dd >= 0.0:
        # Zero drawdown case (monotonically non-decreasing)
        first_date = returns.index[0]
        return DrawdownStats(
            depth=0.0,
            peak_date=first_date,
            trough_date=first_date,
            recovery_date=first_date,
            duration_days=0,
            to_trough_days=0,
            recovery_days=0,
        )

    trough_date = dd.idxmin()

    # Find the peak date immediately preceding the trough
    wealth = (1.0 + returns).cumprod()
    wealth_pre_trough = wealth.loc[:trough_date]
    peak_val = wealth_pre_trough.max()
    peak_candidates = wealth_pre_trough[wealth_pre_trough == peak_val].index
    peak_date = peak_candidates[-1]

    # Find recovery date (first date after trough where wealth >= peak_val)
    wealth_post_trough = wealth.loc[trough_date:]
    recovered_points = wealth_post_trough[wealth_post_trough >= peak_val]

    recovery_date: Optional[pd.Timestamp] = None
    recovery_days: Optional[int] = None
    if not recovered_points.empty:
        recovery_date = recovered_points.index[0]
        recovery_days = int((recovery_date - trough_date).days)
        duration_days = int((recovery_date - peak_date).days)
    else:
        # Not yet recovered through end of sample
        end_date = returns.index[-1]
        duration_days = int((end_date - peak_date).days)

    to_trough_days = int((trough_date - peak_date).days)

    return DrawdownStats(
        depth=min_dd,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        duration_days=duration_days,
        to_trough_days=to_trough_days,
        recovery_days=recovery_days,
    )


def calmar_ratio(returns: pd.Series, trading_days: int = DEFAULT_TRADING_DAYS) -> float:
    """Compute Calmar Ratio (CAGR / |Max Drawdown|).

    Formula:
        Calmar = CAGR / |max_drawdown|

    Args:
        returns: Daily simple returns Series.
        trading_days: Annualization factor (default: 252).

    Returns:
        Calmar ratio as float. Returns np.inf if max drawdown is 0.0.
    """
    from prismrisk.metrics.returns import cagr

    annual_cagr = cagr(returns, trading_days=trading_days)
    dd_stats = max_drawdown(returns)
    abs_mdd = abs(dd_stats.depth)

    if abs_mdd < 1e-8:
        return float(np.inf) if annual_cagr > 0 else 0.0

    return float(annual_cagr / abs_mdd)
