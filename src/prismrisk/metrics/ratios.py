"""Risk-adjusted performance and benchmark-relative ratios.

Enforces:
- Rule R-F4: 252 trading days annualization.
- Rule R-F6: Risk-free rate converted to daily consistently.
- Rule R-F7: Common sample. Ensure aligned indices when comparing against benchmark.
"""

import numpy as np
import pandas as pd
from prismrisk.utils.validation import validate_returns

DEFAULT_TRADING_DAYS: int = 252


def to_daily_risk_free(rf_annual: float, trading_days: int = DEFAULT_TRADING_DAYS) -> float:
    """Convert an annualized risk-free rate to daily compounding rate.

    Formula:
        r_{f, d} = (1 + r_{f, annual})^(1 / trading_days) - 1
    """
    if rf_annual <= -1.0:
        raise ValueError(f"Annual risk-free rate must be > -1.0, got {rf_annual}")
    return float((1.0 + rf_annual) ** (1.0 / trading_days) - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> float:
    """Compute annualized Sharpe ratio.

    Formula:
        Sharpe = (mean(r - r_{f,d}) / std(r - r_{f,d})) * sqrt(trading_days)

    Args:
        returns: Daily simple returns Series.
        rf_annual: Configured annual risk-free rate (e.g. 0.065 for 6.5%).
        trading_days: Annualization factor (default: 252).

    Returns:
        Annualized Sharpe ratio as float.
    """
    validate_returns(returns)
    if len(returns) < 2:
        return 0.0

    rf_daily = to_daily_risk_free(rf_annual, trading_days=trading_days)
    excess_rets = returns - rf_daily

    daily_mean = float(excess_rets.mean())
    daily_std = float(excess_rets.std(ddof=1))

    if daily_std < 1e-8:
        return 0.0

    return float((daily_mean / daily_std) * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> float:
    """Compute annualized Sortino ratio penalizing only downside volatility.

    Formula:
        downside_dev = sqrt(mean(min(0, r - r_{f,d})^2)) * sqrt(trading_days)
        Sortino = (mean(r - r_{f,d}) * trading_days) / downside_dev

    Args:
        returns: Daily simple returns Series.
        rf_annual: Configured annual risk-free rate.
        trading_days: Annualization factor (default: 252).

    Returns:
        Annualized Sortino ratio as float.
    """
    validate_returns(returns)
    if len(returns) < 2:
        return 0.0

    rf_daily = to_daily_risk_free(rf_annual, trading_days=trading_days)
    excess_rets = returns - rf_daily

    downside = np.minimum(0.0, excess_rets.to_numpy())
    downside_var = float(np.mean(downside**2))
    downside_std_ann = np.sqrt(downside_var * trading_days)

    if downside_std_ann < 1e-8:
        return 0.0

    annual_excess_mean = float(excess_rets.mean()) * trading_days
    return float(annual_excess_mean / downside_std_ann)


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Compute portfolio market beta vs benchmark.

    Formula:
        beta = Cov(r_p, r_m) / Var(r_m)

    Args:
        returns: Daily portfolio return series.
        benchmark: Daily benchmark return series.

    Returns:
        Beta coefficient as float.
    """
    validate_returns(returns)
    validate_returns(benchmark)

    # Align common dates
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0

    r_p = aligned.iloc[:, 0].to_numpy()
    r_m = aligned.iloc[:, 1].to_numpy()

    var_m = np.var(r_m, ddof=1)
    if var_m < 1e-8:
        return 0.0

    cov_pm = np.cov(r_p, r_m, ddof=1)[0, 1]
    return float(cov_pm / var_m)


def tracking_error(
    returns: pd.Series,
    benchmark: pd.Series,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> float:
    """Compute annualized tracking error vs benchmark.

    Formula:
        TE = std(r_p - r_b) * sqrt(trading_days)

    Args:
        returns: Daily portfolio return series.
        benchmark: Daily benchmark return series.
        trading_days: Annualization factor (default: 252).

    Returns:
        Annualized tracking error as float.
    """
    validate_returns(returns)
    validate_returns(benchmark)

    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0

    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.std(ddof=1) * np.sqrt(trading_days))


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> float:
    """Compute Information Ratio (active return / tracking error).

    Formula:
        IR = (mean(r_p - r_b) * trading_days) / TE

    Args:
        returns: Daily portfolio return series.
        benchmark: Daily benchmark return series.
        trading_days: Annualization factor (default: 252).

    Returns:
        Information ratio as float.
    """
    te = tracking_error(returns, benchmark, trading_days=trading_days)
    if te < 1e-8:
        return 0.0

    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    active_mean = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).mean())
    annual_active = active_mean * trading_days
    return float(annual_active / te)
