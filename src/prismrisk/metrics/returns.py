"""Return calculations and time compounding.

Enforces:
- Rule R-F3: Portfolio aggregation uses simple returns. Time aggregation
  and statistics that assume additivity use log returns.
- Rule R-F4: Annualisation via compounding (CAGR), not multiplying mean by 252.
- Rule R-F5: Returns are negative when losing.
"""

from typing import Union
import numpy as np
import pandas as pd
from prismrisk.utils.validation import (
    ValidationError,
    validate_prices,
    validate_returns,
    validate_weights,
)

DEFAULT_TRADING_DAYS: int = 252


def simple_returns(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Compute daily arithmetic simple percentage returns from price series.

    Formula:
        r_t = (P_t / P_{t-1}) - 1

    Args:
        prices: Asset price DataFrame [date x asset] or Series [date].

    Returns:
        Simple return Series or DataFrame with first NaN row dropped.
        Sign convention: negative for losses.
    """
    if isinstance(prices, pd.DataFrame):
        validate_prices(prices)
        rets = prices.pct_change().iloc[1:]
        validate_returns(rets)
        return rets
    elif isinstance(prices, pd.Series):
        if prices.empty:
            raise ValueError("Prices Series is empty.")
        rets_s = prices.pct_change().iloc[1:]
        validate_returns(rets_s)
        return rets_s
    raise TypeError(f"Expected pd.DataFrame or pd.Series, got {type(prices).__name__}")


def log_returns(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Compute continuously compounded logarithmic returns from price series.

    Formula:
        l_t = ln(P_t / P_{t-1})

    Args:
        prices: Asset price DataFrame [date x asset] or Series [date].

    Returns:
        Log return Series or DataFrame with first NaN row dropped.
        Sign convention: negative for losses.
    """
    if isinstance(prices, pd.DataFrame):
        validate_prices(prices)
        log_rets = np.log(prices / prices.shift(1)).iloc[1:]
        validate_returns(log_rets)
        return log_rets
    elif isinstance(prices, pd.Series):
        if prices.empty:
            raise ValueError("Prices Series is empty.")
        log_rets_s = np.log(prices / prices.shift(1)).iloc[1:]
        validate_returns(log_rets_s)
        return log_rets_s
    raise TypeError(f"Expected pd.DataFrame or pd.Series, got {type(prices).__name__}")


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative compound return series from daily simple returns.

    Formula:
        V_t = prod_{s=1}^t (1 + r_s) - 1

    Args:
        returns: Daily simple returns Series.

    Returns:
        Cumulative return Series indexed by date. Base level starts at 0.0.
        Sign convention: negative when losing.
    """
    validate_returns(returns)
    return (1.0 + returns).cumprod() - 1.0


def cagr(returns: pd.Series, trading_days: int = DEFAULT_TRADING_DAYS) -> float:
    """Compute Compound Annual Growth Rate (CAGR) from simple return series.

    Formula:
        CAGR = (prod_{t=1}^T (1 + r_t))^(trading_days / T) - 1

    Args:
        returns: Daily simple returns Series.
        trading_days: Number of trading days per year (default: 252).

    Returns:
        Annualized compound growth rate as float (e.g. 0.125 for 12.5%).
        Sign convention: negative when losing.
    """
    validate_returns(returns)
    n_obs = len(returns)
    if n_obs == 0:
        return 0.0

    terminal_wealth = float((1.0 + returns).prod())
    if terminal_wealth <= 0:
        return -1.0  # Total loss

    return float(terminal_wealth ** (trading_days / n_obs) - 1.0)


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute portfolio daily simple return series for static weights.

    Formula:
        r_{p, t} = sum_{i=1}^N w_i * r_{i, t}

    Args:
        asset_returns: DataFrame [date x asset] of daily simple returns.
        weights: Series [asset] summing to 1.0.

    Returns:
        Portfolio daily simple return Series.
    """
    missing = [a for a in weights.index if a not in asset_returns.columns]
    if missing:
        raise ValidationError(f"asset_returns missing assets declared in weights: {missing}")

    sub_rets = asset_returns[list(weights.index)]
    validate_returns(sub_rets)
    validate_weights(weights, expected_assets=list(sub_rets.columns))

    aligned_weights = weights.reindex(sub_rets.columns).to_numpy()
    port_rets = sub_rets.to_numpy() @ aligned_weights
    series = pd.Series(port_rets, index=sub_rets.index, name="portfolio")
    validate_returns(series)
    return series
