"""Core investment returns, volatility, drawdown, and ratio metrics.

Enforces Rule R-C1: Pure functions on Pandas/NumPy objects. No I/O, plotting, or UI imports.
"""

from prismrisk.metrics.correlation import correlation_matrix, rolling_correlation
from prismrisk.metrics.drawdown import calmar_ratio, drawdown_series, max_drawdown
from prismrisk.metrics.ratios import (
    beta,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from prismrisk.metrics.returns import (
    cagr,
    cumulative_returns,
    log_returns,
    portfolio_returns,
    simple_returns,
)
from prismrisk.metrics.volatility import annualised_vol, ewma_vol, rolling_vol

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "cagr",
    "portfolio_returns",
    "annualised_vol",
    "rolling_vol",
    "ewma_vol",
    "drawdown_series",
    "max_drawdown",
    "calmar_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "beta",
    "tracking_error",
    "information_ratio",
    "correlation_matrix",
    "rolling_correlation",
]
