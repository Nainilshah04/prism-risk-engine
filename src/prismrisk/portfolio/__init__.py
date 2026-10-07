"""Portfolio optimization, covariance estimation, and walk-forward backtesting.

Enforces:
- Rule R-C1: Pure core without I/O or Streamlit dependencies.
- Rule R-F2: Zero lookahead bias in walk-forward backtesting.
- Rule R-F9: Transaction costs and weight drift.
"""

from prismrisk.portfolio.backtest import (
    compare_strategies,
    identify_rebalance_dates,
    run_backtest,
)
from prismrisk.portfolio.contribution import risk_contributions
from prismrisk.portfolio.covariance import estimate_cov
from prismrisk.portfolio.optimisers import (
    equal_weight,
    inverse_volatility,
    max_sharpe,
    minimum_variance,
    optimise,
    risk_parity,
)

__all__ = [
    "estimate_cov",
    "equal_weight",
    "inverse_volatility",
    "minimum_variance",
    "risk_parity",
    "max_sharpe",
    "optimise",
    "risk_contributions",
    "run_backtest",
    "compare_strategies",
    "identify_rebalance_dates",
]
