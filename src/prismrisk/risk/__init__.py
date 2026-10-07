"""Tail risk and Value-at-Risk (VaR / CVaR) analytics.

Enforces:
- Rule R-C1: Pure functions on Pandas/NumPy objects. No I/O, plotting, or UI imports.
- Rule R-F5: VaR and CVaR are reported as positive loss numbers.
"""

from prismrisk.risk.backtest import (
    backtest_var_model,
    count_breaches,
    kupiec_pof,
    rolling_var,
)
from prismrisk.risk.var import (
    compare_var_models,
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
)

__all__ = [
    "historical_var",
    "historical_cvar",
    "parametric_var",
    "parametric_cvar",
    "monte_carlo_var",
    "compare_var_models",
    "rolling_var",
    "count_breaches",
    "kupiec_pof",
    "backtest_var_model",
]
