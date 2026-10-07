"""Multi-asset portfolio optimizers.

Implements FR-P1:
- Equal Weight
- Inverse Volatility
- Minimum Variance (long-only, with caps)
- Risk Parity / Equal Risk Contribution (ERC)
- Maximum Sharpe (long-only, with caps)
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from prismrisk.utils.validation import validate_returns, validate_weights


def equal_weight(assets: List[str]) -> pd.Series:
    """Generate 1/N equal weight portfolio vector.

    Args:
        assets: List of asset identifier strings.

    Returns:
        pd.Series of weights summing to 1.0.
    """
    n = len(assets)
    if n == 0:
        raise ValueError("Asset list cannot be empty.")
    w = np.full(n, 1.0 / n)
    return pd.Series(w, index=assets, name="equal_weight")


def inverse_volatility(
    returns: pd.DataFrame,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> pd.Series:
    """Generate inverse volatility weighted portfolio vector.

    Weights are proportional to 1 / sigma_i.

    Args:
        returns: DataFrame [date x asset] of daily simple returns.
        max_weight: Maximum weight cap per asset.
        min_weight: Minimum weight floor per asset.

    Returns:
        pd.Series of weights summing to 1.0.
    """
    validate_returns(returns)
    assets = list(returns.columns)
    vols = returns.std(ddof=1).to_numpy()

    # Prevent division by zero
    vols = np.maximum(vols, 1e-8)
    inv_v = 1.0 / vols
    target_w = inv_v / np.sum(inv_v)

    # If within bounds, return directly
    if (target_w <= max_weight + 1e-5).all() and (target_w >= min_weight - 1e-5).all():
        return pd.Series(target_w, index=assets, name="inverse_vol")

    # Otherwise project onto simplex with bounds via quadratic minimization
    def obj(w: np.ndarray) -> float:
        return float(np.sum((w - target_w) ** 2))

    bounds = [(min_weight, max_weight) for _ in assets]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.full(len(assets), 1.0 / len(assets))

    res = minimize(obj, init_w, method="SLSQP", bounds=bounds, constraints=cons)
    w_opt = res.x if res.success else target_w
    w_opt = np.clip(w_opt, min_weight, max_weight)
    w_opt = w_opt / np.sum(w_opt)
    return pd.Series(w_opt, index=assets, name="inverse_vol")


def minimum_variance(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> pd.Series:
    """Compute long-only Minimum Variance portfolio.

    Objective:
        min_w  w^T * Sigma * w
        s.t.   sum(w) = 1, min_w <= w_i <= max_w

    Args:
        cov: Asset covariance DataFrame [asset x asset].
        max_weight: Upper bound weight cap.
        min_weight: Lower bound floor.

    Returns:
        pd.Series of optimal weights summing to 1.0.
    """
    assets = list(cov.columns)
    sigma_mat = cov.to_numpy()
    n = len(assets)

    def obj(w: np.ndarray) -> float:
        return float(w @ sigma_mat @ w)

    bounds = [(min_weight, max_weight) for _ in range(n)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.full(n, 1.0 / n)

    res = minimize(obj, init_w, method="SLSQP", bounds=bounds, constraints=cons)
    w_opt = res.x if res.success else init_w
    w_opt = np.clip(w_opt, min_weight, max_weight)
    w_opt = w_opt / np.sum(w_opt)

    s = pd.Series(w_opt, index=assets, name="min_variance")
    validate_weights(s, expected_assets=assets)
    return s


def risk_parity(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> pd.Series:
    """Compute Equal Risk Contribution (ERC) Risk Parity portfolio.

    Objective:
        min_w  sum_{i=1}^N ( (w_i * (Sigma * w)_i) / (w^T * Sigma * w) - 1/N )^2
        s.t.   sum(w) = 1, min_w <= w_i <= max_w

    Args:
        cov: Asset covariance DataFrame [asset x asset].
        max_weight: Upper bound weight cap.
        min_weight: Lower bound floor.

    Returns:
        pd.Series of optimal risk parity weights summing to 1.0.
    """
    assets = list(cov.columns)
    sigma_mat = cov.to_numpy()
    n = len(assets)
    target_rc_pct = 1.0 / n

    def obj(w: np.ndarray) -> float:
        port_var = float(w @ sigma_mat @ w)
        if port_var < 1e-12:
            return 1e6
        # Percentage risk contribution per asset: w_i * (Sigma w)_i / port_var
        mrc = sigma_mat @ w
        pct_rc = (w * mrc) / port_var
        return float(np.sum((pct_rc - target_rc_pct) ** 2))

    bounds = [(min_weight, max_weight) for _ in range(n)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.full(n, 1.0 / n)

    res = minimize(
        obj,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    w_opt = res.x if res.success else init_w
    w_opt = np.clip(w_opt, min_weight, max_weight)
    w_opt = w_opt / np.sum(w_opt)

    s = pd.Series(w_opt, index=assets, name="risk_parity")
    validate_weights(s, expected_assets=assets)
    return s


def max_sharpe(
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    rf_annual: float = 0.065,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    trading_days: int = 252,
) -> pd.Series:
    """Compute long-only Maximum Sharpe Ratio portfolio.

    Objective:
        max_w  (w^T * mu_ann - rf_ann) / sqrt(w^T * Sigma_ann * w)

    Note: Max Sharpe is sensitive to sample estimation error in mu,
    often resulting in concentrated corner solutions.

    Args:
        returns: Historical return DataFrame [date x asset].
        cov: Asset covariance DataFrame [asset x asset].
        rf_annual: Annual risk-free rate hurdle.
        max_weight: Upper bound weight cap.
        min_weight: Lower bound floor.
        trading_days: Annualization factor (252).

    Returns:
        pd.Series of optimal max Sharpe weights summing to 1.0.
    """
    validate_returns(returns)
    assets = list(cov.columns)
    n = len(assets)

    mu_ann = returns[assets].mean().to_numpy() * trading_days
    cov_ann = cov.loc[assets, assets].to_numpy() * trading_days

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu_ann)
        port_vol = np.sqrt(max(1e-12, float(w @ cov_ann @ w)))
        sr = (port_ret - rf_annual) / port_vol
        return -float(sr)

    bounds = [(min_weight, max_weight) for _ in range(n)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.full(n, 1.0 / n)

    res = minimize(
        neg_sharpe,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    w_opt = res.x if res.success else init_w
    w_opt = np.clip(w_opt, min_weight, max_weight)
    w_opt = w_opt / np.sum(w_opt)

    s = pd.Series(w_opt, index=assets, name="max_sharpe")
    validate_weights(s, expected_assets=assets)
    return s


def optimise(
    strategy: str,
    returns_window: pd.DataFrame,
    cov: pd.DataFrame,
    max_weight: float = 0.70,
    min_weight: float = 0.0,
    rf_annual: float = 0.065,
) -> pd.Series:
    """Unified optimizer dispatcher for portfolio strategies.

    Args:
        strategy: Name of strategy ('equal_weight', 'inverse_vol',
                  'min_variance', 'risk_parity', 'max_sharpe').
        returns_window: Return matrix in the current lookback slice.
        cov: Estimated covariance matrix.
        max_weight: Weight cap constraint.
        min_weight: Weight floor constraint.
        rf_annual: Annual risk-free rate.

    Returns:
        pd.Series of target portfolio weights.
    """
    assets = list(returns_window.columns)
    strat = strategy.lower().replace(" ", "_").replace("-", "_")

    if strat in ("equal_weight", "ew"):
        return equal_weight(assets)
    elif strat in ("inverse_vol", "inverse_volatility", "inv_vol"):
        return inverse_volatility(returns_window, max_weight=max_weight, min_weight=min_weight)
    elif strat in ("min_variance", "minimum_variance", "min_var"):
        return minimum_variance(cov, max_weight=max_weight, min_weight=min_weight)
    elif strat in ("risk_parity", "erc"):
        return risk_parity(cov, max_weight=max_weight, min_weight=min_weight)
    elif strat in ("max_sharpe", "maximum_sharpe"):
        return max_sharpe(
            returns_window,
            cov,
            rf_annual=rf_annual,
            max_weight=max_weight,
            min_weight=min_weight,
        )
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Supported: equal_weight, inverse_vol, "
            f"min_variance, risk_parity, max_sharpe"
        )
