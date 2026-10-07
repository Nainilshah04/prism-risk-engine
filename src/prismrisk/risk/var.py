"""Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall) models.

Enforces:
- Rule R-F5: VaR and CVaR are reported as positive loss fractions.
- Rule R-T2: Invariants: VaR(99%) >= VaR(95%) and CVaR >= VaR.
- Rule R-C7: Deterministic Monte Carlo simulation via explicit random seed.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

from prismrisk.models import VarResult
from prismrisk.utils.validation import (
    validate_confidence,
    validate_returns,
    validate_weights,
)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute empirical Historical Value at Risk (VaR).

    Formula:
        VaR_alpha = -quantile(r, 1 - alpha)

    Args:
        returns: Daily simple returns Series (time series or simulated distribution).
        confidence: Confidence level in (0, 1), e.g. 0.95 or 0.99.

    Returns:
        Positive loss fraction as float (e.g. 0.021 = 2.1% loss).
    """
    validate_returns(returns, require_datetime=False)
    validate_confidence(confidence)
    if returns.empty:
        return 0.0

    cutoff = float(np.percentile(returns.to_numpy(), (1.0 - confidence) * 100.0))
    # Report as positive loss fraction
    return max(0.0, -cutoff)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute empirical Historical Conditional Value at Risk (CVaR / Expected Shortfall).

    Formula:
        CVaR_alpha = -E[r | r <= -VaR_alpha]

    Args:
        returns: Daily simple returns Series.
        confidence: Confidence level in (0, 1).

    Returns:
        Positive loss fraction as float. Guaranteed >= historical_var.
    """
    validate_returns(returns, require_datetime=False)
    validate_confidence(confidence)
    if returns.empty:
        return 0.0

    var_threshold = -historical_var(returns, confidence=confidence)
    tail_losses = returns[returns <= var_threshold]
    if tail_losses.empty:
        return historical_var(returns, confidence=confidence)

    cvar_val = float(-tail_losses.mean())
    return max(historical_var(returns, confidence=confidence), cvar_val)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute Parametric (Gaussian / Normal) Value at Risk.

    Formula:
        VaR_alpha = -(mu + z_{1-alpha} * sigma) = z_alpha * sigma - mu

    Args:
        returns: Daily simple returns Series.
        confidence: Confidence level in (0, 1).

    Returns:
        Positive loss fraction as float.
    """
    validate_returns(returns)
    validate_confidence(confidence)
    if len(returns) < 2:
        return 0.0

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = norm.ppf(confidence)
    var_val = z * sigma - mu
    return max(0.0, float(var_val))


def parametric_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute Parametric (Gaussian) Conditional Value at Risk (Expected Shortfall).

    Formula:
        CVaR_alpha = -mu + sigma * (phi(z_alpha) / (1 - alpha))

    Args:
        returns: Daily simple returns Series.
        confidence: Confidence level in (0, 1).

    Returns:
        Positive loss fraction as float.
    """
    validate_returns(returns)
    validate_confidence(confidence)
    if len(returns) < 2:
        return 0.0

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = norm.ppf(confidence)
    pdf_z = norm.pdf(z)
    cvar_val = -mu + sigma * (pdf_z / (1.0 - confidence))
    return max(parametric_var(returns, confidence), float(cvar_val))


def monte_carlo_var(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    n_paths: int = 10000,
    seed: int = 42,
) -> VarResult:
    """Estimate portfolio VaR and CVaR using correlated Monte Carlo simulation.

    Process:
        1. Decomposes covariance matrix Sigma = L * L^T via Cholesky.
        2. Simulates N correlated multivariate normal asset returns ~ MVN(mu, Sigma).
        3. Computes portfolio returns R_p = sum(w_i * r_i).
        4. Extracts empirical VaR and CVaR from the simulated distribution.

    Args:
        mu: Asset mean daily returns Series.
        cov: Asset covariance matrix DataFrame.
        weights: Portfolio asset weights Series.
        confidence: Confidence level in (0, 1).
        n_paths: Number of simulation iterations (default: 10,000).
        seed: Random seed for deterministic reproducibility (Rule R-C7).

    Returns:
        VarResult containing VaR, CVaR, and simulated portfolio path series.
    """
    validate_confidence(confidence)
    validate_weights(weights, expected_assets=list(mu.index))

    assets = list(mu.index)
    aligned_mu = mu.loc[assets].to_numpy()
    aligned_cov = cov.loc[assets, assets].to_numpy()
    aligned_w = weights.loc[assets].to_numpy()

    # Cholesky decomposition
    chol = np.linalg.cholesky(aligned_cov)

    rng = np.random.default_rng(seed)
    z_normals = rng.standard_normal(size=(n_paths, len(assets)))
    simulated_asset_rets = aligned_mu + (z_normals @ chol.T)

    simulated_port_rets = simulated_asset_rets @ aligned_w
    port_series = pd.Series(simulated_port_rets, name="mc_returns")

    var_val = historical_var(port_series, confidence=confidence)
    cvar_val = historical_cvar(port_series, confidence=confidence)

    return VarResult(
        method="monte_carlo",
        confidence=confidence,
        var=var_val,
        cvar=cvar_val,
        simulated_returns=port_series,
    )


def compare_var_models(
    returns: pd.Series,
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None,
    weights: Optional[pd.Series] = None,
    confidences: Optional[List[float]] = None,
    n_paths: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a structured comparative table of VaR and CVaR across models.

    Args:
        returns: Portfolio daily returns Series.
        mu: Optional asset drift Series for Monte Carlo.
        cov: Optional asset covariance DataFrame for Monte Carlo.
        weights: Optional portfolio weights for Monte Carlo.
        confidences: List of confidence levels (default: [0.95, 0.99]).
        n_paths: Monte Carlo paths.
        seed: Random seed.

    Returns:
        pd.DataFrame comparing Historical, Parametric, and Monte Carlo VaR & CVaR.
    """
    if confidences is None:
        confidences = [0.95, 0.99]

    rows = []
    for conf in sorted(confidences):
        h_var = historical_var(returns, confidence=conf)
        h_cvar = historical_cvar(returns, confidence=conf)
        p_var = parametric_var(returns, confidence=conf)
        p_cvar = parametric_cvar(returns, confidence=conf)

        row_dict = {
            "Confidence": f"{conf:.0%}",
            "Historical VaR": h_var,
            "Historical CVaR": h_cvar,
            "Parametric VaR": p_var,
            "Parametric CVaR": p_cvar,
        }

        if mu is not None and cov is not None and weights is not None:
            mc_res = monte_carlo_var(
                mu=mu,
                cov=cov,
                weights=weights,
                confidence=conf,
                n_paths=n_paths,
                seed=seed,
            )
            row_dict["Monte Carlo VaR"] = mc_res.var
            row_dict["Monte Carlo CVaR"] = mc_res.cvar

        rows.append(row_dict)

    df = pd.DataFrame(rows).set_index("Confidence")
    return df
