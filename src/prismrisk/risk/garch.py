"""GARCH(1,1) volatility forecasting and Filtered Historical Simulation (FHS).

Implements FR-R5 (Stretch Requirement):
- Maximum Likelihood Estimation (MLE) of GARCH(1,1) parameters (omega, alpha, beta).
- 1-step-ahead conditional variance forecast.
- Filtered Historical Simulation (FHS) VaR/CVaR.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from prismrisk.models import VarResult
from prismrisk.risk.var import historical_cvar, historical_var
from prismrisk.utils.validation import validate_confidence, validate_returns


@dataclass(frozen=True)
class GarchFitResult:
    """Estimated GARCH(1,1) parameter estimates and diagnostics."""
    mu: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_vol_ann: float
    last_conditional_vol_ann: float
    forecast_vol_ann: float
    standardized_residuals: pd.Series
    conditional_volatility: pd.Series


def fit_garch11(returns: pd.Series, trading_days: int = 252) -> GarchFitResult:
    """Fit GARCH(1,1) model to daily return series via Maximum Likelihood.

    Process:
        Variance equation: sigma_t^2 = omega + alpha * (r_{t-1} - mu)^2 + beta * sigma_{t-1}^2
        Stationarity condition: alpha + beta < 1.0

    Args:
        returns: Daily simple returns Series.
        trading_days: Annualization scaling factor (default: 252).

    Returns:
        GarchFitResult containing estimated parameters, residuals, and forecasts.
    """
    validate_returns(returns)
    r = returns.to_numpy()
    n = len(r)

    mu_sample = float(np.mean(r))
    var_sample = float(np.var(r, ddof=1))

    # Objective: Gaussian negative log-likelihood
    def neg_log_likelihood(params: np.ndarray) -> float:
        mu, omega, alpha, beta = params
        eps = r - mu
        sig2 = np.empty(n)
        sig2[0] = var_sample

        for t in range(1, n):
            sig2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sig2[t - 1]

        # Prevent numerical zero or negative variance
        sig2 = np.maximum(sig2, 1e-10)
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sig2) + (eps**2) / sig2)
        return float(-ll)

    # Initial guesses: typical financial time series
    # omega = var_sample * (1 - alpha - beta)
    init_alpha, init_beta = 0.08, 0.88
    init_omega = var_sample * (1.0 - init_alpha - init_beta)
    init_params = np.array([mu_sample, max(1e-6, init_omega), init_alpha, init_beta])

    bounds = [
        (None, None),          # mu
        (1e-9, None),          # omega > 0
        (0.001, 0.40),         # alpha in (0, 0.4)
        (0.50, 0.98),          # beta in (0.5, 0.98)
    ]
    # Constraint: alpha + beta <= 0.999
    cons = {"type": "ineq", "fun": lambda p: 0.999 - (p[2] + p[3])}

    opt = minimize(
        neg_log_likelihood,
        init_params,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    mu_opt, omega_opt, alpha_opt, beta_opt = opt.x if opt.success else init_params
    persistence = alpha_opt + beta_opt

    # Reconstruct conditional variance trajectory
    eps_opt = r - mu_opt
    sig2_traj = np.empty(n)
    sig2_traj[0] = var_sample
    for t in range(1, n):
        sig2_traj[t] = omega_opt + alpha_opt * (eps_opt[t - 1] ** 2) + beta_opt * sig2_traj[t - 1]

    cond_vol = np.sqrt(sig2_traj)
    z = eps_opt / cond_vol

    # 1-step-ahead forecast for T+1
    forecast_sig2 = omega_opt + alpha_opt * (eps_opt[-1] ** 2) + beta_opt * sig2_traj[-1]
    forecast_vol_ann = float(np.sqrt(forecast_sig2 * trading_days))
    last_vol_ann = float(cond_vol[-1] * np.sqrt(trading_days))
    uncond_sig2 = omega_opt / max(1e-4, (1.0 - persistence))
    uncond_vol_ann = float(np.sqrt(uncond_sig2 * trading_days))

    return GarchFitResult(
        mu=float(mu_opt),
        omega=float(omega_opt),
        alpha=float(alpha_opt),
        beta=float(beta_opt),
        persistence=float(persistence),
        unconditional_vol_ann=uncond_vol_ann,
        last_conditional_vol_ann=last_vol_ann,
        forecast_vol_ann=forecast_vol_ann,
        standardized_residuals=pd.Series(z, index=returns.index, name="garch_z"),
        conditional_volatility=pd.Series(
            cond_vol * np.sqrt(trading_days), index=returns.index, name="garch_vol_ann"
        ),
    )


def filtered_historical_simulation_var(
    returns: pd.Series,
    confidence: float = 0.95,
    trading_days: int = 252,
) -> VarResult:
    """Compute Filtered Historical Simulation (FHS) VaR and CVaR.

    Scales historical standardized residuals by the GARCH(1,1) next-day
    conditional volatility forecast.

    Args:
        returns: Daily simple returns Series.
        confidence: Confidence level in (0, 1).
        trading_days: Annualization scaling factor.

    Returns:
        VarResult containing FHS VaR, CVaR, and simulated scenarios.
    """
    validate_confidence(confidence)
    fit = fit_garch11(returns, trading_days=trading_days)
    daily_forecast_sigma = fit.forecast_vol_ann / np.sqrt(trading_days)

    # FHS synthetic scenarios: r* = mu + sigma_{T+1} * z
    z = fit.standardized_residuals.to_numpy()
    fhs_scenarios = fit.mu + daily_forecast_sigma * z
    fhs_series = pd.Series(fhs_scenarios, name="fhs_returns")

    var_val = historical_var(fhs_series, confidence=confidence)
    cvar_val = historical_cvar(fhs_series, confidence=confidence)

    return VarResult(
        method="filtered_historical_simulation",
        confidence=confidence,
        var=var_val,
        cvar=cvar_val,
        simulated_returns=fhs_series,
    )
