"""Unit tests for GARCH(1,1) volatility fitting and Filtered Historical Simulation (FHS)."""

import numpy as np
import pandas as pd
import pytest
from prismrisk.risk.garch import filtered_historical_simulation_var, fit_garch11


@pytest.fixture
def garch_synthetic_series() -> pd.Series:
    """Simulate a GARCH(1,1) process with volatility clustering."""
    rng = np.random.default_rng(42)
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    omega = 0.00001
    alpha = 0.10
    beta = 0.85

    eps = np.zeros(n)
    sig2 = np.zeros(n)
    sig2[0] = omega / (1.0 - alpha - beta)
    eps[0] = rng.normal(0, np.sqrt(sig2[0]))

    for t in range(1, n):
        sig2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sig2[t - 1]
        eps[t] = rng.normal(0, np.sqrt(sig2[t]))

    return pd.Series(eps, index=dates, name="garch_synthetic")


def test_fit_garch11_parameters(garch_synthetic_series: pd.Series) -> None:
    """Test GARCH(1,1) MLE parameter estimation."""
    res = fit_garch11(garch_synthetic_series)
    assert res.alpha > 0.0
    assert res.beta > 0.5
    assert res.persistence < 1.0  # Stationarity constraint
    assert res.forecast_vol_ann > 0.0
    assert len(res.standardized_residuals) == len(garch_synthetic_series)
    assert len(res.conditional_volatility) == len(garch_synthetic_series)


def test_filtered_historical_simulation_invariants(garch_synthetic_series: pd.Series) -> None:
    """Test FHS VaR/CVaR ordering invariants."""
    fhs_95 = filtered_historical_simulation_var(garch_synthetic_series, confidence=0.95)
    fhs_99 = filtered_historical_simulation_var(garch_synthetic_series, confidence=0.99)

    assert fhs_99.var >= fhs_95.var
    assert fhs_95.cvar >= fhs_95.var
    assert fhs_99.cvar >= fhs_99.var
    assert fhs_95.method == "filtered_historical_simulation"
