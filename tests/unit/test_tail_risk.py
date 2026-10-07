"""Unit tests for tail risk models, VaR/CVaR invariants, rolling backtests, and Kupiec test.

Enforces:
- Rule R-T2: VaR(99%) >= VaR(95%) and CVaR >= VaR.
- Rule R-C7: Seeded Monte Carlo reproducibility.
- Rule R-T4: Leakage protection.
"""

import numpy as np
import pandas as pd
import pytest
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


@pytest.fixture
def sample_returns() -> pd.Series:
    """Deterministic return series with known negative skew and fat tails."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    # Mixture of normal and occasional sharp drops
    rets = rng.normal(0.0005, 0.012, 1000)
    # Inject 15 negative shocks into left tail
    crash_idx = rng.choice(1000, size=15, replace=False)
    rets[crash_idx] -= 0.04
    return pd.Series(rets, index=dates, name="sample_portfolio")


def test_var_ordering_invariants(sample_returns: pd.Series) -> None:
    """Invariant: VaR(99%) >= VaR(95%) across all models."""
    # Historical
    h_95 = historical_var(sample_returns, confidence=0.95)
    h_99 = historical_var(sample_returns, confidence=0.99)
    assert h_99 >= h_95
    assert h_95 > 0.0

    # Parametric
    p_95 = parametric_var(sample_returns, confidence=0.95)
    p_99 = parametric_var(sample_returns, confidence=0.99)
    assert p_99 >= p_95
    assert p_95 > 0.0


def test_cvar_greater_than_var_invariants(sample_returns: pd.Series) -> None:
    """Invariant: CVaR >= VaR at all confidence levels."""
    for conf in [0.95, 0.99]:
        h_var = historical_var(sample_returns, confidence=conf)
        h_cvar = historical_cvar(sample_returns, confidence=conf)
        assert h_cvar >= h_var

        p_var = parametric_var(sample_returns, confidence=conf)
        p_cvar = parametric_cvar(sample_returns, confidence=conf)
        assert p_cvar >= p_var


def test_monte_carlo_seed_reproducibility() -> None:
    """Verify Monte Carlo is 100% deterministic with fixed seed (Rule R-C7)."""
    mu = pd.Series({"asset1": 0.0005, "asset2": 0.0003})
    cov = pd.DataFrame(
        {
            "asset1": [0.0002, 0.00005],
            "asset2": [0.00005, 0.00015],
        },
        index=["asset1", "asset2"],
    )
    weights = pd.Series({"asset1": 0.60, "asset2": 0.40})

    run1 = monte_carlo_var(mu, cov, weights, confidence=0.95, n_paths=5000, seed=42)
    run2 = monte_carlo_var(mu, cov, weights, confidence=0.95, n_paths=5000, seed=42)
    assert run1.var == run2.var
    assert run1.cvar == run2.cvar

    # Different seed produces different draws
    run3 = monte_carlo_var(mu, cov, weights, confidence=0.95, n_paths=5000, seed=123)
    assert run1.var != run3.var


def test_kupiec_known_answers() -> None:
    """Hand-computed Kupiec POF test checks.

    For N = 1000, confidence = 0.95 (expected breaches = 50):
    - When breaches = 50: LR stat = 0.0, p-value = 1.0 -> PASS.
    - When breaches = 85: LR stat is large, p-value < 0.05 -> REJECT.
    """
    perfect_res = kupiec_pof(breaches=50, n_obs=1000, confidence=0.95)
    assert perfect_res.expected_breaches == 50.0
    assert perfect_res.breach_rate == 0.05
    assert pytest.approx(perfect_res.lr_stat, 1e-4) == 0.0
    assert pytest.approx(perfect_res.p_value, 1e-4) == 1.0
    assert not perfect_res.reject_null

    failing_res = kupiec_pof(breaches=85, n_obs=1000, confidence=0.95)
    assert failing_res.lr_stat > 3.841  # 95% critical value of chi2(1)
    assert failing_res.p_value < 0.05
    assert failing_res.reject_null


def test_kupiec_boundary_zero_breaches() -> None:
    """Ensure zero breaches does not cause log(0) crash."""
    zero_res = kupiec_pof(breaches=0, n_obs=500, confidence=0.99)
    assert zero_res.breaches == 0
    assert zero_res.lr_stat > 0.0


def test_rolling_var_lookahead_leakage(sample_returns: pd.Series) -> None:
    """Rule R-T4 Leakage Test: Perturbing future returns must not alter past rolling VaR."""
    window = 250
    roll_baseline = rolling_var(sample_returns, window=window, confidence=0.95)

    # Copy and perturb data after index 600
    perturbed_returns = sample_returns.copy()
    perturbed_returns.iloc[600:] += 0.05

    roll_perturbed = rolling_var(perturbed_returns, window=window, confidence=0.95)

    # Values strictly before index 600 must be identical
    eval_dates_before = sample_returns.index[window:600]
    pd.testing.assert_series_equal(
        roll_baseline.loc[eval_dates_before],
        roll_perturbed.loc[eval_dates_before],
    )


def test_backtest_var_model_summary(sample_returns: pd.Series) -> None:
    """Test full backtesting pipeline across historical and parametric models."""
    summary_df, results_map = backtest_var_model(
        sample_returns,
        window=250,
        confidences=[0.95, 0.99],
        methods=["historical", "parametric"],
    )
    assert len(summary_df) == 4
    assert "historical_95" in results_map
    assert "parametric_99" in results_map
    assert "Kupiec Decision" in summary_df.columns


def test_compare_var_models_table(sample_returns: pd.Series) -> None:
    """Test comparative table generation."""
    table = compare_var_models(sample_returns, confidences=[0.95, 0.99])
    assert "Historical VaR" in table.columns
    assert "Parametric VaR" in table.columns
    assert len(table) == 2
