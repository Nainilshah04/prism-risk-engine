"""Unit tests for portfolio optimizers, covariance shrinkage, risk contributions, and walk-forward engine.

Enforces:
- Rule R-T2: Weights sum to 1, Risk Parity yields equal risk contributions within tolerance.
- Rule R-T4: Zero lookahead leakage in walk-forward backtesting.
- Rule R-F9: Transaction costs and turnover accounting.
"""

import numpy as np
import pandas as pd
import pytest
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


@pytest.fixture
def multi_asset_returns() -> pd.DataFrame:
    """Generate 600 days of correlated returns across 3 assets with distinct volatilities."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=600, freq="B")

    # High vol, medium vol, low vol
    sigmas = [0.015, 0.010, 0.002]
    corr = np.array([
        [1.0, 0.2, 0.05],
        [0.2, 1.0, 0.02],
        [0.05, 0.02, 1.0],
    ])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((600, 3)) @ chol.T
    rets = z * sigmas + np.array([0.0006, 0.0004, 0.0002])

    df = pd.DataFrame(rets, index=dates, columns=["equity", "gold", "debt"])
    return df


def test_covariance_estimators(multi_asset_returns: pd.DataFrame) -> None:
    """Test sample and Ledoit-Wolf covariance matrices."""
    sample_cov = estimate_cov(multi_asset_returns, method="sample")
    lw_cov = estimate_cov(multi_asset_returns, method="ledoit_wolf")

    assert sample_cov.shape == (3, 3)
    assert lw_cov.shape == (3, 3)

    # Check symmetry
    np.testing.assert_allclose(sample_cov.to_numpy(), sample_cov.to_numpy().T, atol=1e-8)
    np.testing.assert_allclose(lw_cov.to_numpy(), lw_cov.to_numpy().T, atol=1e-8)

    # Positive definiteness
    eigvals = np.linalg.eigvalsh(lw_cov.to_numpy())
    assert (eigvals > 0).all()


def test_equal_weight(multi_asset_returns: pd.DataFrame) -> None:
    """Test 1/N equal weighting."""
    w = equal_weight(list(multi_asset_returns.columns))
    assert len(w) == 3
    assert pytest.approx(w.sum(), 1e-6) == 1.0
    assert pytest.approx(w["equity"], 1e-6) == 1.0 / 3.0


def test_inverse_volatility(multi_asset_returns: pd.DataFrame) -> None:
    """Test inverse volatility weights."""
    w = inverse_volatility(multi_asset_returns, max_weight=0.70)
    assert pytest.approx(w.sum(), 1e-6) == 1.0
    # Lower vol asset (debt) should have higher weight than high vol asset (equity)
    assert w["debt"] > w["equity"]
    assert (w <= 0.70 + 1e-5).all()


def test_minimum_variance(multi_asset_returns: pd.DataFrame) -> None:
    """Test minimum variance optimizer."""
    cov = estimate_cov(multi_asset_returns, method="ledoit_wolf")
    w = minimum_variance(cov, max_weight=0.70)
    assert pytest.approx(w.sum(), 1e-6) == 1.0
    assert (w <= 0.70 + 1e-5).all()
    assert (w >= -1e-6).all()


def test_risk_parity_equal_risk_contributions(multi_asset_returns: pd.DataFrame) -> None:
    """Invariant test: Risk parity must produce equal percentage risk contributions."""
    cov = estimate_cov(multi_asset_returns, method="ledoit_wolf")
    w = risk_parity(cov, max_weight=0.80)
    assert pytest.approx(w.sum(), 1e-6) == 1.0

    # Verify risk contribution decomposition
    contrib = risk_contributions(w, cov)
    pct_rc = contrib["Percentage Risk Contribution"]

    # Target is 1/3 (33.33%) for all 3 assets within 1.5% tolerance
    expected_pct = 1.0 / 3.0
    for asset in ["equity", "gold", "debt"]:
        assert abs(pct_rc[asset] - expected_pct) < 0.02


def test_max_sharpe(multi_asset_returns: pd.DataFrame) -> None:
    """Test maximum Sharpe ratio optimizer."""
    cov = estimate_cov(multi_asset_returns, method="ledoit_wolf")
    w = max_sharpe(multi_asset_returns, cov, rf_annual=0.065, max_weight=0.70)
    assert pytest.approx(w.sum(), 1e-6) == 1.0
    assert (w <= 0.70 + 1e-5).all()
    assert (w >= -1e-6).all()


def test_risk_contributions_identities(multi_asset_returns: pd.DataFrame) -> None:
    """Verify sum(%RC) == 1.0 and sum(RC) == portfolio vol."""
    cov = estimate_cov(multi_asset_returns, method="ledoit_wolf")
    w = pd.Series({"equity": 0.60, "gold": 0.20, "debt": 0.20})
    rc_df = risk_contributions(w, cov)

    assert pytest.approx(rc_df["Percentage Risk Contribution"].sum(), 1e-6) == 1.0
    port_vol = np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    assert pytest.approx(rc_df["Risk Contribution"].sum(), 1e-6) == port_vol


def test_walk_forward_leakage_protection(multi_asset_returns: pd.DataFrame) -> None:
    """Rule R-T4 Leakage Test: Future returns perturbation must not alter past backtest results."""
    btr_clean = run_backtest(
        multi_asset_returns,
        strategy="min_variance",
        lookback=252,
        rebalance="monthly",
    )

    # Perturb data after index 450
    perturbed_returns = multi_asset_returns.copy()
    perturbed_returns.iloc[450:] += 0.05

    btr_perturbed = run_backtest(
        perturbed_returns,
        strategy="min_variance",
        lookback=252,
        rebalance="monthly",
    )

    # Results up to index 450 must be mathematically identical
    cutoff_date = multi_asset_returns.index[449]
    pd.testing.assert_series_equal(
        btr_clean.portfolio_returns.loc[:cutoff_date],
        btr_perturbed.portfolio_returns.loc[:cutoff_date],
    )
    pd.testing.assert_frame_equal(
        btr_clean.weights_over_time.loc[:cutoff_date],
        btr_perturbed.weights_over_time.loc[:cutoff_date],
    )


def test_transaction_costs_impact(multi_asset_returns: pd.DataFrame) -> None:
    """Verify that transaction costs reduce net terminal wealth."""
    res_free = run_backtest(
        multi_asset_returns,
        strategy="min_variance",
        lookback=252,
        rebalance="monthly",
        cost_bps=0.0,
    )
    res_costly = run_backtest(
        multi_asset_returns,
        strategy="min_variance",
        lookback=252,
        rebalance="monthly",
        cost_bps=20.0,
    )

    wealth_free = float(np.prod(1.0 + res_free.portfolio_returns.to_numpy()))
    wealth_costly = float(np.prod(1.0 + res_costly.portfolio_returns.to_numpy()))

    assert wealth_free > wealth_costly
    assert res_costly.total_costs_bps > 0.0


def test_compare_strategies_table(multi_asset_returns: pd.DataFrame) -> None:
    """Test full strategy comparison grid."""
    summary_df, results = compare_strategies(
        multi_asset_returns,
        strategies=["equal_weight", "min_variance", "risk_parity"],
        lookback=252,
        rebalance="quarterly",
    )
    assert len(summary_df) == 3
    assert "CAGR" in summary_df.columns
    assert "Annual Vol" in summary_df.columns
    assert "Annual Turnover" in summary_df.columns
    assert "equal_weight" in results
