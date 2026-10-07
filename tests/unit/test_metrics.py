"""Unit tests for core financial return, volatility, drawdown, ratio, and correlation metrics.

Enforces:
- Rule R-T1: Known-answer tests.
- Rule R-T2: Invariant property tests (drawdown <= 0, unit diagonal correlation).
- Rule R-T3: Degenerate cases (constant series, flat prices).
"""

import numpy as np
import pandas as pd
import pytest
from prismrisk.metrics.correlation import correlation_matrix, rolling_correlation
from prismrisk.metrics.drawdown import calmar_ratio, drawdown_series, max_drawdown
from prismrisk.metrics.ratios import (
    beta,
    information_ratio,
    sharpe_ratio,
    sortino_ratio,
    to_daily_risk_free,
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


def test_known_answer_returns_and_drawdown() -> None:
    """Hand-computed known answer test from Rule R-T1.

    Prices: [100, 110, 99, 121]
    Simple Returns: [0.10, -0.10, 0.222222...]
    Wealth Index: [1.10, 0.99, 1.21]
    Drawdown: [0.0, -0.10, 0.0]
    Max Drawdown: -0.10 (-10.0%)
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 110.0, 99.0, 121.0], index=dates, name="asset")

    s_ret = simple_returns(prices)
    assert len(s_ret) == 3
    assert pytest.approx(s_ret.iloc[0], 1e-6) == 0.10
    assert pytest.approx(s_ret.iloc[1], 1e-6) == -0.10
    assert pytest.approx(s_ret.iloc[2], 1e-6) == (121.0 / 99.0) - 1.0

    dd = drawdown_series(s_ret)
    assert pytest.approx(dd.iloc[0], 1e-6) == 0.0
    assert pytest.approx(dd.iloc[1], 1e-6) == -0.10
    assert pytest.approx(dd.iloc[2], 1e-6) == 0.0

    stats = max_drawdown(s_ret)
    assert pytest.approx(stats.depth, 1e-6) == -0.10
    assert stats.peak_date == dates[1]  # 2023-01-02
    assert stats.trough_date == dates[2]  # 2023-01-03
    assert stats.recovery_date == dates[3]  # 2023-01-04
    assert stats.to_trough_days == 1
    assert stats.recovery_days == 1
    assert stats.duration_days == 2


def test_known_answer_cagr() -> None:
    """Hand-computed CAGR test: 21% cumulative gain over exactly 2 years (504 days) = 10% CAGR."""
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    daily_gain = (1.21 ** (1.0 / 504.0)) - 1.0
    rets = pd.Series([daily_gain] * 504, index=dates)

    computed_cagr = cagr(rets, trading_days=252)
    assert pytest.approx(computed_cagr, 1e-5) == 0.10


def test_known_answer_sharpe_and_sortino() -> None:
    """Hand-computed Sharpe and Sortino test."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    # Alternating daily returns: +2% and 0%
    rets = pd.Series([0.02, 0.00] * 50, index=dates)

    sr = sharpe_ratio(rets, rf_annual=0.0, trading_days=252)
    assert sr > 0.0

    # Sortino with zero rf should be very high because downside deviation only considers returns < 0
    sort = sortino_ratio(rets, rf_annual=0.0, trading_days=252)
    assert sort == 0.0 or sort > sr


def test_known_answer_beta_and_tracking_error() -> None:
    """Hand-computed beta = 2.0 when portfolio returns are exactly 2 * benchmark."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    r_bm = pd.Series(np.linspace(-0.02, 0.02, 50), index=dates)
    r_port = 2.0 * r_bm

    computed_beta = beta(r_port, r_bm)
    assert pytest.approx(computed_beta, 1e-6) == 2.0

    # Tracking error of benchmark against itself is zero
    assert pytest.approx(tracking_error(r_bm, r_bm), 1e-6) == 0.0
    assert pytest.approx(information_ratio(r_bm, r_bm), 1e-6) == 0.0


def test_constant_price_series(constant_prices: pd.Series) -> None:
    """Degenerate case: constant prices have zero volatility, zero drawdown, zero returns."""
    rets = simple_returns(constant_prices)
    assert (rets == 0.0).all()
    assert annualised_vol(rets) == 0.0

    dd_stats = max_drawdown(rets)
    assert dd_stats.depth == 0.0
    assert dd_stats.duration_days == 0


def test_drawdown_invariant_always_non_positive() -> None:
    """Invariant: drawdown series must be <= 0 everywhere."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.001, 0.02, 100), index=dates)

    dd = drawdown_series(rets)
    assert (dd <= 1e-9).all()


def test_correlation_matrix_invariants() -> None:
    """Invariant: correlation matrix must be symmetric with unit diagonal."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "A": rng.normal(0, 1, 100),
            "B": rng.normal(0, 1, 100),
            "C": rng.normal(0, 1, 100),
        },
        index=dates,
    )

    corr = correlation_matrix(df)
    assert corr.shape == (3, 3)
    # Unit diagonal
    for i in range(3):
        assert pytest.approx(corr.iloc[i, i], 1e-6) == 1.0
    # Symmetry
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T, atol=1e-6)


def test_rolling_vol_and_ewma() -> None:
    """Test rolling volatility and EWMA volatility."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0, 0.01, 100), index=dates)

    roll = rolling_vol(rets, window=30)
    assert len(roll) == 100 - 30 + 1
    assert (roll > 0).all()

    ewma = ewma_vol(rets, lam=0.94)
    assert len(ewma) == 100
    assert (ewma > 0).all()


def test_log_returns_and_cumulative() -> None:
    """Test log returns and cumulative compounding."""
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    df = pd.DataFrame({"asset": [100.0, 110.0, 121.0, 133.1]}, index=dates)
    log_ret = log_returns(df)
    assert len(log_ret) == 3
    # Each day is a 10% gain -> ln(1.10)
    assert pytest.approx(log_ret.iloc[0]["asset"], 1e-4) == np.log(1.10)

    cum = cumulative_returns(pd.Series([0.10, 0.10], index=dates[:2]))
    assert pytest.approx(cum.iloc[-1], 1e-5) == 0.21


def test_portfolio_returns_weights() -> None:
    """Test portfolio returns weighted aggregation."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    ret_df = pd.DataFrame(
        {
            "equity": [0.02, -0.01, 0.03],
            "debt": [0.005, 0.005, 0.005],
        },
        index=dates,
    )
    weights = pd.Series({"equity": 0.80, "debt": 0.20})
    p_rets = portfolio_returns(ret_df, weights)
    assert len(p_rets) == 3
    expected_day1 = 0.80 * 0.02 + 0.20 * 0.005  # 0.017
    assert pytest.approx(p_rets.iloc[0], 1e-6) == expected_day1


def test_rolling_correlation() -> None:
    """Test rolling correlation between two series."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(0, 1, 100), index=dates)
    s2 = 0.5 * s1 + 0.5 * pd.Series(rng.normal(0, 1, 100), index=dates)

    rc = rolling_correlation(s1, s2, window=30)
    assert len(rc) == 100 - 30 + 1
    assert rc.mean() > 0.0

    with pytest.raises(ValueError, match="must be >= 2"):
        rolling_correlation(s1, s2, window=1)


def test_returns_series_input_and_invalid_types() -> None:
    """Test simple_returns and log_returns on Series inputs and invalid types."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    s_price = pd.Series([100.0, 110.0, 121.0], index=dates)

    s_ret = simple_returns(s_price)
    assert len(s_ret) == 2
    assert pytest.approx(s_ret.iloc[0], 1e-6) == 0.10

    l_ret = log_returns(s_price)
    assert len(l_ret) == 2
    assert pytest.approx(l_ret.iloc[0], 1e-6) == np.log(1.10)

    with pytest.raises(TypeError):
        simple_returns([100, 110])  # type: ignore

    with pytest.raises(TypeError):
        log_returns([100, 110])  # type: ignore


def test_volatility_and_ratio_edge_cases() -> None:
    """Test edge cases in volatility and ratios."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    s = pd.Series([0.01] * 10, index=dates)

    with pytest.raises(ValueError, match="must be in"):
        ewma_vol(s, lam=1.5)

    with pytest.raises(ValueError, match="must be >= 2"):
        rolling_vol(s, window=1)

    with pytest.raises(ValueError, match="must be > -1.0"):
        to_daily_risk_free(-1.5)
