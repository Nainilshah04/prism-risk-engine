"""Independent verification test: Compares engine output against independent spreadsheet formulas.

Satisfies DoD for Phase 2:
'Metrics table for the demo portfolio and benchmark matches an independent spreadsheet check for at least 3 metrics.'
"""

import numpy as np
import pandas as pd
import pytest
from prismrisk.config import Config
from prismrisk.data.calendar import align_prices
from prismrisk.data.providers import SnapshotProvider
from prismrisk.metrics.drawdown import max_drawdown
from prismrisk.metrics.ratios import beta, sharpe_ratio
from prismrisk.metrics.returns import cagr, portfolio_returns, simple_returns
from prismrisk.metrics.volatility import annualised_vol


def test_metrics_match_independent_spreadsheet_formulas() -> None:
    """Validate 5 core metrics against raw spreadsheet formulas on real snapshot data."""
    # 1. Load demo portfolio data from committed snapshot
    cfg = Config.load("configs/demo.yaml")
    tickers = {name: asset.ticker for name, asset in cfg.assets.items()}
    provider = SnapshotProvider("data/snapshot/market_data_snapshot.parquet")
    raw_prices = provider.fetch_all(tickers=tickers, start=cfg.data.start, end=cfg.data.end)
    aligned_prices, _ = align_prices(raw_prices, max_ffill_days=cfg.data.max_ffill_days)

    asset_returns = simple_returns(aligned_prices)
    weights = pd.Series(cfg.portfolio.weights)

    # Portfolio return series from engine
    p_rets = portfolio_returns(asset_returns, weights)
    bm_rets = asset_returns["benchmark"]
    n_days = len(p_rets)

    # -------------------------------------------------------------
    # SPREADSHEET CHECK 1: CAGR = (Product(1 + r_t))^(252/N) - 1
    # -------------------------------------------------------------
    spreadsheet_wealth_terminal = float(np.prod(1.0 + p_rets.to_numpy()))
    spreadsheet_cagr = (spreadsheet_wealth_terminal ** (252.0 / n_days)) - 1.0

    engine_cagr = cagr(p_rets, trading_days=252)
    assert pytest.approx(engine_cagr, 1e-6) == spreadsheet_cagr

    # -------------------------------------------------------------
    # SPREADSHEET CHECK 2: Annualized Vol = STDEV.S(r) * SQRT(252)
    # -------------------------------------------------------------
    spreadsheet_daily_std = float(p_rets.std(ddof=1))
    spreadsheet_vol = spreadsheet_daily_std * np.sqrt(252.0)

    engine_vol = annualised_vol(p_rets, trading_days=252)
    assert pytest.approx(engine_vol, 1e-6) == spreadsheet_vol

    # -------------------------------------------------------------
    # SPREADSHEET CHECK 3: Max Drawdown = MIN(V_t / MAX(V_{1:t}) - 1)
    # -------------------------------------------------------------
    wealth_series = np.cumprod(1.0 + p_rets.to_numpy())
    running_max_series = np.maximum.accumulate(wealth_series)
    running_max_series = np.maximum(running_max_series, 1.0)
    spreadsheet_dd_series = (wealth_series / running_max_series) - 1.0
    spreadsheet_mdd = float(np.min(spreadsheet_dd_series))

    engine_dd_stats = max_drawdown(p_rets)
    assert pytest.approx(engine_dd_stats.depth, 1e-6) == spreadsheet_mdd

    # -------------------------------------------------------------
    # SPREADSHEET CHECK 4: Sharpe = (AVERAGE(r - rf_d) / STDEV.S(r - rf_d)) * SQRT(252)
    # -------------------------------------------------------------
    rf_daily = (1.0 + cfg.risk_free.annual_rate) ** (1.0 / 252.0) - 1.0
    excess_rets = p_rets.to_numpy() - rf_daily
    spreadsheet_sharpe = (np.mean(excess_rets) / np.std(excess_rets, ddof=1)) * np.sqrt(252.0)

    engine_sharpe = sharpe_ratio(p_rets, rf_annual=cfg.risk_free.annual_rate, trading_days=252)
    assert pytest.approx(engine_sharpe, 1e-6) == spreadsheet_sharpe

    # -------------------------------------------------------------
    # SPREADSHEET CHECK 5: Beta = COVARIANCE.S(r_p, r_bm) / VAR.S(r_bm)
    # -------------------------------------------------------------
    cov_matrix = np.cov(p_rets.to_numpy(), bm_rets.to_numpy(), ddof=1)
    spreadsheet_beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    engine_beta = beta(p_rets, bm_rets)
    assert pytest.approx(engine_beta, 1e-6) == spreadsheet_beta
