"""Deterministic synthetic market data generator for offline testing and committed snapshot.

Follows Rule R-C7 (Deterministic: fixed seed) and Rule R-T5 (Offline testing).
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def generate_synthetic_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2018-01-01",
    end_date: str = "2024-01-01",
    seed: int = 42,
    initial_prices: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Generate realistic correlated daily adjusted prices using Geometric Brownian Motion.

    Args:
        tickers: List of asset column names.
        start_date: Start date string.
        end_date: End date string.
        seed: Random seed for 100% determinism.
        initial_prices: Starting price mapping.

    Returns:
        DataFrame[date x asset] with tz-naive DatetimeIndex, sorted, strictly positive.
    """
    if tickers is None:
        tickers = ["equity", "gold", "debt", "benchmark", "fx"]

    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    rng = np.random.default_rng(seed)

    # Asset drift (annualized) and volatilities
    params = {
        "equity": {"mu": 0.12, "sigma": 0.18, "start": 100.0},
        "gold": {"mu": 0.08, "sigma": 0.13, "start": 3000.0},
        "debt": {"mu": 0.065, "sigma": 0.015, "start": 1000.0},
        "benchmark": {"mu": 0.11, "sigma": 0.17, "start": 10000.0},
        "fx": {"mu": 0.03, "sigma": 0.06, "start": 75.0},
    }

    # Correlation matrix
    corr = np.array([
        [1.00,  0.05,  0.02,  0.96,  0.10],  # equity
        [0.05,  1.00,  0.05,  0.06, -0.15],  # gold
        [0.02,  0.05,  1.00,  0.02,  0.01],  # debt
        [0.96,  0.06,  0.02,  1.00,  0.08],  # benchmark
        [0.10, -0.15,  0.01,  0.08,  1.00],  # fx
    ])

    # Subset or adapt correlation if different tickers passed
    if n_assets == 5 and tickers == ["equity", "gold", "debt", "benchmark", "fx"]:
        c_matrix = corr
    else:
        c_matrix = np.eye(n_assets)

    # Cholesky decomposition for correlated normal draws
    chol = np.linalg.cholesky(c_matrix)

    dt = 1.0 / 252.0
    uncorrelated_normals = rng.standard_normal(size=(n_days, n_assets))
    correlated_normals = uncorrelated_normals @ chol.T

    price_dict: Dict[str, np.ndarray] = {}
    for i, ticker in enumerate(tickers):
        p_cfg = params.get(ticker, {"mu": 0.08, "sigma": 0.15, "start": 100.0})
        mu = p_cfg["mu"]
        sigma = p_cfg["sigma"]
        start_p = (initial_prices or {}).get(ticker, p_cfg["start"])

        # Geometric Brownian Motion step
        daily_drift = (mu - 0.5 * sigma**2) * dt
        daily_diff = sigma * np.sqrt(dt) * correlated_normals[:, i]
        log_ret = daily_drift + daily_diff

        # Cumulative compounding
        cum_ret = np.cumsum(log_ret)
        cum_ret = np.insert(cum_ret[:-1], 0, 0.0)
        prices = start_p * np.exp(cum_ret)
        price_dict[ticker] = prices

    df = pd.DataFrame(price_dict, index=dates)
    df.index.name = "date"
    return df


def save_default_snapshot(target_dir: str = "data/snapshot") -> Path:
    """Generate and commit synthetic benchmark snapshot in Parquet format."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "market_data_snapshot.parquet"

    df = generate_synthetic_prices()
    df.to_parquet(out_path, engine="pyarrow")
    return out_path
