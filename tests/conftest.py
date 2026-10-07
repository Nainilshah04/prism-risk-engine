"""Global pytest fixtures for Prism Risk Engine test suite."""

import pytest
import pandas as pd
from prismrisk.config import Config
from tests.fixtures.synthetic_data import generate_synthetic_prices


@pytest.fixture
def demo_config() -> Config:
    """Load the default demo configuration."""
    return Config.load("configs/demo.yaml")


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic multi-asset price matrix (2018-2024)."""
    return generate_synthetic_prices(seed=42)


@pytest.fixture
def known_drawdown_prices() -> pd.Series:
    """Hand-computed price series with exact known drawdown characteristics.

    Prices: [100.0, 110.0, 99.0, 121.0]
    Peak: 110.0
    Trough: 99.0 -> Max Drawdown = (99 / 110) - 1 = -0.10 (-10.0%)
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    return pd.Series([100.0, 110.0, 99.0, 121.0], index=dates, name="asset")


@pytest.fixture
def constant_prices() -> pd.Series:
    """Degenerate constant price series with zero volatility and zero drawdown."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates, name="constant_asset")
