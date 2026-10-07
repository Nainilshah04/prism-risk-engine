"""Unit tests for configuration loading and validation schemas."""

import pytest
from pydantic import ValidationError
from prismrisk.config import Config


def test_load_valid_demo_config() -> None:
    """Test loading and validating configs/demo.yaml."""
    cfg = Config.load("configs/demo.yaml")
    assert cfg.project == "demo-60-20-20"
    assert "equity" in cfg.assets
    assert "gold" in cfg.assets
    assert "debt" in cfg.assets
    assert cfg.portfolio.weights["equity"] == 0.60
    assert cfg.portfolio.weights["gold"] == 0.20
    assert cfg.portfolio.weights["debt"] == 0.20
    assert abs(sum(cfg.portfolio.weights.values()) - 1.0) < 1e-4
    assert cfg.risk_free.annual_rate == 0.065
    assert cfg.metrics.trading_days == 252


def test_load_scenarios(demo_config: Config) -> None:
    """Test loading companion scenarios file."""
    scenarios = demo_config.load_scenarios()
    assert len(scenarios.historical) >= 2
    assert any(s.name == "COVID crash" for s in scenarios.historical)
    assert len(scenarios.hypothetical) >= 3
    assert len(scenarios.grids) >= 1


def test_invalid_weights_sum_fails() -> None:
    """Ensure weights that do not sum to 1.0 raise validation error."""
    raw = {
        "assets": {
            "equity": {"ticker": "EQ"},
            "debt": {"ticker": "DB"},
        },
        "portfolio": {
            "weights": {"equity": 0.5, "debt": 0.4},  # Sums to 0.90
        },
    }
    with pytest.raises(ValidationError, match="Portfolio weights must sum to 1.0"):
        Config.model_validate(raw)


def test_negative_weight_fails() -> None:
    """Ensure negative weights are rejected in long-only configuration."""
    raw = {
        "assets": {
            "equity": {"ticker": "EQ"},
            "debt": {"ticker": "DB"},
        },
        "portfolio": {
            "weights": {"equity": 1.2, "debt": -0.2},
        },
    }
    with pytest.raises(ValidationError, match="Long-only constraint violated"):
        Config.model_validate(raw)


def test_missing_asset_declaration_fails() -> None:
    """Ensure weights cannot reference undeclared assets."""
    raw = {
        "assets": {
            "equity": {"ticker": "EQ"},
        },
        "portfolio": {
            "weights": {"equity": 0.5, "crypto": 0.5},
        },
    }
    with pytest.raises(ValidationError, match="reference assets not declared"):
        Config.model_validate(raw)
