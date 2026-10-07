"""Configuration management and Pydantic validation schemas.

Loads, validates, and serializes YAML configs according to the design specification.
Fails fast with descriptive errors for invalid weights, unknown strategies, or missing tickers.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AssetConfig(BaseModel):
    """Specification for an asset or proxy ticker."""
    ticker: str
    type: Literal["equity", "gold", "debt", "benchmark", "fx", "other"] = "equity"
    sensitivity: Dict[str, float] = Field(default_factory=dict)
    duration: Optional[float] = None
    convexity: Optional[float] = None
    synthetic_yield: Optional[float] = None


class DataConfig(BaseModel):
    """Data ingestion and storage configuration."""
    start: Optional[str] = "2015-01-01"
    end: Optional[str] = None
    source: Literal["yfinance", "fred", "snapshot"] = "yfinance"
    cache_dir: str = "data/raw"
    snapshot_dir: str = "data/snapshot"
    max_ffill_days: int = Field(default=3, ge=0, le=10)


class PortfolioConfig(BaseModel):
    """Static portfolio definition and baseline rebalancing schedule."""
    weights: Dict[str, float]
    rebalance: Literal["none", "daily", "monthly", "quarterly", "annual"] = "quarterly"

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Portfolio weights dictionary cannot be empty.")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.4f}")
        for asset, w in v.items():
            if w < 0:
                raise ValueError(f"Long-only constraint violated for '{asset}': weight {w} < 0")
        return v


class RiskFreeConfig(BaseModel):
    """Risk-free proxy definition."""
    type: Literal["flat", "series"] = "flat"
    annual_rate: float = Field(default=0.065, ge=-0.05, le=0.50)
    ticker: Optional[str] = None


class MetricsConfig(BaseModel):
    """Parameters for returns and volatility computation."""
    trading_days: int = Field(default=252, gt=0)
    vol_windows: List[int] = Field(default_factory=lambda: [30, 90])
    ewma_lambda: float = Field(default=0.94, gt=0.0, lt=1.0)


class RiskConfig(BaseModel):
    """Parameters for tail-risk (VaR/CVaR) models."""
    confidence: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    mc_paths: int = Field(default=10000, gt=100)
    seed: int = Field(default=42)
    var_window: int = Field(default=252, gt=10)

    @field_validator("confidence")
    @classmethod
    def check_confidences(cls, v: List[float]) -> List[float]:
        for conf in v:
            if not (0.5 < conf < 1.0):
                raise ValueError(f"Confidence level must be in (0.5, 1.0), got {conf}")
        return sorted(v)


class CapsConfig(BaseModel):
    """Asset weight constraints for portfolio optimization."""
    max_weight: float = Field(default=0.70, gt=0.0, le=1.0)
    min_weight: float = Field(default=0.00, ge=0.0, lt=1.0)


class StrategiesConfig(BaseModel):
    """Walk-forward backtest strategy comparison configuration."""
    lookback: int = Field(default=252, gt=20)
    rebalance: Literal["daily", "monthly", "quarterly", "annual"] = "monthly"
    cost_bps: float = Field(default=10.0, ge=0.0)
    covariance: Literal["ledoit_wolf", "sample"] = "ledoit_wolf"
    caps: CapsConfig = Field(default_factory=CapsConfig)
    include: List[str] = Field(
        default_factory=lambda: [
            "equal_weight",
            "inverse_vol",
            "min_variance",
            "risk_parity",
            "max_sharpe",
        ]
    )


class OutputConfig(BaseModel):
    """Output generation parameters."""
    dir: str = "outputs"
    factsheet: bool = True


class HistoricalScenarioConfig(BaseModel):
    """Historical market crisis window."""
    name: str
    start: str
    end: str


class HypotheticalScenarioConfig(BaseModel):
    """Hypothetical stress shocks across asset classes or factors."""
    name: str
    shocks: Dict[str, float]


class GridScenarioConfig(BaseModel):
    """Two-dimensional sensitivity stress grid."""
    name: str
    x: str
    x_values: List[float]
    y: str
    y_values: List[float]


class ScenariosConfig(BaseModel):
    """Full scenario definitions loaded from scenarios YAML."""
    historical: List[HistoricalScenarioConfig] = Field(default_factory=list)
    hypothetical: List[HypotheticalScenarioConfig] = Field(default_factory=list)
    grids: List[GridScenarioConfig] = Field(default_factory=list)


class Config(BaseModel):
    """Main Prism Risk Engine configuration root."""
    project: str = "prism-demo"
    data: DataConfig = Field(default_factory=DataConfig)
    assets: Dict[str, AssetConfig]
    portfolio: PortfolioConfig
    risk_free: RiskFreeConfig = Field(default_factory=RiskFreeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    scenarios_file: Optional[str] = "configs/scenarios.yaml"
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_portfolio_assets(self) -> "Config":
        """Verify that every portfolio weight corresponds to a defined asset."""
        missing = [a for a in self.portfolio.weights if a not in self.assets]
        if missing:
            raise ValueError(
                f"Portfolio weights reference assets not declared in 'assets': {missing}"
            )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load and validate configuration from YAML file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        return cls.model_validate(raw_data)

    def load_scenarios(self) -> ScenariosConfig:
        """Load companion scenarios file if configured."""
        if not self.scenarios_file:
            return ScenariosConfig()
        p = Path(self.scenarios_file)
        if not p.exists():
            return ScenariosConfig()
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return ScenariosConfig.model_validate(raw)
