"""Core domain models and output dataclasses for Prism Risk Engine.

Enforces Rule R-F5: Returns and drawdowns are negative when losing.
VaR/CVaR are reported as positive loss numbers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass(frozen=True)
class DrawdownStats:
    """Drawdown characteristics for an asset or portfolio.

    Attributes:
        depth: Peak-to-trough decline (negative float, e.g. -0.28 for -28%).
        peak_date: Date on which the pre-drawdown high was established.
        trough_date: Date on which the maximum drawdown low occurred.
        recovery_date: Date on which the prior peak was recovered, or None if still in drawdown.
        duration_days: Calendar days from peak to recovery (or to end if unrecovered).
        to_trough_days: Calendar days from peak to trough.
        recovery_days: Calendar days from trough to recovery (or None).
    """
    depth: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    duration_days: int
    to_trough_days: int
    recovery_days: Optional[int]


@dataclass(frozen=True)
class VarResult:
    """Value at Risk and Conditional Value at Risk result.

    Note: VaR and CVaR are reported as positive loss percentages (e.g. 0.021 = 2.1% loss).
    """
    method: str
    confidence: float
    var: float
    cvar: float
    simulated_returns: Optional[pd.Series] = None


@dataclass(frozen=True)
class KupiecResult:
    """Kupiec Proportion of Failures (POF) Likelihood Ratio test for VaR backtesting.

    Attributes:
        confidence: Evaluated VaR confidence level (e.g. 0.95, 0.99).
        n_obs: Total observations tested.
        breaches: Number of realized losses exceeding VaR.
        expected_breaches: Theoretical expected breaches = n_obs * (1 - confidence).
        breach_rate: Realized breach proportion = breaches / n_obs.
        lr_stat: Likelihood-ratio test statistic (~ chi-squared with 1 df).
        p_value: Probability under null hypothesis that VaR model is well-calibrated.
        reject_null: True if model is rejected at 5% significance level (p_value < 0.05).
    """
    confidence: float
    n_obs: int
    breaches: int
    expected_breaches: float
    breach_rate: float
    lr_stat: float
    p_value: float
    reject_null: bool


@dataclass
class BacktestResult:
    """Result of a walk-forward portfolio backtest.

    Attributes:
        strategy_name: Name of the allocation strategy.
        portfolio_returns: Daily net portfolio returns after costs.
        weights_over_time: DataFrame [date x asset] of daily held weights.
        rebalance_trades: DataFrame of executed rebalancing weight deltas.
        cumulative_turnover: Total annualized weight turnover.
        total_costs_bps: Cumulative transaction drag in basis points.
        cagr: Compound Annual Growth Rate.
        annual_vol: Annualized standard deviation.
        sharpe: Sharpe ratio.
        max_drawdown: Maximum drawdown (negative float).
    """
    strategy_name: str
    portfolio_returns: pd.Series
    weights_over_time: pd.DataFrame
    rebalance_trades: pd.DataFrame
    cumulative_turnover: float
    total_costs_bps: float
    cagr: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


@dataclass
class ScenarioResult:
    """Output of a historical replay or hypothetical macro shock test.

    Attributes:
        name: Scenario description or crisis name.
        asset_returns: Realized or shocked returns for each individual asset.
        portfolio_impact: Aggregate portfolio return/loss (negative if loss).
        category: 'historical' or 'hypothetical'.
        details: Optional breakdown dictionary.
    """
    name: str
    asset_returns: pd.Series
    portfolio_impact: float
    category: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Data quality diagnostic summary for ingested assets."""
    missing_pct: pd.Series
    longest_gap_days: pd.Series
    zero_return_days: pd.Series
    outlier_count: pd.Series
    start_date_per_asset: pd.Series
    effective_start: pd.Timestamp
    effective_end: pd.Timestamp
    total_trading_days: int
    passed_checks: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Results:
    """Top-level analytical container returned by run_analysis pipeline.

    Clean separation: UI and reporters only consume this container; they perform zero analytics.
    """
    prices: pd.DataFrame
    returns: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: Optional[pd.Series]
    metrics_table: pd.DataFrame
    drawdown_series: pd.Series
    drawdown_stats: DrawdownStats
    var_table: pd.DataFrame
    var_backtest: pd.DataFrame
    kupiec_results: Dict[float, KupiecResult]
    strategy_table: pd.DataFrame
    scenario_table: pd.DataFrame
    risk_contributions: pd.DataFrame
    dq_report: DataQualityReport
    config: Any
