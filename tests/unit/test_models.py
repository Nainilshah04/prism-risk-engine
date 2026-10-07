"""Unit tests for domain models and dataclasses."""

import pandas as pd
from prismrisk.models import (
    BacktestResult,
    DataQualityReport,
    DrawdownStats,
    KupiecResult,
    Results,
    ScenarioResult,
    VarResult,
)


def test_drawdown_stats_instantiation() -> None:
    """Test DrawdownStats dataclass properties."""
    now = pd.Timestamp("2023-01-01")
    stats = DrawdownStats(
        depth=-0.15,
        peak_date=now,
        trough_date=now + pd.Timedelta(days=10),
        recovery_date=now + pd.Timedelta(days=25),
        duration_days=25,
        to_trough_days=10,
        recovery_days=15,
    )
    assert stats.depth == -0.15
    assert stats.duration_days == 25


def test_var_result_instantiation() -> None:
    """Test VarResult dataclass."""
    res = VarResult(method="historical", confidence=0.95, var=0.021, cvar=0.035)
    assert res.var == 0.021
    assert res.cvar == 0.035
    assert res.confidence == 0.95


def test_kupiec_result_instantiation() -> None:
    """Test KupiecResult dataclass."""
    res = KupiecResult(
        confidence=0.99,
        n_obs=1000,
        breaches=12,
        expected_breaches=10.0,
        breach_rate=0.012,
        lr_stat=0.38,
        p_value=0.53,
        reject_null=False,
    )
    assert not res.reject_null
    assert res.breaches == 12


def test_scenario_result_instantiation() -> None:
    """Test ScenarioResult dataclass."""
    sr = ScenarioResult(
        name="COVID crash",
        asset_returns=pd.Series({"equity": -0.25, "gold": 0.05, "debt": 0.01}),
        portfolio_impact=-0.14,
        category="historical",
    )
    assert sr.portfolio_impact == -0.14
    assert sr.category == "historical"


def test_backtest_result_instantiation() -> None:
    """Test BacktestResult dataclass."""
    dates = pd.date_range("2023-01-01", periods=5)
    btr = BacktestResult(
        strategy_name="equal_weight",
        portfolio_returns=pd.Series([0.01] * 5, index=dates),
        weights_over_time=pd.DataFrame({"equity": [0.5] * 5, "debt": [0.5] * 5}, index=dates),
        rebalance_trades=pd.DataFrame(),
        cumulative_turnover=0.10,
        total_costs_bps=5.0,
        cagr=0.12,
        annual_vol=0.15,
        sharpe=0.80,
        max_drawdown=-0.08,
    )
    assert btr.strategy_name == "equal_weight"
    assert btr.cagr == 0.12


def test_results_container() -> None:
    """Test top-level Results dataclass container."""
    now = pd.Timestamp("2023-01-01")
    dd_stats = DrawdownStats(
        depth=0.0,
        peak_date=now,
        trough_date=now,
        recovery_date=None,
        duration_days=0,
        to_trough_days=0,
        recovery_days=None,
    )
    dq = DataQualityReport(
        missing_pct=pd.Series(),
        longest_gap_days=pd.Series(),
        zero_return_days=pd.Series(),
        outlier_count=pd.Series(),
        start_date_per_asset=pd.Series(),
        effective_start=now,
        effective_end=now,
        total_trading_days=1,
        passed_checks=True,
    )
    res = Results(
        prices=pd.DataFrame(),
        returns=pd.DataFrame(),
        portfolio_returns=pd.Series(),
        benchmark_returns=None,
        metrics_table=pd.DataFrame(),
        drawdown_series=pd.Series(),
        drawdown_stats=dd_stats,
        var_table=pd.DataFrame(),
        var_backtest=pd.DataFrame(),
        kupiec_results={},
        strategy_table=pd.DataFrame(),
        scenario_table=pd.DataFrame(),
        risk_contributions=pd.DataFrame(),
        dq_report=dq,
        config=None,
    )
    assert res.benchmark_returns is None
    assert res.dq_report.passed_checks is True
