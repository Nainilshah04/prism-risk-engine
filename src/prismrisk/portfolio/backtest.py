"""Lookahead-safe walk-forward portfolio backtesting engine with realistic drift and costs.

Enforces:
- Rule R-F2: Zero look-ahead bias. Weights for day t depend strictly on data up to t-1.
- Rule R-F9: Transaction costs in basis points and explicit weight drift between rebalances.
- Rule FR-P3 & FR-P5: Walk-forward engine and strategy comparison table.
"""

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from prismrisk.metrics.drawdown import max_drawdown
from prismrisk.metrics.ratios import sharpe_ratio
from prismrisk.metrics.returns import cagr
from prismrisk.metrics.volatility import annualised_vol
from prismrisk.models import BacktestResult
from prismrisk.portfolio.covariance import estimate_cov
from prismrisk.portfolio.optimisers import optimise
from prismrisk.utils.validation import validate_returns


def identify_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "monthly",
) -> List[pd.Timestamp]:
    """Identify calendar dates on which rebalancing takes place.

    Args:
        dates: Sorted DatetimeIndex of trading days.
        frequency: Rebalance frequency ('daily', 'monthly', 'quarterly', 'annual').

    Returns:
        List of timestamps marked as rebalance points.
    """
    if frequency == "daily":
        return list(dates)

    series_df = pd.DataFrame({"date": dates}, index=dates)

    if frequency == "monthly":
        # First trading day of each year-month
        grouped = series_df.groupby([dates.year, dates.month]).first()
        return list(grouped["date"])
    elif frequency == "quarterly":
        # First trading day of each year-quarter
        grouped = series_df.groupby([dates.year, dates.quarter]).first()
        return list(grouped["date"])
    elif frequency == "annual":
        grouped = series_df.groupby(dates.year).first()
        return list(grouped["date"])
    else:
        raise ValueError(
            f"Unknown rebalance frequency '{frequency}'. Supported: daily, monthly, quarterly, annual"
        )


def run_backtest(
    returns: pd.DataFrame,
    strategy: str,
    lookback: int = 252,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    covariance_method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
    max_weight: float = 0.70,
    min_weight: float = 0.0,
    rf_annual: float = 0.065,
    trading_days: int = 252,
) -> BacktestResult:
    """Execute out-of-sample walk-forward backtest without lookahead bias.

    Algorithm:
        Between rebalances:
            Weights drift with daily asset returns:
            w_{t+1}^{drift} = [w_{i,t} * (1 + r_{i,t})] / (1 + r_{p,t}^{gross})
        On rebalance date t:
            window = returns[t - lookback : t] (strictly data prior to t)
            w_target = optimise(window)
            turnover = sum(|w_target - w_t^{drift}|)
            cost_t = (cost_bps / 1e4) * turnover
            w_held = w_target
            r_{p,t}^{net} = (w_held @ r_t) - cost_t

    Args:
        returns: Asset simple returns DataFrame [date x asset].
        strategy: Optimization strategy name.
        lookback: Estimation lookback window in days (default: 252).
        rebalance: Rebalance frequency ('daily', 'monthly', 'quarterly', 'annual').
        cost_bps: Proportional transaction cost in basis points (default: 10 bps).
        covariance_method: Covariance estimator ('ledoit_wolf' or 'sample').
        max_weight: Upper bound weight cap per asset.
        min_weight: Lower bound weight floor per asset.
        rf_annual: Annual risk-free rate.
        trading_days: Annualization scaling factor (252).

    Returns:
        BacktestResult dataclass.
    """
    validate_returns(returns)
    assets = list(returns.columns)
    n_assets = len(assets)
    n_days = len(returns)

    if n_days <= lookback:
        raise ValueError(
            f"Insufficient data ({n_days} days) for lookback window of {lookback} days."
        )

    all_dates = returns.index
    eval_dates = all_dates[lookback:]
    rebal_dates_set = set(identify_rebalance_dates(all_dates, frequency=rebalance))

    weights_held = np.empty((len(eval_dates), n_assets))
    trades_executed = np.empty((len(eval_dates), n_assets))
    portfolio_net_rets = np.empty(len(eval_dates))
    daily_turnovers = np.zeros(len(eval_dates))
    daily_costs = np.zeros(len(eval_dates))

    current_w = np.full(n_assets, 1.0 / n_assets)
    is_initialized = False

    cost_multiplier = cost_bps / 10000.0

    for idx, t in enumerate(range(lookback, n_days)):
        date_t = all_dates[t]
        daily_asset_r = returns.iloc[t].to_numpy()

        is_rebal_day = (date_t in rebal_dates_set) or (not is_initialized)

        if is_rebal_day:
            # Strictly past data ending at t (i.e. rows t-lookback to t-1)
            past_window = returns.iloc[t - lookback : t]
            cov = estimate_cov(past_window, method=covariance_method)

            target_w_series = optimise(
                strategy=strategy,
                returns_window=past_window,
                cov=cov,
                max_weight=max_weight,
                min_weight=min_weight,
                rf_annual=rf_annual,
            )
            target_w = target_w_series.loc[assets].to_numpy()

            if not is_initialized:
                # Initial portfolio acquisition
                trade = target_w
                turnover = float(np.sum(np.abs(target_w)))
                is_initialized = True
            else:
                trade = target_w - current_w
                turnover = float(np.sum(np.abs(trade)))

            cost = turnover * cost_multiplier
            current_w = target_w
        else:
            trade = np.zeros(n_assets)
            turnover = 0.0
            cost = 0.0

        # Execute day t with current weights
        gross_p_ret = float(np.dot(current_w, daily_asset_r))
        net_p_ret = gross_p_ret - cost

        weights_held[idx] = current_w
        trades_executed[idx] = trade
        portfolio_net_rets[idx] = net_p_ret
        daily_turnovers[idx] = turnover
        daily_costs[idx] = cost

        # Drift weights at end of day t for day t+1
        denom = 1.0 + gross_p_ret
        if denom > 1e-8:
            current_w = (current_w * (1.0 + daily_asset_r)) / denom
        current_w = np.maximum(0.0, current_w)
        current_w = current_w / np.sum(current_w)

    p_rets_series = pd.Series(portfolio_net_rets, index=eval_dates, name=f"{strategy}_returns")
    w_df = pd.DataFrame(weights_held, index=eval_dates, columns=assets)
    trades_df = pd.DataFrame(trades_executed, index=eval_dates, columns=assets)

    # Performance metrics
    total_turnover = float(np.sum(daily_turnovers))
    years = len(eval_dates) / trading_days
    annual_turnover = total_turnover / max(1e-4, years)
    total_costs_bps = float(np.sum(daily_costs)) * 10000.0

    cagr_val = cagr(p_rets_series, trading_days=trading_days)
    vol_val = annualised_vol(p_rets_series, trading_days=trading_days)
    sharpe_val = sharpe_ratio(p_rets_series, rf_annual=rf_annual, trading_days=trading_days)
    dd_stats = max_drawdown(p_rets_series)

    return BacktestResult(
        strategy_name=strategy,
        portfolio_returns=p_rets_series,
        weights_over_time=w_df,
        rebalance_trades=trades_df,
        cumulative_turnover=annual_turnover,
        total_costs_bps=total_costs_bps,
        cagr=cagr_val,
        annual_vol=vol_val,
        sharpe=sharpe_val,
        max_drawdown=dd_stats.depth,
    )


def compare_strategies(
    returns: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    lookback: int = 252,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    covariance_method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
    max_weight: float = 0.70,
    min_weight: float = 0.0,
    rf_annual: float = 0.065,
    trading_days: int = 252,
) -> Tuple[pd.DataFrame, Dict[str, BacktestResult]]:
    """Compare multiple walk-forward portfolio optimization strategies.

    Args:
        returns: Asset daily returns DataFrame.
        strategies: List of strategies to test (default: all 5).
        lookback: Lookback window in trading days.
        rebalance: Rebalance frequency.
        cost_bps: Transaction cost in bps.
        covariance_method: Covariance shrinkage or sample.
        max_weight: Maximum weight cap.
        min_weight: Minimum weight floor.
        rf_annual: Annual risk-free rate proxy.
        trading_days: Annualization factor (252).

    Returns:
        Tuple of:
            - Comparison summary DataFrame formatted for publication/display.
            - Dictionary of strategy_name -> BacktestResult.
    """
    if strategies is None:
        strategies = [
            "equal_weight",
            "inverse_vol",
            "min_variance",
            "risk_parity",
            "max_sharpe",
        ]

    results_dict: Dict[str, BacktestResult] = {}
    rows = []

    for strat in strategies:
        btr = run_backtest(
            returns=returns,
            strategy=strat,
            lookback=lookback,
            rebalance=rebalance,
            cost_bps=cost_bps,
            covariance_method=covariance_method,
            max_weight=max_weight,
            min_weight=min_weight,
            rf_annual=rf_annual,
            trading_days=trading_days,
        )
        results_dict[strat] = btr

        calmar = btr.cagr / abs(btr.max_drawdown) if abs(btr.max_drawdown) > 1e-6 else np.inf

        rows.append(
            {
                "Strategy": strat.replace("_", " ").title(),
                "CAGR": f"{btr.cagr:.2%}",
                "Annual Vol": f"{btr.annual_vol:.2%}",
                "Sharpe": f"{btr.sharpe:.2f}",
                "Max Drawdown": f"{btr.max_drawdown:.2%}",
                "Calmar": f"{calmar:.2f}",
                "Annual Turnover": f"{btr.cumulative_turnover:.1%}",
                "Total Costs (bps)": f"{btr.total_costs_bps:.1f}",
            }
        )

    summary_df = pd.DataFrame(rows).set_index("Strategy")
    return summary_df, results_dict
