"""Script to analyze tail risk, run Kupiec tests, and generate comparison charts.

Produces docs/var_tail_risk_comparison.png and prints findings for the README.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prismrisk.config import Config
from prismrisk.data.calendar import align_prices
from prismrisk.data.providers import SnapshotProvider
from prismrisk.metrics.returns import portfolio_returns, simple_returns
from prismrisk.risk.backtest import backtest_var_model, rolling_var
from prismrisk.risk.var import compare_var_models, historical_var, parametric_var


def run_tail_risk_analysis() -> None:
    cfg = Config.load("configs/demo.yaml")
    provider = SnapshotProvider("data/snapshot/market_data_snapshot.parquet")
    tickers = {name: a.ticker for name, a in cfg.assets.items()}
    raw = provider.fetch_all(tickers=tickers, start=cfg.data.start, end=cfg.data.end)
    aligned, _ = align_prices(raw, max_ffill_days=cfg.data.max_ffill_days)
    rets = simple_returns(aligned)

    weights = pd.Series(cfg.portfolio.weights)
    p_rets = portfolio_returns(rets[list(weights.index)], weights)

    # 1. Compare models
    mu = rets[list(weights.index)].mean()
    cov = rets[list(weights.index)].cov()
    var_table = compare_var_models(
        returns=p_rets,
        mu=mu,
        cov=cov,
        weights=weights,
        confidences=[0.95, 0.99],
        n_paths=20000,
        seed=42,
    )
    print("\n=== Model Comparison Table ===")
    print(var_table.to_string())

    # 2. Rolling Backtests
    summary_df, results_map = backtest_var_model(
        p_rets,
        window=252,
        confidences=[0.95, 0.99],
        methods=["historical", "parametric"],
    )
    print("\n=== VaR Backtesting & Kupiec Test Results ===")
    print(summary_df.to_string())

    # 3. Create visual chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Return histogram & VaR cutoff lines
    h_95 = historical_var(p_rets, 0.95)
    p_95 = parametric_var(p_rets, 0.95)
    h_99 = historical_var(p_rets, 0.99)
    p_99 = parametric_var(p_rets, 0.99)

    axes[0].hist(p_rets * 100, bins=70, density=True, alpha=0.6, color="#1f77b4", label="Realized Returns")
    axes[0].axvline(-h_95 * 100, color="#d4a017", linestyle="--", linewidth=1.8, label=f"Historical 95% VaR: {h_95:.2%}")
    axes[0].axvline(-p_95 * 100, color="#d4a017", linestyle=":", linewidth=1.8, label=f"Parametric 95% VaR: {p_95:.2%}")
    axes[0].axvline(-h_99 * 100, color="#c0392b", linestyle="--", linewidth=2.2, label=f"Historical 99% VaR: {h_99:.2%}")
    axes[0].axvline(-p_99 * 100, color="#c0392b", linestyle=":", linewidth=2.2, label=f"Parametric 99% VaR: {p_99:.2%}")

    axes[0].set_title("Return Distribution vs VaR Thresholds (Fat Tails vs Gaussian)", fontweight="bold")
    axes[0].set_xlabel("Daily Return (%)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Rolling 99% VaR thresholds and realized returns
    roll_h99 = rolling_var(p_rets, window=252, confidence=0.99, method="historical")
    roll_p99 = rolling_var(p_rets, window=252, confidence=0.99, method="parametric")
    common_idx = roll_h99.index
    sub_rets = p_rets.loc[common_idx]

    axes[1].plot(sub_rets.index, sub_rets * 100, color="#6c757d", alpha=0.5, linewidth=0.8, label="Realized Return")
    axes[1].plot(roll_h99.index, -roll_h99 * 100, color="#c0392b", linewidth=1.5, label="Rolling Historical 99% VaR")
    axes[1].plot(roll_p99.index, -roll_p99 * 100, color="#d4a017", linestyle="--", linewidth=1.5, label="Rolling Parametric 99% VaR")

    # Mark breaches
    h_breaches = sub_rets[sub_rets < -roll_h99]
    axes[1].scatter(h_breaches.index, h_breaches * 100, color="#900C3F", s=30, zorder=5, label=f"Hist 99% Breaches ({len(h_breaches)})")

    axes[1].set_title("1-Step-Ahead Rolling 99% VaR Backtest & Exceedances", fontweight="bold")
    axes[1].set_ylabel("Daily Return (%)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("docs/var_tail_risk_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison chart to {out_path}")


if __name__ == "__main__":
    run_tail_risk_analysis()
