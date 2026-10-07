# System Architecture

Project: **Prism Risk Engine** (`prismrisk`)  
Author: Nainil Shah

---

## 1. Core Principles

1. **Pure Core, Thin Edges**: Analytical modules (`metrics/`, `risk/`, `portfolio/`, `scenarios/`) are pure, deterministic functions operating on pandas and numpy structures. They contain zero I/O, no plotting libraries, no printing, and no Streamlit dependencies.
2. **One-Way Dependency Flow**: UI and CLI interfaces import the pipeline orchestration layer; the orchestration layer imports the domain analytics and data layers; core analytics never import presentation code.
3. **Configuration Over Code**: Tickers, risk-free rate, confidence intervals, rebalance horizons, caps, and sensitivities are fully declared in YAML and validated via Pydantic models.
4. **Unified Pipeline**: The Typer CLI, HTML factsheet generator, and Streamlit app all invoke the identical `pipeline.run_analysis(config)` workflow.
5. **100% Offline Testability**: Unit and integration tests run entirely against committed Parquet snapshots and synthetic fixtures without external network calls.

---

## 2. Layered Architecture Diagram

```
+--------------------------------------------------------------+
|                         INTERFACES                           |
|       CLI (Typer)     |   Streamlit App   |   HTML Factsheet |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
|                        ORCHESTRATION                         |
|         pipeline.run_analysis(config) -> Results             |
+--------------------------------------------------------------+
       |                 |                  |               |
       v                 v                  v               v
+-------------+   +--------------+   +--------------+  +---------------+
|   metrics   |   |     risk     |   |  portfolio   |  |   scenarios   |
| returns     |   | var / cvar   |   | optimisers   |  | historical    |
| vol, dd     |   | backtest     |   | backtest     |  | shocks        |
| ratios      |   | kupiec POF   |   | risk contrib |  | sensitivities |
+-------------+   +--------------+   +--------------+  +---------------+
       \                 |                  |               /
        v                v                  v              v
+--------------------------------------------------------------+
|                         DATA LAYER                           |
|      Providers (yfinance) -> Parquet Cache -> Snapshot       |
|      Calendar Alignment & Missing-Data Fill Policy            |
|      Data Quality (DQ) Diagnostic Report                     |
+--------------------------------------------------------------+
```

---

## 3. Data Flow

1. `Config.load("configs/demo.yaml")` validates user inputs, portfolio weights, and options into typed Pydantic structures.
2. Data provider fetches adjusted daily close prices for all tickers or loads from the local Parquet cache (`data/raw/` or `data/snapshot/`).
3. `calendar.align()` harmonizes trading calendars across international exchanges, applying the configured forward-fill policy up to `max_ffill_days`.
4. `quality.generate_report()` evaluates data integrity (missing percentage, max gap, outliers, zero-return days).
5. Analytics modules calculate returns, volatility series, drawdowns, tail risk, and walk-forward strategy backtests.
6. `Results` dataclass bundles all computed DataFrames, series, and diagnostics into a single clean object.
7. Frontends (CLI, HTML factsheet, Streamlit dashboard) receive `Results` and render views with zero internal business logic.
