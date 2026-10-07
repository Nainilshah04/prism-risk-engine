# Product Requirements Document (PRD)

**Project:** Prism Risk Engine (`prismrisk`)  
**Repo:** `https://github.com/Nainilshah04/prism-risk-engine`  
**Owner:** Nainil Shah  
**Status:** In Progress (Phase 0 Complete)

---

## 1. Problem Statement
Investment teams managing multi-asset portfolios need rigorous answers to core risk questions:
- *How much can the portfolio lose in extreme tails?*
- *How severe can a drawdown get, and how long does it take to recover?*
- *What happens under named historical crises and hypothetical macro shocks?*
- *How do allocation strategies compare when rebalancing and transaction frictions are accounted for without lookahead bias?*

Most student or amateur projects stop at plotting price charts and calculating moving averages. Prism Risk Engine delivers an institutional-grade, tested, reproducible quantitative risk engine that computes standard fund metrics, estimates and backtests tail risk (VaR/CVaR), evaluates lookahead-safe walk-forward allocation strategies, and renders results through a print-ready HTML factsheet and an interactive Streamlit application.

---

## 2. Goals
1. Ingest and clean real multi-asset market data (equity, gold, debt/liquid, FX, global benchmark).
2. Compute standard fund and risk metrics correctly (CAGR, annual vol, rolling/EWMA vol, drawdowns, Sharpe, Sortino, Calmar, beta, tracking error, IR).
3. Estimate and rigorously backtest tail risk (Historical, Parametric, Monte Carlo VaR/CVaR; Kupiec POF test).
4. Construct and backtest multi-asset portfolio strategies (Equal Weight, Inverse Vol, Min Variance, Risk Parity, Max Sharpe) with Ledoit-Wolf shrinkage, periodic rebalancing, drift, and transaction costs.
5. Execute historical crisis replays (e.g. 2020 COVID crash, 2022 rate hike phase) and hypothetical macro shocks (equity %, gold %, rates bp with duration/convexity approximation).
6. Provide an auto-generated HTML factsheet and an interactive Streamlit dashboard.
7. Maintain software engineering discipline: 100% offline testability, >= 85% test coverage, strict typing, CI pipeline, reproducible data snapshot.

---

## 3. Non-Goals (Explicitly Out of Scope)
- No price forecasting or trading signals (no "buy/sell" output).
- No real-time intraday feeds or broker execution.
- No technical indicator dashboards (RSI, MACD, etc.).
- No financial or investment advice language.

---

## 4. Functional Requirements Summary

- **Data (FR-D)**:
  - FR-D1: Download adjusted daily prices via `yfinance`.
  - FR-D2: Parquet cache with `--refresh` flag; committed snapshot for offline execution.
  - FR-D3: Calendar alignment with forward-fill policy up to max gap (default 3 days).
  - FR-D4: Data quality report (gaps, zero-return days, outliers > N sigma, history start dates).
- **Metrics (FR-M)**:
  - FR-M1: Simple & log returns; cumulative return; CAGR.
  - FR-M2: Annualised volatility; rolling volatility (30/90 days); EWMA volatility ($\lambda = 0.94$).
  - FR-M3: Drawdown series, max drawdown, peak/trough/recovery dates, duration; Calmar ratio.
  - FR-M4: Sharpe and Sortino ratios with configured risk-free proxy.
  - FR-M5: Beta, tracking error, information ratio vs benchmark.
  - FR-M6: Correlation matrix and rolling pairwise correlation.
- **Tail Risk (FR-R)**:
  - FR-R1: Historical VaR / CVaR (95% and 99%).
  - FR-R2: Parametric (normal) VaR / CVaR.
  - FR-R3: Monte Carlo VaR / CVaR using estimated covariance and Cholesky decomposition.
  - FR-R4: Rolling VaR breach backtesting and Kupiec POF hypothesis test.
- **Portfolio (FR-P)**:
  - FR-P1: Strategies: equal weight, inverse volatility, minimum variance, risk parity, max Sharpe (long-only).
  - FR-P2: Covariance estimation with Ledoit-Wolf shrinkage.
  - FR-P3: Lookahead-safe walk-forward backtest with periodic rebalance, weight drift, and transaction costs.
  - FR-P4: Marginal and percentage risk contribution per asset.
  - FR-P5: Strategy comparison table (return, vol, max DD, Sharpe, turnover).
- **Scenarios (FR-S)**:
  - FR-S1: Historical replay of named crisis windows applied to current weights.
  - FR-S2: Hypothetical shocks across equities, FX, and rates mapped via asset sensitivities.
  - FR-S3: Bond/liquid asset shock via modified duration and convexity approximation.
  - FR-S4: 2D Sensitivity grid (loss across multi-factor shock combinations).
- **Reporting and UI (FR-U)**:
  - FR-U1: Self-contained HTML factsheet (Jinja2 template + embedded base64 figures).
  - FR-U2: Streamlit multi-page dashboard.
  - FR-U3: Typer CLI tool (`prismrisk run`, `report`, `refresh-data`, `app`).
