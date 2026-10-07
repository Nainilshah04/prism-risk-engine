# Implementation Phases Roadmap

Project: **Prism Risk Engine** (`prismrisk`)

---

### Phase 0 — Foundations
- **Objective**: Clean automated skeleton, tooling, typing, config schema, CI, documentation.
- **Deliverables**: `pyproject.toml`, directory layout, `.gitignore`, `Makefile`, `.github/workflows/ci.yml`, `docs/`, `Config` Pydantic models, `models.py` contracts, `tests/fixtures/synthetic_data.py`, unit tests with > 85% coverage.
- **Status**: Completed.

---

### Phase 1 — Data Layer & Quality Engine
- **Objective**: Ingest multi-asset market data, cache locally, calendar align, and detect anomalies.
- **Deliverables**:
  - `src/prismrisk/data/providers.py` (yfinance interface)
  - `src/prismrisk/data/cache.py` (Parquet read/write with refresh toggle)
  - `src/prismrisk/data/calendar.py` (Trading calendar alignment + forward-fill policy)
  - `src/prismrisk/data/quality.py` (Data quality diagnostics: missingness, gaps, outliers, zero-returns)
  - Unit and snapshot integration tests.

---

### Phase 2 — Core Return & Risk Metrics Engine
- **Objective**: Pure analytical calculation of standard investment performance and dispersion statistics.
- **Deliverables**:
  - `src/prismrisk/metrics/returns.py` (simple/log returns, cumulative return, CAGR)
  - `src/prismrisk/metrics/volatility.py` (annualized vol, rolling vol, EWMA vol)
  - `src/prismrisk/metrics/drawdown.py` (drawdown series, max drawdown, peak/trough/recovery dates)
  - `src/prismrisk/metrics/ratios.py` (Sharpe, Sortino, Calmar, beta, tracking error, IR)
  - `src/prismrisk/metrics/correlation.py` (static and rolling correlation)
  - Hand-computed known-answer unit tests for every metric.

---

### Phase 3 — Tail Risk & VaR Backtesting
- **Objective**: Comprehensive tail-risk modeling and statistical breach validation.
- **Deliverables**:
  - `src/prismrisk/risk/var.py` (Historical, Parametric Normal, and Monte Carlo VaR/CVaR)
  - `src/prismrisk/risk/backtest.py` (Rolling VaR breaches, Kupiec POF test with Chi-squared p-values)
  - `src/prismrisk/risk/contribution.py` (Marginal and percentage risk contributions per asset)
  - Unit tests with invariant checks.

---

### Phase 4 — Portfolio Construction & Walk-Forward Engine
- **Objective**: Lookahead-free walk-forward portfolio optimization with realistic execution friction.
- **Deliverables**:
  - `src/prismrisk/portfolio/covariance.py` (Sample covariance and Ledoit-Wolf shrinkage)
  - `src/prismrisk/portfolio/optimisers.py` (Equal weight, inverse vol, minimum variance, risk parity, max Sharpe)
  - `src/prismrisk/portfolio/backtest.py` (Walk-forward simulator with rebalancing intervals, weight drift, and transaction costs)
  - Anti-leakage tests verifying past invariance under future data shifts.

---

### Phase 5 — Stress Testing & Scenario Analysis
- **Objective**: Macro stress tests, historical replays, and sensitivity surfaces.
- **Deliverables**:
  - `src/prismrisk/scenarios/historical.py` (COVID crash, 2022 rate hikes replay)
  - `src/prismrisk/scenarios/shocks.py` (Hypothetical factor shocks + bond duration/convexity model)
  - `src/prismrisk/scenarios/sensitivity.py` (2D sensitivity grids)
  - Unit tests for scenario shifts and sensitivity grids.

---

### Phase 6 — Orchestration Pipeline, Factsheet & CLI
- **Objective**: Unified execution pipeline, HTML factsheet generation, and Typer CLI.
- **Deliverables**:
  - `src/prismrisk/pipeline.py` (`run_analysis(config) -> Results`)
  - `src/prismrisk/reporting/charts.py` (Matplotlib publication figure generators)
  - `src/prismrisk/reporting/factsheet.py` (Self-contained HTML factsheet with base64 embedded charts)
  - `src/prismrisk/cli.py` (`prismrisk run`, `report`, `refresh-data`, `app`)

---

### Phase 7 — Interactive Streamlit Dashboard
- **Objective**: Responsive, multi-page quantitative dashboard matching visual design language.
- **Deliverables**:
  - `src/prismrisk/app/streamlit_app.py`
  - Multi-page views: Overview, Risk Analysis, VaR & Tail Risk, Portfolio Strategies, Scenarios & Stress, Data Quality.
  - Interactive sliders, date selection, risk-free adjustment, and dynamic Plotly visualizations.
