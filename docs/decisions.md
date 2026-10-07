# Architecture Decision Records (ADRs)

Project: **Prism Risk Engine** (`prismrisk`)  
Author: Nainil Shah

---

### ADR-1: Return Types for Multi-Period vs Cross-Sectional Aggregation
- **Status**: Accepted
- **Context**: Portfolio aggregation across assets requires linear combination ($r_p = \sum w_i r_i$), which holds strictly for simple arithmetic returns ($r_t = P_t / P_{t-1} - 1$). Conversely, multi-period compounding and time aggregation assume log returns ($\ln(1+r)$) to ensure time additivity.
- **Decision**: All asset/portfolio aggregation functions will take simple returns. Long-horizon statistical compounding and time series models will explicitly convert to or state log returns. Function signatures must never conflate them (`simple_returns`, `log_returns`).

---

### ADR-2: Long-Only & Fully-Invested Baseline Constraints
- **Status**: Accepted
- **Context**: Retail and benchmark UCITS/AIF-style multi-asset portfolios generally forbid naked short selling and uncollateralized leverage.
- **Decision**: Default portfolio weights are constrained to $w_i \ge 0$ and $\sum w_i = 1.0$. Caps (e.g. $w_i \le 0.70$) are configurable via YAML.

---

### ADR-3: Covariance Shrinkage (Ledoit-Wolf Default)
- **Status**: Accepted
- **Context**: Sample covariance matrices are ill-conditioned and suffer from severe estimation noise when $N$ (number of assets) is comparable to $T$ (observations), leading to extreme and unstable weights in minimum-variance or mean-variance optimizations.
- **Decision**: Ledoit-Wolf shrinkage towards constant correlation is the default covariance estimator. Sample covariance is preserved strictly as a baseline benchmark.

---

### ADR-4: Parquet Snapshot Committed for Deterministic Offline Runs
- **Status**: Accepted
- **Context**: Dependency on live external network calls (e.g., Yahoo Finance) in continuous integration (CI) or reviewer demos causes intermittent failures due to throttling or schema changes.
- **Decision**: A calibrated parquet snapshot is committed to `data/snapshot/market_data_snapshot.parquet`. Unit and regression tests never touch the network (Rule R-T5).

---

### ADR-5: Pure Analytical Core, Thin Edge Interfaces
- **Status**: Accepted
- **Context**: Embedding business formulas inside Streamlit callbacks or CLI scripts creates non-testable code and regression drift.
- **Decision**: Modules under `metrics/`, `risk/`, `portfolio/`, `scenarios/` are pure mathematical functions on pandas/numpy objects with zero side effects or UI imports. Streamlit, CLI, and HTML reports only consume the unified `Results` dataclass.

---

### ADR-6: Pure Quantitative Risk & Stress Estimation (No ML Price Prediction)
- **Status**: Accepted
- **Context**: Financial markets are non-stationary with low signal-to-noise ratios. Student and hobbyist projects often employ black-box regressors claiming to predict stock prices.
- **Decision**: Focus strictly on tail risk, drawdown dynamics, factor sensitivities, and robust portfolio construction. No speculative buy/sell signals.

---

### ADR-7: Verified Asset Universe & Locked Sample Window
- **Status**: Accepted
- **Context**: Multi-asset cross-sectional analysis requires verifying live availability, ticker symbols, and history length across Indian and global asset classes.
- **Verification Findings**:
  - `equity` (`NIFTYBEES.NS`): Verified available since `2009-01-02` (4,300+ trading days).
  - `gold` (`GOLDBEES.NS`): Verified available since `2009-01-02`.
  - `debt` (`LIQUIDBEES.NS`): Verified available since `2009-01-02`. Daily fractional unit dividend model produces flat ~1,000 INR price on Yahoo Finance; engine provides `synthetic_yield` compounding option to reflect true debt accrual.
  - `benchmark` (`^NSEI`): Verified available since `2007-09-17`.
  - `fx` (`USDINR=X`): Verified available since `2003-12-01`.
  - `global_benchmark` (`^GSPC`): Verified available since `1927-12-30`.
- **Decision**: Lock the primary 6-asset universe above. Baseline effective sample window is locked from `2015-01-01` to latest available (`10+ years`), ensuring a 100% overlapping trading history across all asset classes with zero survivorship or truncation bias.
