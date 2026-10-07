# Project Rules & Contract

The following rules govern the implementation of **Prism Risk Engine**. If a rule is broken, fix the code, not the rule.

---

## A. Financial Correctness Rules (F)

- **R-F1 Adjusted prices only**: Use adjusted close so splits and dividends do not create artificial returns.
- **R-F2 No look-ahead**: Any weight, parameter, or threshold used at time $t$ may only depend on data up to $t$. Unit tests must include leakage tests verifying that perturbing data after $t$ has zero effect on decisions at $t$.
- **R-F3 Return types**: Portfolio aggregation across assets requires simple arithmetic returns. Time aggregation and statistics that assume additivity use log returns. Function names must explicitly state their return type (`simple_returns`, `log_returns`).
- **R-F4 Annualisation**: Standard 252 trading days/year. Volatility scales by $\sqrt{252}$. Returns scale via compounding (CAGR), never arithmetic multiplication unless explicitly labelled arithmetic.
- **R-F5 Sign conventions**: Returns and drawdowns are negative when losing. VaR and CVaR are reported as positive loss numbers. Docstrings and chart labels must explicitly declare this convention.
- **R-F6 Risk-free rate**: Use a configured proxy (flat annual rate or liquid-fund/T-bill return series). Always state the rate in outputs and convert to daily consistently.
- **R-F7 Common sample**: Compare assets and strategies only over identical date windows. Output the effective start/end date on every table and chart.
- **R-F8 Missing data**: Forward-fill only up to the configured max gap (default 3 days); beyond that, flag and exclude. Never silently fill missing values with zeros.
- **R-F9 Costs and drift**: Backtests must include transaction costs (in bps) and allow weights to drift between rebalancing periods.
- **R-F10 Estimation windows**: Covariance and VaR lookback windows must be configurable, never hard-coded.
- **R-F11 Assumptions visible**: Duration, convexity, sensitivities, risk-free proxy, and fill policies must be displayed in config and echoed in generated reports.
- **R-F12 No advice language**: Outputs describe historical risk. Avoid language like "should buy" or predictive performance guarantees.

---

## B. Code Rules (C)

- **R-C1 Pure core**: Core analytics packages contain no I/O, no plotting, no console prints, and no Streamlit imports.
- **R-C2 Strictly typed**: Type annotations on every public function; `mypy` clean.
- **R-C3 Descriptive docstrings**: State formula/reference, inputs with shapes, output, and sign conventions.
- **R-C4 Small units**: Single responsibility functions; source files kept under ~300 lines; functions kept concise.
- **R-C5 No magic numbers**: Constants (252, 0.94, 0.95) must reside in named constants or configuration objects.
- **R-C6 Vectorization**: Favor vectorized NumPy/Pandas operations over raw loops, except where an explicit loop clarifies sequential state (such as the walk-forward rebalancing engine).
- **R-C7 Deterministic**: Monte Carlo simulations and randomized algorithms must accept an explicit random seed from configuration.
- **R-C8 Explicit errors**: Raise clear exceptions on invalid inputs (non-sorted indexes, weights not summing to 1.0, unexpected NaNs). No silent fixes.

---

## C. Testing Rules (T)

- **R-T1 Known-answer tests**: Every metric must be verified against hand-computed numeric examples.
- **R-T2 Invariant tests**: Verify mathematical invariants ($\sum w_i = 1$, $\text{VaR}_{99\%} \ge \text{VaR}_{95\%}$, $\text{CVaR} \ge \text{VaR}$, correlation matrix symmetric with unit diagonal, drawdown $\le 0$).
- **R-T3 Degenerate cases**: Flat prices, single asset, very short series, all-NaN columns.
- **R-T4 Leakage test**: Perturb future series; past backtest results must remain identical.
- **R-T5 Offline**: Unit tests must never initiate network calls.
- **R-T6 Coverage gate**: Minimum 85% test coverage enforced by CI.
