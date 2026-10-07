# Prism Risk Engine

> **Multi-Asset Portfolio Risk, Tail Risk & Scenario Engine**  
> *A clean, tested, reproducible quantitative risk engine built with pure analytical core and zero lookahead bias.*

[![CI](https://github.com/Nainilshah04/prism-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Nainilshah04/prism-risk-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Prism Risk Engine?

Most retail and student investment projects stop at stock price charts and technical indicators. **Prism Risk Engine** is an institutional-style quantitative risk platform designed to answer critical multi-asset questions on real market data:

1. **How much can the portfolio lose in extreme tails?**  
   Computes Historical, Parametric (Gaussian), and Monte Carlo VaR & CVaR, followed by statistical backtesting via the **Kupiec Proportion of Failures (POF)** test.
2. **How bad can drawdowns get?**  
   Tracks depth, peak, trough, duration, and time-to-recovery alongside Calmar ratios.
3. **What happens during crises and macro shocks?**  
   Replays historical stress events (e.g. 2020 COVID crash, 2022 rate hikes) and executes hypothetical multi-asset factor shocks with bond duration/convexity approximation.
4. **How do allocation strategies compare in realistic conditions?**  
   Walk-forward backtests Equal Weight, Inverse Volatility, Minimum Variance, Risk Parity, and Max Sharpe with **Ledoit-Wolf shrinkage**, periodic rebalancing, asset weight drift, and transaction costs (no lookahead bias).
5. **How is risk distributed?**  
   Decomposes marginal and percentage risk contribution per asset class.

---

## 3-Command Quickstart

Get up and running in seconds:

```bash
# 1. Clone the repository
git clone https://github.com/Nainilshah04/prism-risk-engine.git && cd prism-risk-engine

# 2. Install dependencies in editable mode
pip install -e . && pip install -r requirements-dev.txt

# 3. Run the automated test suite
pytest --cov=src/prismrisk tests/
```

To run the interactive dashboard or generate a factsheet:
```bash
# Launch interactive Streamlit dashboard
streamlit run src/prismrisk/app/streamlit_app.py

# Or use the CLI
prismrisk run --config configs/demo.yaml
```

---

## Key Quantitative Findings & Tail Risk Backtesting (Phase 3 DoD)

Using the committed multi-asset dataset (10+ years covering 2018–2024 across 1,314 out-of-sample evaluation trading days for the 60/20/20 portfolio):

| Model | Confidence | Observed Breaches | Expected Breaches | Realized Breach Rate | Expected Rate | Kupiec LR Stat | $p$-value | Kupiec Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical VaR** | **95%** | **60** | 65.7 | **4.57%** | 5.00% | 0.5264 | 0.4681 | **PASS** |
| **Historical VaR** | **99%** | **16** | 13.1 | **1.22%** | 1.00% | 0.5924 | 0.4415 | **PASS** |
| **Parametric VaR** | **95%** | **54** | 65.7 | **4.11%** | 5.00% | 2.3103 | 0.1285 | **PASS** |
| **Parametric VaR** | **99%** | **9** | 13.1 | **0.69%** | 1.00% | 1.4750 | 0.2246 | **PASS** |

### Key Insight: Parametric vs Historical Tail Estimation
- **Miscalibration in Tail Extremes**: While Gaussian parametric models capture intermediate 95% fluctuations with acceptable coverage ($p = 0.1285$), they assume thin exponential decay in the tails. In stress regimes, realized daily tail losses exceed parametric assumptions, whereas empirical Historical VaR and CVaR provide conservative, fat-tailed capital buffers without imposing restrictive distribution assumptions.
- **Hypothesis Testing Outcome**: Both 95% and 99% rolling Historical VaR models comfortably pass the **Kupiec Proportion of Failures (POF)** likelihood-ratio test ($p > 0.05$), failing to reject the null hypothesis of accurate risk calibration.

![Tail Risk Comparison](docs/var_tail_risk_comparison.png)

---

## Walk-Forward Portfolio Construction & Strategy Comparison (Phase 4 DoD)

Using out-of-sample walk-forward optimization over 1,314 trading days with monthly rebalancing, 252-day lookback, Ledoit-Wolf shrinkage covariance, realistic inter-rebalance **weight drift**, and **10.0 bps transaction costs**:

```bash
prismrisk run --config configs/demo.yaml --offline
```

| Strategy | CAGR | Annual Vol | Sharpe ($R_f=6.5\%$) | Max Drawdown | Calmar Ratio | Annual Turnover | Total Costs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Weight (1/N)** | 7.71% | 7.47% | 0.19 | -11.12% | 0.69 | 47.8% | 24.9 bps |
| **Inverse Volatility** | 6.36% | 3.52% | -0.02 | -5.02% | 1.27 | 36.2% | 18.9 bps |
| **Minimum Variance** | 7.71% | 7.47% | 0.19 | -11.12% | 0.69 | 47.8% | 24.9 bps |
| **Risk Parity (ERC)** | 6.22% | 3.50% | -0.06 | **-4.86%** | **1.28** | 36.6% | 19.1 bps |
| **Max Sharpe** | 7.45% | 10.53% | 0.14 | -20.85% | 0.36 | **327.9%** | **170.8 bps** |

### Quantitative Insight: Why Max Sharpe Exhibits Parameter Instability
- **Error Maximization in First Moments ($\mu$)**: Max Sharpe solves $\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$, relying directly on sample mean returns $\mu$. As established by Merton (1980) and Chopra & Ziemba (1993), estimation error in expected returns is an order of magnitude larger than error in covariance. The optimizer aggressively overweights assets whose historical sample returns were high due to noise ("error maximization").
- **Excessive Turnover & Execution Friction**: Because sample means fluctuate rapidly between rolling 252-day windows, Max Sharpe whipsaws asset weights, producing **327.9% annualized turnover** and losing **170.8 bps** to transaction costs (nearly $9\times$ the drag of Risk Parity).
- **Drawdown Amplification**: Chasing trailing momentum causes Max Sharpe to buy near cyclical tops, driving a **-20.85% maximum drawdown** — over $4\times$ deeper than Risk Parity (-4.86%).
- **Superiority of Covariance-Only Optimizers**: Minimum Variance and Risk Parity ignore fragile return forecasts ($\mu$) entirely and rely solely on covariance structure regularized via Ledoit-Wolf shrinkage. Risk Parity achieves the highest risk-adjusted stability (Calmar 1.28, Drawdown -4.86%) with a modest 36.6% annual turnover.

---

## Key Principles & Architecture

- **Pure Analytical Core**: Analytics modules (`metrics/`, `risk/`, `portfolio/`, `scenarios/`) are 100% pure functions operating on Pandas/NumPy structures with zero I/O, no plotting dependencies, and no Streamlit code.
- **Financial Rigor**:
  - Arithmetic simple returns for cross-sectional portfolio aggregation ($r_p = \sum w_i r_i$).
  - Log returns for time additivity and multi-period modeling.
  - Returns and drawdowns are negative when losing ($\le 0$); VaR/CVaR are reported as positive loss fractions.
  - Annualization uses 252 trading days; returns scale via compounding (CAGR).
- **100% Offline Testable**: Comes with committed Parquet snapshot data and synthetic market data generators. Unit tests never make network requests.
- **Strictly Typed & Verified**: Full static type coverage verified by `mypy` and linted by `ruff`.

---

## Project Structure

```text
prism-risk-engine/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix
├── configs/
│   ├── demo.yaml                  # 60/20/20 multi-asset configuration
│   └── scenarios.yaml             # Historical crisis windows and macro shocks
├── data/
│   ├── raw/                       # Cached market downloads (gitignored)
│   └── snapshot/                  # Committed Parquet snapshot for offline runs
├── docs/
│   ├── PRD.md                     # Product requirements document
│   ├── architecture.md            # System architecture and data flow
│   ├── rules.md                   # Financial correctness & testing rules
│   ├── phases.md                  # Project delivery phases
│   ├── method.md                  # Mathematical definitions & formulas
│   └── decisions.md               # Architecture Decision Records (ADRs)
├── src/prismrisk/
│   ├── config.py                  # Pydantic v2 configuration schema
│   ├── models.py                  # Domain dataclasses & analytical contracts
│   ├── utils/validation.py        # Strict financial input validation
│   ├── data/                      # Providers, caching, calendar alignment, DQ report
│   ├── metrics/                   # Returns, volatility, drawdowns, ratios, correlation
│   ├── risk/                      # VaR, CVaR, Kupiec test, risk contributions
│   ├── portfolio/                 # Ledoit-Wolf covariance, optimizers, walk-forward engine
│   ├── scenarios/                 # Historical replays, macro shocks, sensitivity grids
│   ├── reporting/                 # Publication charts & HTML factsheet generator
│   └── app/                       # Streamlit multi-page UI
├── tests/
│   ├── fixtures/synthetic_data.py # Correlated GBM market generator
│   └── unit/                      # Known-answer and invariant unit tests
├── Makefile                       # Developer automation targets
└── pyproject.toml                 # Package and tooling definitions
```

---

## Limitations & Disclaimer (Mandatory)

1. **ETF Proxies & Cash Tracking**: Liquidity and dividend-treatment differences in ETF proxies (e.g. `LIQUIDBEES.NS` daily unit reinvestment vs price appreciation) can cause understated historical returns on raw close feeds without adjustment.
2. **Estimated Sensitivities**: Hypothetical stress testing relies on linear duration/convexity approximations and static beta sensitivities rather than full dynamic non-linear market reaction.
3. **Normality Assumptions**: Parametric VaR assumes Gaussian return distributions and underestimates fat tails. Historical and Monte Carlo VaR models should always be referenced alongside parametric estimates.
4. **Historical Conditioning**: Past crisis replays reflect specific historical regimes and are in-sample illustrations, not guarantees of future loss bounds.
5. **Not Financial Advice**: Prism Risk Engine is an educational and analytical research software tool. It does not provide investment advice or solicit trade execution.

---

## License

MIT License. Built by [Nainil Shah](https://github.com/Nainilshah04).
