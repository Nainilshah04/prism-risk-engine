# Mathematical Reference and Methodology Guide

Project: **Prism Risk Engine** (`prismrisk`)  
Author: Nainil Shah

---

## 1. Returns and Compounding (Rule R-F3, Rule R-F4)

### 1.1 Simple Arithmetic Return
Used strictly for cross-sectional portfolio aggregation:
$$r_t = \frac{P_t}{P_{t-1}} - 1$$
Portfolio return at time $t$ for weights $w$:
$$r_{p, t} = \sum_{i=1}^N w_i \cdot r_{i, t}$$

### 1.2 Logarithmic Return
Used for multi-period time aggregation and continuous variance modelling:
$$l_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(1 + r_t)$$

### 1.3 Compound Annual Growth Rate (CAGR)
Computed via compound multi-period terminal wealth, assuming 252 trading days/year:
$$\text{CAGR} = \left(\prod_{t=1}^T (1 + r_{p, t})\right)^{\frac{252}{T}} - 1$$

---

## 2. Volatility and Dispersion (FR-M2)

### 2.1 Annualized Volatility
$$\sigma_{\text{ann}} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (r_t - \bar{r})^2} \times \sqrt{252}$$

### 2.2 Exponentially Weighted Moving Average (EWMA) Volatility
Following RiskMetrics specification with decay factor $\lambda = 0.94$:
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2$$
$$\sigma_{\text{EWMA}, t} = \sqrt{\sigma_t^2 \times 252}$$

---

## 3. Drawdowns (FR-M3, Rule R-F5)

Wealth index $V_t = \prod_{s=1}^t (1 + r_s)$ with $V_0 = 1.0$.  
Running peak $M_t = \max_{s \le t} V_s$.
$$DD_t = \frac{V_t}{M_t} - 1 \quad (\le 0)$$
- **Max Drawdown**: $\min_{t} DD_t$
- **Peak Date**: Date when $M_t$ was established prior to the lowest trough.
- **Trough Date**: Date when $\min DD_t$ occurred.
- **Recovery Date**: First date $t > \text{trough}$ where $V_t \ge M_{\text{peak}}$, or None if unrecovered.

---

## 4. Risk-Adjusted Ratios (FR-M4, FR-M5)

- **Daily Risk-Free Rate**: $r_{f, d} = (1 + r_{f, \text{annual}})^{1/252} - 1$
- **Sharpe Ratio**:
$$\text{Sharpe} = \frac{\text{mean}(r - r_{f, d})}{\text{std}(r - r_{f, d})} \times \sqrt{252}$$
- **Downside Deviation (Sortino)**:
$$\delta_{\text{down}} = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, r_t - r_{f, d})^2} \times \sqrt{252}$$
$$\text{Sortino} = \frac{\text{mean}(r - r_{f, d}) \times 252}{\delta_{\text{down}}}$$
- **Calmar Ratio**:
$$\text{Calmar} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}$$
- **Beta vs Benchmark**:
$$\beta = \frac{\text{Cov}(r_p, r_b)}{\text{Var}(r_b)}$$
- **Tracking Error & Information Ratio**:
$$\text{TE} = \text{std}(r_p - r_b) \times \sqrt{252}$$
$$\text{IR} = \frac{\text{mean}(r_p - r_b) \times 252}{\text{TE}}$$

---

## 5. Tail Risk (VaR & CVaR) (FR-R1 to FR-R4)

All VaR and CVaR measures are reported as **positive loss fractions** (e.g. $0.025 = 2.5\%$ loss).

### 5.1 Historical VaR & CVaR
For confidence $\alpha \in \{0.95, 0.99\}$:
$$\text{VaR}_\alpha = -Q_{1-\alpha}(r)$$
$$\text{CVaR}_\alpha = -\mathbb{E}[r \mid r \le - \text{VaR}_\alpha]$$

### 5.2 Parametric (Normal) VaR & CVaR
$$\text{VaR}_\alpha = -(\mu + z_{1-\alpha} \sigma)$$
$$\text{CVaR}_\alpha = -\left(\mu - \sigma \frac{\phi(z_{1-\alpha})}{1-\alpha}\right)$$

### 5.3 Monte Carlo VaR
Simulate $N$ paths from $\mathcal{N}(\mu, \Sigma)$ via Cholesky decomposition $\Sigma = L L^T$.

### 5.4 Kupiec Proportion of Failures (POF) Likelihood Ratio Test
Tests null hypothesis $H_0: p = 1 - \alpha$ against realized failure rate $\hat{p} = \frac{x}{N}$:
$$LR_{\text{POF}} = -2 \ln \left[ \frac{(1-p)^{N-x} p^x}{(1 - x/N)^{N-x} (x/N)^x} \right] \sim \chi^2(1)$$

---

## 6. Portfolio Optimization & Risk Contributions (FR-P1 to FR-P4)

- **Ledoit-Wolf Shrinkage**:
$$\hat{\Sigma}_{\text{LW}} = \delta F + (1 - \delta) S$$
where $S$ is sample covariance and $F$ is a single-index / constant-correlation target.
- **Marginal Risk Contribution (MRC)**:
$$MRC_i = \frac{(\Sigma w)_i}{\sigma_p}$$
- **Percentage Risk Contribution**:
$$\%RC_i = \frac{w_i \cdot MRC_i}{\sigma_p} = \frac{w_i (\Sigma w)_i}{w^T \Sigma w}, \quad \sum_{i=1}^N \%RC_i = 100\%$$

---

## 7. Stress Testing & Bond Approximation (FR-S2, FR-S3)

For fixed-income / debt assets with modified duration $D$ and convexity $C$, under yield change $\Delta y$:
$$\frac{\Delta P}{P} \approx -D \Delta y + \frac{1}{2} C (\Delta y)^2$$
