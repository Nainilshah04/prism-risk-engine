# Design Specifications & Visual Language

Project: **Prism Risk Engine** (`prismrisk`)

---

## 1. Visual Language & Color Palette

To ensure consistency and professional aesthetics across all reports and interactive charts:

- **Asset Color Palette**:
  - **Equity**: `#1f77b4` (Classic institutional blue)
  - **Gold**: `#d4a017` (Deep amber)
  - **Debt / Liquid**: `#2a9d8f` (Calm teal)
  - **Benchmark**: `#6c757d` (Neutral grey)
  - **Aggregate Portfolio**: `#111827` (Near-black / deep navy)
  - **Loss / Drawdown / Stress**: `#c0392b` (Deep red)
  - **Breach Markers**: Red scatter dots (`#e74c3c`)

- **Typography & Formatting**:
  - UI Font: Clean modern sans-serif (`Inter`, `system-ui`).
  - Table Figures: Monospace / tabular numbers to maintain clean decimal alignment.
  - Return / Vol / Drawdown formatting: `XX.X%` (e.g. `12.3%`).
  - Ratios: 2 decimal places (e.g. `1.45`).
  - Dates: `YYYY-MM-DD`.
  - Losses & Negative Values: Always formatted in red with an explicit minus sign (`-12.4%`).

---

## 2. Factsheet Design (HTML)

The factsheet is generated via Jinja2 template (`templates/factsheet.html.j2`) and renders into a self-contained, print-friendly 2-page document:
- **Page 1 (Summary)**: Header (Period, Allocation, Rebalance schedule), Key Metrics Table (CAGR, Vol, Sharpe, Max DD, Calmar, Beta, TE, IR), Cumulative Return line chart vs benchmark, Drawdown area chart with top 3 drawdowns annotated, and Rolling 30/90 Volatility chart.
- **Page 2 (Risk & Scenarios)**: VaR/CVaR comparison table (Historical, Parametric, Monte Carlo @ 95/99%), VaR Breach timeline with Kupiec test pass/fail badge, Risk Contribution stacked bar chart, Strategy comparison table, Scenario replays & stress table, 2D sensitivity heatmap, and Assumptions & Limitations block.
- **Self-Contained Requirement**: All chart figures are encoded directly as Base64 PNGs.

---

## 3. Interactive Streamlit Dashboard

- **Sidebar**:
  - Portfolio configuration selector
  - Date range slider
  - Dynamic weight sliders with automatic re-normalization to 100%
  - Confidence level selector (95% / 99%)
  - Rolling window slider
  - Persistent "Assumptions & Methodology" expander
- **Pages**:
  1. `Overview`: High-level KPI cards, cumulative performance vs benchmark, asset allocation donut.
  2. `Risk`: Rolling & EWMA volatility, drawdown curve with worst-drawdown table, correlation heatmap, rolling pairwise correlation.
  3. `VaR`: Method comparison table, return histogram with VaR/CVaR cutoff lines, breach timeline, Kupiec POF test outcome.
  4. `Portfolio`: Strategy comparison (EW, Inv-Vol, Min-Var, Risk Parity, Max Sharpe), weights-over-time stacked area chart, turnover and transaction cost analysis, risk contribution bar chart.
  5. `Scenarios`: Named historical crisis replays, interactive shock sliders, 2D sensitivity surface.
  6. `Data Quality`: Missing data audit, longest gap, zero-return streaks, outlier return counts, effective overlapping sample window.
