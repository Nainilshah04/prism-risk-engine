"""Rolling Value-at-Risk backtesting and Kupiec POF hypothesis test.

Enforces:
- Rule R-F2: No look-ahead. VaR at time t depends strictly on data up to t-1.
- Rule R-F5: Loss exceeds VaR when return < -VaR.
- Rule FR-R4: Rolling VaR, breach counting, Kupiec proportion-of-failures test.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from prismrisk.models import KupiecResult
from prismrisk.risk.var import historical_var, parametric_var
from prismrisk.utils.validation import validate_confidence, validate_returns


def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Compute 1-step-ahead out-of-sample rolling VaR series without lookahead bias.

    At date t, the VaR threshold is estimated strictly over the lookback window
    ending at t-1 (returns[t - window : t]), preventing lookahead leakage.

    Args:
        returns: Daily simple returns Series.
        window: Estimation window length in trading days (default: 252).
        confidence: Confidence level in (0, 1).
        method: Estimation method ('historical' or 'parametric').

    Returns:
        Series of positive VaR loss thresholds aligned with date t.
    """
    validate_returns(returns)
    validate_confidence(confidence)
    if window < 30:
        raise ValueError(f"Estimation window must be >= 30 days, got {window}")
    if len(returns) <= window:
        raise ValueError(
            f"Series length ({len(returns)}) must be greater than estimation window ({window})"
        )

    # Shift returns by 1 to strictly enforce zero lookahead (data up to t-1)
    shifted = returns.shift(1)

    if method == "historical":
        q = shifted.rolling(window=window).quantile(1.0 - confidence)
        roll = (-q).dropna()
    elif method == "parametric":
        mu = shifted.rolling(window=window).mean()
        sigma = shifted.rolling(window=window).std(ddof=1)
        z = norm.ppf(confidence)
        roll = (z * sigma - mu).dropna()
    else:
        raise ValueError(f"Unknown method '{method}', must be 'historical' or 'parametric'")

    roll = roll.clip(lower=0.0)
    roll.name = f"var_{method}_{int(confidence*100)}"
    return roll


def count_breaches(
    realized_returns: pd.Series,
    rolling_var_series: pd.Series,
) -> pd.Series:
    """Identify VaR exceedances (breaches) where realized daily loss exceeds estimated VaR.

    Formula:
        breach_t = 1 if r_t < -VaR_t else 0

    Args:
        realized_returns: Realized daily returns Series.
        rolling_var_series: 1-step-ahead positive VaR threshold Series.

    Returns:
        Integer Series (1 for breach, 0 for non-breach) aligned to common dates.
    """
    aligned = pd.concat([realized_returns, rolling_var_series], axis=1, join="inner").dropna()
    r = aligned.iloc[:, 0]
    var_th = aligned.iloc[:, 1]

    # Breach when return is less than negative VaR threshold
    breaches = (r < -var_th).astype(int)
    breaches.name = "breach"
    return breaches


def kupiec_pof(
    breaches: int,
    n_obs: int,
    confidence: float,
) -> KupiecResult:
    """Execute Kupiec Proportion of Failures (POF) Likelihood Ratio test.

    Tests the null hypothesis H0: failure probability p = 1 - confidence
    against empirical breach rate p_hat = x / N.

    Formula:
        LR_POF = 2 * [ x * ln(p_hat / p) + (N - x) * ln((1 - p_hat) / (1 - p)) ] ~ chi2(1)

    Args:
        breaches: Number of observed losses exceeding VaR (x).
        n_obs: Total observations tested (N).
        confidence: VaR confidence level (e.g. 0.95 or 0.99).

    Returns:
        KupiecResult containing test statistic, p-value, and pass/reject decision.
    """
    validate_confidence(confidence)
    if n_obs <= 0:
        raise ValueError(f"Observations must be > 0, got {n_obs}")
    if not (0 <= breaches <= n_obs):
        raise ValueError(f"Breaches must be between 0 and {n_obs}, got {breaches}")

    p = 1.0 - confidence
    expected = float(n_obs * p)
    p_hat = breaches / n_obs

    # Handle boundary conditions for log-likelihood
    if breaches == 0:
        # lim_{x->0} x * ln(x/N) = 0
        lr_stat = float(-2.0 * n_obs * np.log(1.0 - p))
    elif breaches == n_obs:
        lr_stat = float(-2.0 * n_obs * np.log(p))
    else:
        term1 = breaches * np.log(p_hat / p)
        term2 = (n_obs - breaches) * np.log((1.0 - p_hat) / (1.0 - p))
        lr_stat = float(2.0 * (term1 + term2))

    lr_stat = max(0.0, lr_stat)
    p_value = float(chi2.sf(lr_stat, df=1))
    # Reject null if p-value < 0.05 (5% significance)
    reject_null = bool(p_value < 0.05)

    return KupiecResult(
        confidence=confidence,
        n_obs=n_obs,
        breaches=breaches,
        expected_breaches=round(expected, 1),
        breach_rate=round(p_hat, 4),
        lr_stat=round(lr_stat, 4),
        p_value=round(p_value, 4),
        reject_null=reject_null,
    )


def backtest_var_model(
    returns: pd.Series,
    window: int = 252,
    confidences: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, KupiecResult]]:
    """Run full rolling VaR backtest across configured models and confidence levels.

    Args:
        returns: Daily simple returns Series.
        window: Lookback estimation window (default: 252).
        confidences: List of confidence levels (default: [0.95, 0.99]).
        methods: List of models (default: ['historical', 'parametric']).

    Returns:
        Tuple of:
            - DataFrame summary table of breaches, failure rates, and Kupiec test outcomes.
            - Dictionary of key -> KupiecResult instances.
    """
    if confidences is None:
        confidences = [0.95, 0.99]
    if methods is None:
        methods = ["historical", "parametric"]

    rows = []
    results_map: Dict[str, KupiecResult] = {}

    for method in methods:
        for conf in confidences:
            key = f"{method}_{int(conf*100)}"
            roll_v = rolling_var(returns, window=window, confidence=conf, method=method)
            br = count_breaches(returns, roll_v)

            n_obs = len(br)
            n_breaches = int(br.sum())
            res = kupiec_pof(breaches=n_breaches, n_obs=n_obs, confidence=conf)
            results_map[key] = res

            rows.append(
                {
                    "Model": method.capitalize(),
                    "Confidence": f"{conf:.0%}",
                    "Observed Breaches": n_breaches,
                    "Expected Breaches": res.expected_breaches,
                    "Breach Rate": f"{res.breach_rate:.2%}",
                    "Expected Rate": f"{1.0 - conf:.2%}",
                    "LR Stat": res.lr_stat,
                    "p-value": res.p_value,
                    "Kupiec Decision": "REJECT" if res.reject_null else "PASS",
                }
            )

    df_summary = pd.DataFrame(rows).set_index(["Model", "Confidence"])
    return df_summary, results_map
