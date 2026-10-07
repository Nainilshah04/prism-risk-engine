"""Risk decomposition and percentage risk contribution per asset.

Enforces:
- Rule FR-P4: Risk contribution per asset (marginal and percentage).
- Sum of percentage risk contributions equals 100%.
"""

import numpy as np
import pandas as pd
from prismrisk.utils.validation import validate_weights


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Decompose portfolio volatility into asset risk contributions.

    Formulas:
        sigma_p = sqrt(w^T * Sigma * w)
        MRC_i = (Sigma * w)_i / sigma_p
        RC_i = w_i * MRC_i
        %RC_i = RC_i / sigma_p = (w_i * (Sigma * w)_i) / (w^T * Sigma * w)

    Args:
        weights: Portfolio weights Series [asset].
        cov: Asset covariance DataFrame [asset x asset].

    Returns:
        DataFrame [asset x 4] with Weight, Marginal RC, Total RC, and % RC.
    """
    assets = list(weights.index)
    validate_weights(weights, expected_assets=list(cov.columns))

    w = weights.loc[assets].to_numpy()
    sigma_mat = cov.loc[assets, assets].to_numpy()

    port_var = float(w @ sigma_mat @ w)
    port_vol = np.sqrt(max(1e-12, port_var))

    # Marginal Risk Contribution: (Sigma * w) / sigma_p
    cov_w = sigma_mat @ w
    mrc = cov_w / port_vol

    # Absolute Risk Contribution: w_i * MRC_i
    rc = w * mrc

    # Percentage Risk Contribution: RC_i / sigma_p
    pct_rc = rc / port_vol

    df = pd.DataFrame(
        {
            "Weight": w,
            "Marginal Risk Contribution": mrc,
            "Risk Contribution": rc,
            "Percentage Risk Contribution": pct_rc,
        },
        index=assets,
    )
    df.index.name = "Asset"
    return df
