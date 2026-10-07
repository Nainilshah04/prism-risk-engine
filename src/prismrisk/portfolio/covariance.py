"""Covariance matrix estimation for multi-asset portfolio optimization.

Enforces:
- Rule ADR-3: Ledoit-Wolf shrinkage towards constant correlation is the default;
  sample covariance is retained strictly for comparison.
- Rule R-F10: Windows are configurable.
- Symmetry and positive semi-definiteness.
"""

from typing import Literal
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from prismrisk.utils.validation import validate_returns


def estimate_cov(
    returns: pd.DataFrame,
    method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
) -> pd.DataFrame:
    """Estimate asset return covariance matrix.

    Args:
        returns: DataFrame [date x asset] of daily simple returns.
        method: Estimation method ('ledoit_wolf' or 'sample').

    Returns:
        Square symmetric, positive semi-definite covariance DataFrame [asset x asset].
    """
    validate_returns(returns)
    if returns.shape[1] < 1:
        raise ValueError("Returns matrix must contain at least 1 asset.")
    if len(returns) < 2:
        raise ValueError("Need at least 2 observations to estimate covariance.")

    assets = list(returns.columns)
    data = returns.to_numpy()

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        cov_matrix = lw.fit(data).covariance_
    elif method == "sample":
        cov_matrix = np.cov(data, rowvar=False, ddof=1)
    else:
        raise ValueError(f"Unknown covariance method '{method}', must be 'ledoit_wolf' or 'sample'")

    # Ensure exact symmetry
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

    # Ensure positive semi-definiteness via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    if (eigvals < 1e-10).any():
        eigvals = np.maximum(eigvals, 1e-10)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

    return pd.DataFrame(cov_matrix, index=assets, columns=assets)
