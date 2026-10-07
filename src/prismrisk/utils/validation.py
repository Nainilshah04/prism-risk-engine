"""Input validation helpers for financial time series and portfolio vectors.

Enforces Rule R-C8: Explicit errors. Raise clear exceptions on bad input
(non-sorted index, weights not summing to 1, unexpected NaNs). No silent fixes.
"""

from typing import Iterable, Sequence
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when financial inputs fail contract validation."""
    pass


def validate_prices(prices: pd.DataFrame) -> None:
    """Validate asset price matrix.

    Requirements:
        - Must be a pandas DataFrame.
        - Index must be a sorted DatetimeIndex without duplicates.
        - Columns must be non-empty strings.
        - Prices must be strictly positive and finite (no NaNs, no inf).

    Args:
        prices: Asset price DataFrame [date x asset].

    Raises:
        ValidationError: If any condition is violated.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValidationError(f"Prices must be a pd.DataFrame, got {type(prices).__name__}")
    if prices.empty:
        raise ValidationError("Prices DataFrame is empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValidationError(f"Prices index must be DatetimeIndex, got {type(prices.index).__name__}")
    if not prices.index.is_monotonic_increasing:
        raise ValidationError("Prices DatetimeIndex must be strictly sorted ascending.")
    if not prices.index.is_unique:
        duplicates = prices.index[prices.index.duplicated()].tolist()
        raise ValidationError(f"Prices DatetimeIndex contains duplicate dates: {duplicates[:3]}")
    if prices.isna().any().any():
        nan_cols = prices.columns[prices.isna().any()].tolist()
        raise ValidationError(f"Prices contains NaN values in assets: {nan_cols}")
    if np.isinf(prices.to_numpy()).any():
        raise ValidationError("Prices contains infinite values.")
    if (prices <= 0).any().any():
        raise ValidationError("Prices must be strictly positive (> 0).")


def validate_returns(returns: pd.Series | pd.DataFrame) -> None:
    """Validate return series or return matrix.

    Requirements:
        - Must be a pandas Series or DataFrame.
        - Index must be a sorted DatetimeIndex without duplicates.
        - Values must be finite (no NaNs, no inf).

    Args:
        returns: Return series or return matrix.

    Raises:
        ValidationError: If any condition is violated.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise ValidationError(f"Returns must be pd.Series or pd.DataFrame, got {type(returns).__name__}")
    if returns.empty:
        raise ValidationError("Returns object is empty.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValidationError(f"Returns index must be DatetimeIndex, got {type(returns.index).__name__}")
    if not returns.index.is_monotonic_increasing:
        raise ValidationError("Returns DatetimeIndex must be strictly sorted ascending.")
    if not returns.index.is_unique:
        raise ValidationError("Returns DatetimeIndex contains duplicate dates.")
    if returns.isna().any() if isinstance(returns, pd.Series) else returns.isna().any().any():
        raise ValidationError("Returns contain NaN values.")
    values = returns.to_numpy()
    if np.isinf(values).any():
        raise ValidationError("Returns contain infinite values.")


def validate_weights(
    weights: pd.Series,
    expected_assets: Sequence[str] | None = None,
    allow_short: bool = False,
    tolerance: float = 1e-4,
) -> None:
    """Validate portfolio asset weight vector.

    Requirements:
        - Must be a pandas Series.
        - Sum of weights must equal 1.0 within tolerance.
        - If allow_short=False, all weights must be >= 0.
        - If expected_assets provided, index must match expected assets.

    Args:
        weights: Portfolio weights Series indexed by asset name.
        expected_assets: Optional sequence of asset identifiers.
        allow_short: If False, enforces long-only constraints (w >= 0).
        tolerance: Absolute tolerance for sum(weights) == 1.0.

    Raises:
        ValidationError: If any condition is violated.
    """
    if not isinstance(weights, pd.Series):
        raise ValidationError(f"Weights must be pd.Series, got {type(weights).__name__}")
    if weights.empty:
        raise ValidationError("Weights Series is empty.")
    if weights.isna().any():
        raise ValidationError("Weights contain NaN values.")

    total_weight = float(weights.sum())
    if abs(total_weight - 1.0) > tolerance:
        raise ValidationError(f"Weights must sum to 1.0 (got sum = {total_weight:.6f}).")

    if not allow_short and (weights < -1e-6).any():
        violators = weights[weights < -1e-6].to_dict()
        raise ValidationError(f"Long-only constraint violated. Negative weights found: {violators}")

    if expected_assets is not None:
        expected_set = set(expected_assets)
        weights_set = set(weights.index)
        if expected_set != weights_set:
            diff = expected_set.symmetric_difference(weights_set)
            raise ValidationError(
                f"Weights index does not match expected assets. Mismatched: {diff}"
            )


def validate_confidence(confidence: float) -> None:
    """Validate tail-risk confidence level (e.g. 0.95, 0.99)."""
    if not (0.0 < confidence < 1.0):
        raise ValidationError(f"Confidence must be in (0, 1), got {confidence}")
