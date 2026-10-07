"""Unit tests for financial validation helpers."""

import numpy as np
import pandas as pd
import pytest
from prismrisk.utils.validation import (
    ValidationError,
    validate_prices,
    validate_returns,
    validate_weights,
)


def test_validate_prices_success(synthetic_prices: pd.DataFrame) -> None:
    """Ensure valid price matrix passes without error."""
    validate_prices(synthetic_prices)


def test_validate_prices_empty() -> None:
    """Ensure empty price matrix raises ValidationError."""
    with pytest.raises(ValidationError, match="empty"):
        validate_prices(pd.DataFrame())


def test_validate_prices_non_datetime_index() -> None:
    """Ensure integer index raises ValidationError."""
    df = pd.DataFrame({"asset": [10.0, 11.0]}, index=[0, 1])
    with pytest.raises(ValidationError, match="DatetimeIndex"):
        validate_prices(df)


def test_validate_prices_unsorted() -> None:
    """Ensure non-monotonic index raises ValidationError."""
    idx = pd.to_datetime(["2023-01-02", "2023-01-01"])
    df = pd.DataFrame({"asset": [10.0, 11.0]}, index=idx)
    with pytest.raises(ValidationError, match="strictly sorted"):
        validate_prices(df)


def test_validate_prices_duplicate_dates() -> None:
    """Ensure duplicate dates raise ValidationError."""
    idx = pd.to_datetime(["2023-01-01", "2023-01-01"])
    df = pd.DataFrame({"asset": [10.0, 11.0]}, index=idx)
    with pytest.raises(ValidationError, match="duplicate dates"):
        validate_prices(df)


def test_validate_prices_contains_nan() -> None:
    """Ensure NaNs raise ValidationError."""
    idx = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"asset": [10.0, np.nan, 12.0]}, index=idx)
    with pytest.raises(ValidationError, match="contains NaN"):
        validate_prices(df)


def test_validate_prices_non_positive() -> None:
    """Ensure non-positive prices raise ValidationError."""
    idx = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"asset": [10.0, 0.0, 12.0]}, index=idx)
    with pytest.raises(ValidationError, match="strictly positive"):
        validate_prices(df)


def test_validate_returns_success() -> None:
    """Ensure valid return series passes."""
    idx = pd.date_range("2023-01-01", periods=3)
    s = pd.Series([0.01, -0.02, 0.015], index=idx)
    validate_returns(s)


def test_validate_returns_with_nan() -> None:
    """Ensure returns with NaN fail."""
    idx = pd.date_range("2023-01-01", periods=3)
    s = pd.Series([0.01, np.nan, 0.015], index=idx)
    with pytest.raises(ValidationError, match="contain NaN"):
        validate_returns(s)


def test_validate_prices_contains_inf() -> None:
    """Ensure infinite price raises ValidationError."""
    idx = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({"asset": [10.0, np.inf, 12.0]}, index=idx)
    with pytest.raises(ValidationError, match="infinite values"):
        validate_prices(df)


def test_validate_returns_with_inf() -> None:
    """Ensure returns with infinity fail."""
    idx = pd.date_range("2023-01-01", periods=3)
    s = pd.Series([0.01, np.inf, 0.015], index=idx)
    with pytest.raises(ValidationError, match="infinite values"):
        validate_returns(s)


def test_validate_returns_empty() -> None:
    """Ensure empty returns fail."""
    with pytest.raises(ValidationError, match="empty"):
        validate_returns(pd.Series(dtype=float))


def test_validate_confidence_levels() -> None:
    """Ensure confidence validation accepts valid and rejects out-of-bound levels."""
    from prismrisk.utils.validation import validate_confidence

    validate_confidence(0.95)
    validate_confidence(0.99)
    with pytest.raises(ValidationError, match="must be in"):
        validate_confidence(1.05)
    with pytest.raises(ValidationError, match="must be in"):
        validate_confidence(-0.05)


def test_validate_weights_success() -> None:
    """Ensure proper weights pass."""
    w = pd.Series({"equity": 0.60, "gold": 0.20, "debt": 0.20})
    validate_weights(w, expected_assets=["equity", "gold", "debt"])


def test_validate_weights_sum_error() -> None:
    """Ensure sum != 1 raises ValidationError."""
    w = pd.Series({"equity": 0.50, "gold": 0.20, "debt": 0.20})
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        validate_weights(w)


def test_validate_weights_negative_rejected() -> None:
    """Ensure negative weights are rejected in long-only mode."""
    w = pd.Series({"equity": 1.20, "debt": -0.20})
    with pytest.raises(ValidationError, match="Long-only constraint violated"):
        validate_weights(w)


def test_validate_weights_mismatched_assets() -> None:
    """Ensure mismatched asset keys fail."""
    w = pd.Series({"equity": 0.50, "cash": 0.50})
    with pytest.raises(ValidationError, match="does not match expected assets"):
        validate_weights(w, expected_assets=["equity", "debt"])
