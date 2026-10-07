"""Unit tests for calendar alignment and forward-fill policy."""

import numpy as np
import pandas as pd
import pytest
from prismrisk.data.calendar import align_prices
from prismrisk.utils.validation import ValidationError


def test_align_prices_basic() -> None:
    """Test standard multi-asset alignment without gaps."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    raw = pd.DataFrame(
        {
            "asset1": np.linspace(100, 110, 10),
            "asset2": np.linspace(50, 55, 10),
        },
        index=dates,
    )
    aligned, start_dates = align_prices(raw, max_ffill_days=3)
    assert len(aligned) == 10
    assert start_dates["asset1"] == dates[0]
    assert start_dates["asset2"] == dates[0]
    assert aligned.isna().sum().sum() == 0


def test_align_prices_small_gap_filled() -> None:
    """Ensure gap of 2 days (<= max_ffill_days=3) is filled via forward-fill."""
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    raw = pd.DataFrame(
        {
            "asset1": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            # 2 missing days on index 2 and 3
            "asset2": [50.0, 51.0, np.nan, np.nan, 54.0, 55.0],
        },
        index=dates,
    )
    aligned, _ = align_prices(raw, max_ffill_days=3)
    assert len(aligned) == 6
    assert aligned.loc[dates[2], "asset2"] == 51.0
    assert aligned.loc[dates[3], "asset2"] == 51.0
    assert aligned.loc[dates[4], "asset2"] == 54.0


def test_align_prices_large_gap_rejected() -> None:
    """Ensure gap exceeding max_ffill_days raises ValidationError when drop_unfillable=False."""
    dates = pd.date_range("2023-01-01", periods=7, freq="D")
    raw = pd.DataFrame(
        {
            "asset1": [100.0] * 7,
            # 4 consecutive NaNs > max_ffill_days=3
            "asset2": [50.0, np.nan, np.nan, np.nan, np.nan, 55.0, 56.0],
        },
        index=dates,
    )
    with pytest.raises(ValidationError, match="Missing price gaps exceed max_ffill_days"):
        align_prices(raw, max_ffill_days=3, drop_unfillable=False)


def test_align_prices_large_gap_dropped() -> None:
    """Ensure gap exceeding max_ffill_days is dropped when drop_unfillable=True."""
    dates = pd.date_range("2023-01-01", periods=7, freq="D")
    raw = pd.DataFrame(
        {
            "asset1": [100.0] * 7,
            # 4 consecutive NaNs > max_ffill_days=3
            "asset2": [50.0, np.nan, np.nan, np.nan, np.nan, 55.0, 56.0],
        },
        index=dates,
    )
    aligned, _ = align_prices(raw, max_ffill_days=3, drop_unfillable=True)
    # The 4th NaN row cannot be filled and should be dropped
    assert len(aligned) < 7
    assert aligned.isna().sum().sum() == 0


def test_align_prices_different_start_dates() -> None:
    """Ensure common sample truncates to latest asset's inception date."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    raw = pd.DataFrame(
        {
            "asset_old": np.linspace(100, 110, 10),
            # Asset young only starts on day 5 (index 4)
            "asset_young": [np.nan] * 4 + list(np.linspace(20, 25, 6)),
        },
        index=dates,
    )
    aligned, start_dates = align_prices(raw, max_ffill_days=3)
    assert start_dates["asset_old"] == dates[0]
    assert start_dates["asset_young"] == dates[4]
    assert aligned.index[0] == dates[4]  # Truncated to common sample
    assert len(aligned) == 6


def test_align_prices_empty_or_non_datetime() -> None:
    """Ensure bad inputs raise ValidationError."""
    with pytest.raises(ValidationError, match="empty"):
        align_prices(pd.DataFrame())

    with pytest.raises(ValidationError, match="DatetimeIndex"):
        align_prices(pd.DataFrame({"a": [1, 2]}, index=[0, 1]))
