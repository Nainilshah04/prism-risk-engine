"""Unit tests for data providers."""

from pathlib import Path
import pandas as pd
import pytest
from prismrisk.data.providers import SnapshotProvider, YFinanceProvider


def test_snapshot_provider_success() -> None:
    """Test loading price series from local Parquet snapshot."""
    provider = SnapshotProvider("data/snapshot/market_data_snapshot.parquet")
    s = provider.fetch_series("equity", start="2020-01-01", end="2020-12-31")
    assert not s.empty
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.min() >= pd.to_datetime("2020-01-01")
    assert s.index.max() <= pd.to_datetime("2020-12-31")
    assert (s > 0).all()


def test_snapshot_provider_missing_ticker() -> None:
    """Ensure nonexistent ticker raises KeyError."""
    provider = SnapshotProvider("data/snapshot/market_data_snapshot.parquet")
    with pytest.raises(KeyError, match="not found in snapshot"):
        provider.fetch_series("NON_EXISTENT_ASSET")


def test_snapshot_provider_nonexistent_file() -> None:
    """Ensure missing snapshot path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SnapshotProvider("data/snapshot/does_not_exist.parquet")


def test_yfinance_normalize_series_tz_and_dupes() -> None:
    """Test YFinanceProvider._normalize_series strips tz and drops duplicates."""
    # Create series with timezone and duplicate dates
    idx = pd.to_datetime(["2023-01-01 09:30:00+05:30", "2023-01-01 15:30:00+05:30", "2023-01-02 09:30:00+05:30"])
    raw_s = pd.Series([100.0, 101.0, 102.0], index=idx)

    normalized = YFinanceProvider._normalize_series(raw_s)
    assert normalized.index.tz is None
    assert len(normalized) == 2  # Duplicates kept last -> 101.0 and 102.0
    assert normalized.iloc[0] == 101.0
    assert normalized.iloc[1] == 102.0
