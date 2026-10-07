"""Unit tests for Parquet caching layer."""

from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
from prismrisk.data.cache import ParquetCache
from prismrisk.data.providers import DataProvider


def test_parquet_cache_roundtrip(tmp_path: Path) -> None:
    """Test reading and writing DataFrames to Parquet cache."""
    cache = ParquetCache(cache_dir=tmp_path / "raw", snapshot_dir=tmp_path / "snap")
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({"asset1": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)

    assert not cache.exists("test.parquet")
    cache.write(df, "test.parquet")
    assert cache.exists("test.parquet")

    loaded = cache.read("test.parquet")
    assert loaded is not None
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)


def test_parquet_cache_refresh_behavior(tmp_path: Path) -> None:
    """Test that refresh=False uses cache, and refresh=True re-fetches."""
    cache = ParquetCache(cache_dir=tmp_path / "raw", snapshot_dir=tmp_path / "snap")
    dates = pd.date_range("2023-01-01", periods=3)
    cached_df = pd.DataFrame({"asset": [100.0, 101.0, 102.0]}, index=dates)
    cache.write(cached_df, "test.parquet")

    mock_provider = MagicMock(spec=DataProvider)
    fresh_df = pd.DataFrame({"asset": [200.0, 201.0, 202.0]}, index=dates)
    mock_provider.fetch_all.return_value = fresh_df

    # With refresh=False, should return cached_df without calling provider
    res1 = cache.load_or_fetch(
        tickers={"asset": "TICKER"},
        provider=mock_provider,
        refresh=False,
        cache_filename="test.parquet",
    )
    assert res1.iloc[0]["asset"] == 100.0
    mock_provider.fetch_all.assert_not_called()

    # With refresh=True, should call provider and update cache
    res2 = cache.load_or_fetch(
        tickers={"asset": "TICKER"},
        provider=mock_provider,
        refresh=True,
        cache_filename="test.parquet",
    )
    assert res2.iloc[0]["asset"] == 200.0
    mock_provider.fetch_all.assert_called_once()


def test_parquet_cache_snapshot_fallback(tmp_path: Path) -> None:
    """Test fallback to committed snapshot if provider raises error."""
    snap_dir = tmp_path / "snapshot"
    snap_dir.mkdir(parents=True)
    dates = pd.date_range("2023-01-01", periods=3)
    snap_df = pd.DataFrame({"equity": [100.0, 102.0, 104.0]}, index=dates)
    snap_df.to_parquet(snap_dir / "market_data_snapshot.parquet")

    cache = ParquetCache(cache_dir=tmp_path / "raw", snapshot_dir=snap_dir)
    failing_provider = MagicMock(spec=DataProvider)
    failing_provider.fetch_all.side_effect = ConnectionError("Network down")

    result = cache.load_or_fetch(
        tickers={"equity": "NIFTYBEES.NS"},
        provider=failing_provider,
        refresh=True,
        cache_filename="test.parquet",
        fallback_to_snapshot=True,
    )
    assert not result.empty
    assert "equity" in result.columns
    assert result.iloc[0]["equity"] == 100.0
