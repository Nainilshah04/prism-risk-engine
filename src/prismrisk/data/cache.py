"""Local Parquet caching layer for market data.

Enforces Rule FR-D2: Cache raw data to Parquet; a --refresh flag forces re-download;
a committed snapshot allows offline runs.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from prismrisk.data.providers import DataProvider, SnapshotProvider

logger = logging.getLogger(__name__)


class ParquetCache:
    """Manages disk caching of time series data via pyarrow Parquet."""

    def __init__(
        self,
        cache_dir: str | Path = "data/raw",
        snapshot_dir: str | Path = "data/snapshot",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str = "raw_prices.parquet") -> Path:
        """Return full path in cache directory."""
        return self.cache_dir / filename

    def get_snapshot_path(self, filename: str = "market_data_snapshot.parquet") -> Path:
        """Return full path in snapshot directory."""
        return self.snapshot_dir / filename

    def exists(self, filename: str = "raw_prices.parquet") -> bool:
        """Check if cached file exists."""
        return self.get_path(filename).exists()

    def read(self, filename: str = "raw_prices.parquet") -> Optional[pd.DataFrame]:
        """Read Parquet cache file if it exists."""
        path = self.get_path(filename)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def write(self, df: pd.DataFrame, filename: str = "raw_prices.parquet") -> Path:
        """Write DataFrame to Parquet cache."""
        path = self.get_path(filename)
        df.to_parquet(path, engine="pyarrow")
        return path

    def load_or_fetch(
        self,
        tickers: Dict[str, str],
        provider: DataProvider,
        refresh: bool = False,
        cache_filename: str = "raw_prices.parquet",
        start: Optional[str] = None,
        end: Optional[str] = None,
        fallback_to_snapshot: bool = True,
    ) -> pd.DataFrame:
        """Fetch raw prices with caching, refresh toggle, and snapshot fallback.

        Args:
            tickers: Asset name -> Ticker mapping (e.g. {'equity': 'NIFTYBEES.NS'}).
            provider: Active DataProvider instance.
            refresh: If True, forces network download and overwrites cache.
            cache_filename: Parquet file name.
            start: Start date string.
            end: End date string.
            fallback_to_snapshot: If network download fails, load committed snapshot.

        Returns:
            pd.DataFrame[date x asset] with raw prices.
        """
        # If cache exists and refresh is not requested, load from cache
        if not refresh and self.exists(cache_filename):
            cached_df = self.read(cache_filename)
            if cached_df is not None and not cached_df.empty:
                logger.info(f"Loaded {len(cached_df)} rows from cache: {self.get_path(cache_filename)}")
                return cached_df

        # Otherwise fetch from provider
        try:
            logger.info("Fetching market data from provider...")
            df = provider.fetch_all(tickers=tickers, start=start, end=end)
            if not df.empty:
                self.write(df, cache_filename)
                return df
        except Exception as exc:
            logger.warning(f"Provider fetch failed: {exc}")
            if not fallback_to_snapshot:
                raise

        # Fallback to committed snapshot
        snap_path = self.get_snapshot_path("market_data_snapshot.parquet")
        if snap_path.exists():
            logger.warning(f"Falling back to committed snapshot: {snap_path}")
            snap_provider = SnapshotProvider(snap_path)
            # Map ticker names to snapshot columns if available
            return snap_provider.fetch_all(tickers=tickers, start=start, end=end)

        raise RuntimeError(
            "Failed to fetch market data and no committed snapshot available."
        )
