"""Market data providers interface and implementations.

Enforces Rule R-F1: Adjusted prices only. Use adjusted close so splits
and dividends do not create fake returns.
Enforces Rule R-C1: Pure data ingestion layer behind a unified interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf


class DataProvider(ABC):
    """Abstract interface for historical price data providers."""

    @abstractmethod
    def fetch_series(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Fetch adjusted close price series for a single ticker.

        Returns:
            pd.Series with tz-naive DatetimeIndex, sorted, float prices.
        """
        pass

    def fetch_all(
        self,
        tickers: Dict[str, str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for multiple named tickers.

        Args:
            tickers: Mapping of asset name -> ticker symbol (e.g. {'equity': 'NIFTYBEES.NS'}).
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            pd.DataFrame[date x asset] with tz-naive DatetimeIndex.
        """
        series_dict: Dict[str, pd.Series] = {}
        for name, ticker in tickers.items():
            try:
                s = self.fetch_series(ticker=ticker, start=start, end=end)
            except KeyError:
                s = self.fetch_series(ticker=name, start=start, end=end)
            s.name = name
            series_dict[name] = s
        return pd.DataFrame(series_dict)


class YFinanceProvider(DataProvider):
    """Yahoo Finance provider fetching split/dividend-adjusted close prices."""

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def fetch_series(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Download adjusted close prices for a ticker using yfinance."""
        try:
            t = yf.Ticker(ticker)
            # auto_adjust=True ensures Close is split/dividend adjusted
            df = t.history(start=start, end=end, auto_adjust=True, timeout=self.timeout)
            if df.empty or "Close" not in df.columns:
                # Fallback to general download
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    timeout=self.timeout,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch data for ticker '{ticker}' from yfinance: {exc}") from exc

        if df.empty:
            raise ValueError(f"No price data returned for ticker '{ticker}'")

        # Extract Close column (handle multi-level columns if present)
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.levels[0]:
                series = df["Close"].iloc[:, 0]
            else:
                series = df.iloc[:, 0]
        else:
            series = df["Close"]

        # Ensure tz-naive DatetimeIndex normalized to midnight
        series = self._normalize_series(series)
        return series

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        """Normalize series to sorted, tz-naive DatetimeIndex without duplicates."""
        s = series.copy().dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)

        # Strip timezone if present
        if s.index.tz is not None:
            s.index = s.index.tz_convert(None)

        # Normalize to date (midnight)
        s.index = pd.to_datetime(s.index.date)
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        s = s.astype(float)
        return s


class SnapshotProvider(DataProvider):
    """Offline data provider reading directly from a local Parquet snapshot."""

    def __init__(self, snapshot_path: str | Path) -> None:
        self.snapshot_path = Path(snapshot_path)
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {self.snapshot_path}")
        self._data = pd.read_parquet(self.snapshot_path)
        if not isinstance(self._data.index, pd.DatetimeIndex):
            self._data.index = pd.to_datetime(self._data.index)
        if self._data.index.tz is not None:
            self._data.index = self._data.index.tz_convert(None)
        self._data = self._data.sort_index()

    TICKER_ALIASES: Dict[str, str] = {
        "NIFTYBEES.NS": "equity",
        "GOLDBEES.NS": "gold",
        "LIQUIDBEES.NS": "debt",
        "^NSEI": "benchmark",
        "USDINR=X": "fx",
        "^GSPC": "global_benchmark",
    }

    def fetch_series(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Fetch series from snapshot."""
        col = ticker
        if col not in self._data.columns:
            # Check known alias
            if ticker in self.TICKER_ALIASES and self.TICKER_ALIASES[ticker] in self._data.columns:
                col = self.TICKER_ALIASES[ticker]
            else:
                # Check case-insensitive match
                matching = [c for c in self._data.columns if c.lower() == ticker.lower()]
                if matching:
                    col = matching[0]
                else:
                    raise KeyError(
                        f"Ticker '{ticker}' not found in snapshot columns: {list(self._data.columns)}"
                    )

        s = self._data[col].dropna()
        if start:
            s = s[s.index >= pd.to_datetime(start)]
        if end:
            s = s[s.index <= pd.to_datetime(end)]
        return s.astype(float)
