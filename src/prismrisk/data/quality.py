"""Data quality diagnostics and reporting.

Enforces Rule FR-D4: Produce a data-quality report:
- Missing % per asset
- Longest consecutive missing gap (days)
- Zero-return days (stale price detection)
- Outlier returns (> N sigma)
- History start dates per asset
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from prismrisk.models import DataQualityReport


def compute_longest_consecutive_gap(series: pd.Series) -> int:
    """Compute maximum consecutive missing (NaN) days in a series."""
    is_na = series.isna()
    if not is_na.any():
        return 0
    # Group consecutive identical boolean blocks
    blocks = (~is_na).cumsum()[is_na]
    counts = blocks.value_counts()
    return int(counts.max()) if not counts.empty else 0


def generate_quality_report(
    aligned_prices: pd.DataFrame,
    raw_prices: Optional[pd.DataFrame] = None,
    asset_start_dates: Optional[Dict[str, pd.Timestamp]] = None,
    outlier_sigma: float = 4.0,
) -> DataQualityReport:
    """Evaluate data quality integrity across all assets.

    Args:
        aligned_prices: Clean, aligned prices DataFrame [date x asset].
        raw_prices: Raw unaligned price matrix before forward fill (optional).
        asset_start_dates: Earliest available date mapping per asset (optional).
        outlier_sigma: Sigma threshold for classifying outlier returns (default: 4.0).

    Returns:
        DataQualityReport containing diagnostic series and overall health flag.
    """
    assets = list(aligned_prices.columns)
    effective_start = aligned_prices.index[0]
    effective_end = aligned_prices.index[-1]
    n_days = len(aligned_prices)

    # 1. Missing percentage & longest gap
    missing_pct_dict: Dict[str, float] = {}
    longest_gap_dict: Dict[str, int] = {}

    ref_raw = raw_prices if raw_prices is not None else aligned_prices
    for asset in assets:
        if asset in ref_raw.columns:
            s = ref_raw.loc[
                (ref_raw.index >= effective_start) & (ref_raw.index <= effective_end),
                asset,
            ]
            missing_pct_dict[asset] = round(float(s.isna().mean() * 100.0), 2)
            longest_gap_dict[asset] = compute_longest_consecutive_gap(s)
        else:
            missing_pct_dict[asset] = 0.0
            longest_gap_dict[asset] = 0

    # 2. Daily returns for zero-return & outlier diagnostics
    simple_rets = aligned_prices.pct_change().dropna(how="all")

    zero_return_dict: Dict[str, int] = {}
    outlier_count_dict: Dict[str, int] = {}
    notes: List[str] = []

    for asset in assets:
        r = simple_rets[asset].dropna()
        # Count identical consecutive prices (return == 0.0)
        zeros = int((r == 0.0).sum())
        zero_return_dict[asset] = zeros
        zero_pct = (zeros / len(r)) * 100.0 if len(r) > 0 else 0.0
        if zero_pct > 20.0:
            notes.append(
                f"Asset '{asset}' exhibits {zero_pct:.1f}% zero-return days (stale pricing or cash proxy)."
            )

        # Outliers (> outlier_sigma)
        std = float(r.std())
        mean = float(r.mean())
        if std > 1e-8:
            z_scores = np.abs((r - mean) / std)
            outliers = int((z_scores > outlier_sigma).sum())
        else:
            outliers = 0
        outlier_count_dict[asset] = outliers
        if outliers > 0:
            notes.append(f"Asset '{asset}' has {outliers} returns beyond {outlier_sigma} sigma.")

    # 3. Start dates per asset
    if asset_start_dates is not None:
        start_dates = pd.Series(asset_start_dates)
    else:
        start_dates = pd.Series({a: effective_start for a in assets})

    missing_series = pd.Series(missing_pct_dict)
    longest_gap_series = pd.Series(longest_gap_dict)
    zero_ret_series = pd.Series(zero_return_dict)
    outlier_series = pd.Series(outlier_count_dict)

    # Health check: pass if no excessive missingness in aligned frame and no multi-week gaps
    passed = bool(
        aligned_prices.isna().sum().sum() == 0 and (longest_gap_series <= 10).all()
    )

    return DataQualityReport(
        missing_pct=missing_series,
        longest_gap_days=longest_gap_series,
        zero_return_days=zero_ret_series,
        outlier_count=outlier_series,
        start_date_per_asset=start_dates,
        effective_start=effective_start,
        effective_end=effective_end,
        total_trading_days=n_days,
        passed_checks=passed,
        notes=notes,
    )


def format_quality_table(report: DataQualityReport) -> pd.DataFrame:
    """Format DataQualityReport into a clean tabular DataFrame for display."""
    df = pd.DataFrame(
        {
            "Start Date": report.start_date_per_asset.dt.strftime("%Y-%m-%d"),
            "Raw Missing %": report.missing_pct.map(lambda x: f"{x:.1f}%"),
            "Longest Gap (Days)": report.longest_gap_days,
            "Zero Return Days": report.zero_return_days,
            "Outliers (>4-sigma)": report.outlier_count,
        }
    )
    df.index.name = "Asset"
    return df
