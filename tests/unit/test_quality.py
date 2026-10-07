"""Unit tests for Data Quality diagnostic reporting."""

import numpy as np
import pandas as pd
from prismrisk.data.quality import (
    compute_longest_consecutive_gap,
    format_quality_table,
    generate_quality_report,
)


def test_longest_consecutive_gap_known_answer() -> None:
    """Test longest consecutive gap counter on known series."""
    # Series with gaps of 2 and 3
    s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 4.0])
    assert compute_longest_consecutive_gap(s) == 3

    # Series with no gaps
    s_full = pd.Series([1.0, 2.0, 3.0])
    assert compute_longest_consecutive_gap(s_full) == 0


def test_quality_report_detects_outliers_and_zeros() -> None:
    """Ensure data quality report detects injected outliers and flat returns."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Regular prices
    rng = np.random.default_rng(42)
    daily_rets = rng.normal(0.0005, 0.01, size=100)
    # Inject massive 10-sigma outlier at index 50
    daily_rets[50] = 0.25  # +25% jump in one day
    asset1_prices = 100.0 * np.exp(np.cumsum(daily_rets))

    # Stale asset (50 days constant price -> zero return)
    stale_prices = [1000.0] * 50 + list(1000.0 * (1 + 0.0002) ** np.arange(50))

    df = pd.DataFrame(
        {
            "volatile": asset1_prices,
            "stale": stale_prices,
        },
        index=dates,
    )

    report = generate_quality_report(aligned_prices=df, outlier_sigma=4.0)

    assert report.total_trading_days == 100
    assert report.outlier_count["volatile"] >= 1
    assert report.zero_return_days["stale"] >= 49
    assert any("stale" in note for note in report.notes)

    # Format table check
    table = format_quality_table(report)
    assert "Raw Missing %" in table.columns
    assert "Zero Return Days" in table.columns
    assert "Outliers (>4-sigma)" in table.columns
    assert len(table) == 2
