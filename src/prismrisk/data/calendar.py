"""Trading calendar alignment and forward-fill policy.

Enforces:
- Rule FR-D3: Align all assets to a common trading calendar with an explicit,
  documented fill policy (forward-fill with a max gap, otherwise flag).
- Rule R-F7: Common sample. Compare assets and strategies only over the same date range.
- Rule R-F8: Missing data. Forward-fill only up to the configured max gap;
  beyond that, flag and exclude. Never silently fill with zeros.
"""

from typing import Dict, Optional, Tuple
import pandas as pd
from prismrisk.utils.validation import ValidationError, validate_prices


def align_prices(
    raw_prices: pd.DataFrame,
    max_ffill_days: int = 3,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    drop_unfillable: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]:
    """Align multi-asset prices to a synchronized calendar using bounded forward-fill.

    Process:
        1. Identifies each asset's earliest available price date.
        2. Computes the common sample window:
           effective_start = max(configured start_date, max(asset first available dates)).
        3. Restricts data to [effective_start, effective_end].
        4. Applies forward-fill strictly up to `max_ffill_days`.
        5. If gaps remain exceeding `max_ffill_days`:
           - If drop_unfillable is True: drops dates where any asset is still NaN and logs warning.
           - If drop_unfillable is False: raises ValidationError detailing the gap.

    Args:
        raw_prices: DataFrame of unaligned prices [date x asset].
        max_ffill_days: Maximum consecutive trading days allowed for forward-fill (default: 3).
        start_date: Desired starting date (YYYY-MM-DD).
        end_date: Desired ending date (YYYY-MM-DD).
        drop_unfillable: Whether to exclude dates exceeding max_ffill_days or raise error.

    Returns:
        Tuple of:
            - Aligned, cleaned pd.DataFrame [date x asset] with tz-naive DatetimeIndex.
            - Mapping of asset name -> first available date pd.Timestamp.

    Raises:
        ValidationError: If input is empty, non-datetime, or unfillable gaps remain
                         with drop_unfillable=False.
    """
    if raw_prices.empty:
        raise ValidationError("Input prices DataFrame is empty.")
    if not isinstance(raw_prices.index, pd.DatetimeIndex):
        raise ValidationError("Prices index must be a pd.DatetimeIndex.")

    # Ensure tz-naive and sorted
    df = raw_prices.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index.date)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 1. Determine first available date per asset
    start_dates: Dict[str, pd.Timestamp] = {}
    for col in df.columns:
        valid_idx = df[col].dropna().index
        if len(valid_idx) == 0:
            raise ValidationError(f"Asset '{col}' has no valid price observations.")
        start_dates[col] = valid_idx[0]

    # 2. Determine common sample start
    latest_asset_start = max(start_dates.values())
    if start_date is not None:
        req_start = pd.to_datetime(start_date)
        effective_start = max(req_start, latest_asset_start)
    else:
        effective_start = latest_asset_start

    # Determine common sample end
    if end_date is not None:
        effective_end = pd.to_datetime(end_date)
        df = df.loc[(df.index >= effective_start) & (df.index <= effective_end)]
    else:
        df = df.loc[df.index >= effective_start]

    if df.empty:
        raise ValidationError(
            f"No overlapping dates found after effective start date {effective_start.date()}."
        )

    # 3. Apply bounded forward fill
    filled_df = df.ffill(limit=max_ffill_days)

    # 4. Check for remaining missing values
    remaining_nans = filled_df.isna()
    if remaining_nans.any().any():
        bad_cols = filled_df.columns[remaining_nans.any()].tolist()
        if not drop_unfillable:
            raise ValidationError(
                f"Missing price gaps exceed max_ffill_days ({max_ffill_days}) for assets: {bad_cols}"
            )
        # Exclude dates where any asset could not be filled
        filled_df = filled_df.dropna(how="any")

    if filled_df.empty:
        raise ValidationError("Cleaned price matrix is empty after dropping unfillable gaps.")

    # 5. Final validation check
    validate_prices(filled_df)

    return filled_df, start_dates
