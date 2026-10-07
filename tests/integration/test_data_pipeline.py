"""Integration test: End-to-end data ingestion from offline snapshot to aligned prices and DQ report."""

from prismrisk.config import Config
from prismrisk.data.calendar import align_prices
from prismrisk.data.providers import SnapshotProvider
from prismrisk.data.quality import format_quality_table, generate_quality_report
from prismrisk.utils.validation import validate_prices


def test_offline_data_pipeline_end_to_end() -> None:
    """Verify that a single workflow produces clean aligned prices and DQ report

    strictly from the snapshot with zero network requests.
    """
    # 1. Load config
    cfg = Config.load("configs/demo.yaml")

    # 2. Extract tickers
    tickers = {name: asset.ticker for name, asset in cfg.assets.items()}

    # 3. Load from committed snapshot
    snapshot_path = "data/snapshot/market_data_snapshot.parquet"
    provider = SnapshotProvider(snapshot_path)

    # Fetch raw data for all configured assets
    raw_df = provider.fetch_all(tickers=tickers, start=cfg.data.start, end=cfg.data.end)
    assert not raw_df.empty
    assert len(raw_df.columns) == len(tickers)

    # 4. Align prices to synchronized trading calendar
    aligned_prices, start_dates = align_prices(
        raw_prices=raw_df,
        max_ffill_days=cfg.data.max_ffill_days,
        start_date=cfg.data.start,
        end_date=cfg.data.end,
    )

    # 5. Validate output contracts
    validate_prices(aligned_prices)
    assert aligned_prices.isna().sum().sum() == 0

    # 6. Generate Data Quality Report
    dq_report = generate_quality_report(
        aligned_prices=aligned_prices,
        raw_prices=raw_df,
        asset_start_dates=start_dates,
    )

    assert dq_report.passed_checks is True
    assert dq_report.total_trading_days > 1000

    # 7. Format quality summary table
    table = format_quality_table(dq_report)
    assert len(table) == len(tickers)
