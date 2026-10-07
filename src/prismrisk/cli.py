"""Command Line Interface for Prism Risk Engine.

Implements CLI commands using Typer and Rich:
- prismrisk check-data [--config configs/demo.yaml] [--offline]
- prismrisk refresh-data [--config configs/demo.yaml]
- prismrisk run [--config configs/demo.yaml]
- prismrisk report [--config configs/demo.yaml]
- prismrisk app
"""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from prismrisk.config import Config
from prismrisk.data.cache import ParquetCache
from prismrisk.data.calendar import align_prices
from prismrisk.data.providers import DataProvider, SnapshotProvider, YFinanceProvider
from prismrisk.data.quality import format_quality_table, generate_quality_report

app = typer.Typer(name="prismrisk", help="Prism Risk Engine: Multi-Asset Risk Analytics", no_args_is_help=True)
console = Console()


@app.command()
def check_data(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c", help="Path to YAML configuration"),
    offline: bool = typer.Option(True, "--offline/--online", help="Run offline using committed snapshot"),
) -> None:
    """Ingest, align prices, and display the Data Quality diagnostic report."""
    cfg = Config.load(config)
    rprint(f"[bold cyan]Prism Risk Engine[/bold cyan] | Project: [green]{cfg.project}[/green]")
    rprint(f"Data Source Mode: [bold yellow]{'OFFLINE SNAPSHOT' if offline else 'LIVE PROVIDER'}[/bold yellow]")

    tickers = {name: asset.ticker for name, asset in cfg.assets.items()}

    provider: DataProvider
    if offline:
        provider = SnapshotProvider(cfg.data.snapshot_dir + "/market_data_snapshot.parquet")
        raw_prices = provider.fetch_all(tickers=tickers, start=cfg.data.start, end=cfg.data.end)
    else:
        cache = ParquetCache(cache_dir=cfg.data.cache_dir, snapshot_dir=cfg.data.snapshot_dir)
        provider = YFinanceProvider()
        raw_prices = cache.load_or_fetch(tickers=tickers, provider=provider, refresh=False, start=cfg.data.start, end=cfg.data.end)

    aligned_prices, start_dates = align_prices(
        raw_prices=raw_prices,
        max_ffill_days=cfg.data.max_ffill_days,
        start_date=cfg.data.start,
        end_date=cfg.data.end,
    )

    report = generate_quality_report(
        aligned_prices=aligned_prices,
        raw_prices=raw_prices,
        asset_start_dates=start_dates,
    )

    # Render Summary Table
    table = Table(title=f"Data Quality Report ({report.effective_start.strftime('%Y-%m-%d')} to {report.effective_end.strftime('%Y-%m-%d')})")
    table.add_column("Asset", style="bold cyan")
    table.add_column("Start Date", justify="center")
    table.add_column("Raw Missing %", justify="right")
    table.add_column("Max Gap (Days)", justify="right")
    table.add_column("Zero Return Days", justify="right")
    table.add_column("Outliers (>4-sigma)", justify="right")

    summary_df = format_quality_table(report)
    for asset, row in summary_df.iterrows():
        table.add_row(
            str(asset),
            str(row["Start Date"]),
            str(row["Raw Missing %"]),
            str(row["Longest Gap (Days)"]),
            str(row["Zero Return Days"]),
            str(row["Outliers (>4-sigma)"]),
        )

    console.print(table)
    rprint(f"\n[bold]Total Trading Days:[/bold] {report.total_trading_days}")
    rprint(f"[bold]Data Integrity Check:[/bold] [{'green]PASSED' if report.passed_checks else 'red]FAILED'}[/]")

    if report.notes:
        rprint("\n[bold yellow]Diagnostic Notes:[/bold yellow]")
        for note in report.notes:
            rprint(f"  * {note}")


@app.command()
def refresh_data(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c", help="Path to YAML configuration"),
) -> None:
    """Download fresh market data from yfinance and rebuild local cache."""
    cfg = Config.load(config)
    rprint(f"[bold cyan]Fetching fresh market data for '{cfg.project}'...[/bold cyan]")
    cache = ParquetCache(cache_dir=cfg.data.cache_dir, snapshot_dir=cfg.data.snapshot_dir)
    provider = YFinanceProvider()
    tickers = {name: asset.ticker for name, asset in cfg.assets.items()}

    df = cache.load_or_fetch(
        tickers=tickers,
        provider=provider,
        refresh=True,
        cache_filename="raw_prices.parquet",
        start=cfg.data.start,
        end=cfg.data.end,
    )
    rprint(f"[bold green]Successfully cached {len(df)} rows across {len(df.columns)} assets to {cache.get_path('raw_prices.parquet')}[/bold green]")


@app.command()
def run(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c", help="Path to YAML configuration"),
) -> None:
    """Execute full risk analysis pipeline (available in subsequent phases)."""
    rprint("[bold yellow]Full analytics pipeline will run here (Phases 2-6). Use 'check-data' for Phase 1.[/bold yellow]")


@app.command()
def report(
    config: str = typer.Option("configs/demo.yaml", "--config", "-c", help="Path to YAML configuration"),
) -> None:
    """Generate self-contained HTML risk factsheet (Phase 6)."""
    rprint("[bold yellow]HTML Factsheet generator will be available in Phase 6.[/bold yellow]")


@app.command()
def app_cmd() -> None:
    """Launch Streamlit dashboard."""
    rprint("[bold yellow]Launching Streamlit application (Phase 7)...[/bold yellow]")


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
