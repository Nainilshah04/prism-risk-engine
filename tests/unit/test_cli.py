"""Unit tests for Prism Risk Engine Typer CLI."""

from typer.testing import CliRunner
from prismrisk.cli import app

runner = CliRunner()


def test_cli_check_data_offline() -> None:
    """Test 'prismrisk check-data --offline' command."""
    result = runner.invoke(app, ["check-data", "--config", "configs/demo.yaml", "--offline"])
    assert result.exit_code == 0
    assert "Prism Risk Engine" in result.output
    assert "Data Integrity Check: PASSED" in result.output
    assert "equity" in result.output


def test_cli_run() -> None:
    """Test 'prismrisk run --offline' command for strategy comparison."""
    result = runner.invoke(app, ["run", "--config", "configs/demo.yaml", "--offline"])
    assert result.exit_code == 0
    assert "Walk-Forward Strategy Comparison Table" in result.output
    assert "Equal" in result.output
    assert "Parity" in result.output
    assert "Sharpe" in result.output
    assert "Methodological Note on Max Sharpe Stability" in result.output


def test_cli_placeholder_commands() -> None:
    """Test remaining stub commands report, app-cmd."""
    res_report = runner.invoke(app, ["report"])
    assert res_report.exit_code == 0
    assert "HTML Factsheet generator" in res_report.output

    res_app = runner.invoke(app, ["app-cmd"])
    assert res_app.exit_code == 0
    assert "Launching Streamlit" in res_app.output

