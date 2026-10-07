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


def test_cli_placeholder_commands() -> None:
    """Test stub commands run, report, app."""
    res_run = runner.invoke(app, ["run"])
    assert res_run.exit_code == 0
    assert "Full analytics pipeline" in res_run.output

    res_report = runner.invoke(app, ["report"])
    assert res_report.exit_code == 0
    assert "HTML Factsheet generator" in res_report.output

    res_app = runner.invoke(app, ["app-cmd"])
    assert res_app.exit_code == 0
    assert "Launching Streamlit" in res_app.output
