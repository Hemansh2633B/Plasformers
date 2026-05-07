import pytest
from typer.testing import CliRunner
from plas.cli import app
import json
from pathlib import Path

runner = CliRunner()

@pytest.mark.integration
def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Packages" in result.stdout

@pytest.mark.integration
def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Plasformers enterprise detection framework" in result.stdout

@pytest.mark.integration
def test_cli_benchmark_dry_run():
    # Benchmark might take time, use a small variant and ensure it runs
    result = runner.invoke(app, ["benchmark", "--variant", "nano"])
    assert result.exit_code == 0
    assert "Benchmark" in result.stdout

@pytest.mark.integration
def test_cli_predict_nonexistent_file():
    result = runner.invoke(app, ["predict", "nonexistent.jpg"])
    assert result.exit_code != 0

@pytest.mark.integration
def test_cli_zoo():
    result = runner.invoke(app, ["zoo"])
    assert result.exit_code == 0
    assert "Model Zoo" in result.stdout

@pytest.mark.integration
def test_cli_export_help():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "Export a model to deployment backends" in result.stdout
