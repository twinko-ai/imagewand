import click
from click.testing import CliRunner

from imagewand.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_autofix():
    runner = CliRunner()
    result = runner.invoke(cli, ["autofix", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_filter():
    runner = CliRunner()
    result = runner.invoke(cli, ["filter", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
