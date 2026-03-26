"""Tests for CLI commands using Click's CliRunner."""

import json

from click.testing import CliRunner

from syntheta.cli.main import cli


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "inspect" in result.output
        assert "validate" in result.output

    def test_generate_dry_run(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate", "--recipe", "sft", "--domain", "test", "--n", "10", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Dry-run estimate" in result.output
        assert "10" in result.output

    def test_recipes_list(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["recipes", "list"])
        assert result.exit_code == 0
        assert "sft" in result.output
        assert "pretrain" in result.output

    def test_recipes_show(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["recipes", "show", "sft"])
        assert result.exit_code == 0
        assert "config_version" in result.output

    def test_recipes_export(self, tmp_path):
        runner = CliRunner()
        output = str(tmp_path / "exported.yaml")
        result = runner.invoke(cli, ["recipes", "export", "sft", "--output", output])
        assert result.exit_code == 0
        assert "exported" in result.output


class TestInspect:
    def test_inspect_file(self, sft_jsonl):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(sft_jsonl)])
        assert result.exit_code == 0
        assert "5" in result.output  # 5 samples
        assert "instruction" in result.output

    def test_inspect_json_output(self, sft_jsonl):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(sft_jsonl), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["samples"] == 5


class TestValidate:
    def test_validate_passes(self, sft_jsonl):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(sft_jsonl), "--format", "sft"])
        assert result.exit_code == 0
        assert "passed" in result.output

    def test_validate_fails_wrong_format(self, sft_jsonl):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(sft_jsonl), "--format", "dpo"])
        assert result.exit_code == 1
        assert "failed" in result.output
