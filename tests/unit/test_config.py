"""Tests for config loader."""

import pytest

from syntheta.config.loader import (
    _deep_merge,
    _expand_dot_notation,
    load_config,
)
from syntheta.exceptions import ConfigError


class TestDotNotation:
    def test_simple(self):
        result = _expand_dot_notation({"llm.model": "gpt-4o"})
        assert result == {"llm": {"model": "gpt-4o"}}

    def test_deep_nesting(self):
        result = _expand_dot_notation({"filters.quality.min_score": 0.8})
        assert result == {"filters": {"quality": {"min_score": 0.8}}}

    def test_no_dots(self):
        result = _expand_dot_notation({"n": 500})
        assert result == {"n": 500}


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"llm": {"model": "gpt-4o-mini", "max_concurrent": 10}}
        override = {"llm": {"model": "gpt-4o"}}
        result = _deep_merge(base, override)
        assert result["llm"]["model"] == "gpt-4o"
        assert result["llm"]["max_concurrent"] == 10

    def test_new_keys_added(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}


class TestLoadConfig:
    def test_defaults_loaded(self):
        config = load_config()
        assert config["n"] == 100
        assert config["llm"]["model"] == "gpt-4o-mini"

    def test_recipe_overrides_defaults(self):
        config = load_config(recipe="sft")
        assert config["n"] == 1000  # recipe sets n=1000

    def test_cli_overrides_recipe(self):
        config = load_config(recipe="sft", cli_overrides={"n": 500})
        assert config["n"] == 500

    def test_dot_notation_cli_override(self):
        config = load_config(cli_overrides={"llm.model": "gpt-4o"})
        assert config["llm"]["model"] == "gpt-4o"

    def test_yaml_config(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text('config_version: "0.1"\nn: 2000\ndomain: "test"\n')
        config = load_config(config_path=str(cfg_file))
        assert config["n"] == 2000
        assert config["domain"] == "test"

    def test_nonexistent_config_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config(config_path="/nonexistent/path.yaml")

    def test_nonexistent_recipe_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config(recipe="nonexistent_recipe")
