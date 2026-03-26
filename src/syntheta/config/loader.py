"""Config loader: YAML + layered overrides + dot-notation merging."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from syntheta.config.defaults import DEFAULTS
from syntheta.exceptions import ConfigError


def load_config(
    config_path: str | None = None,
    recipe: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config with layered overrides: defaults → recipe → YAML → CLI flags.

    Args:
        config_path: Path to user YAML config file.
        recipe: Name of a built-in recipe.
        cli_overrides: Dict of CLI overrides (may use dot notation).

    Returns:
        Merged config dict.
    """
    # Start with defaults
    config = copy.deepcopy(DEFAULTS)

    # Layer recipe
    if recipe:
        recipe_config = _load_recipe(recipe)
        config = _deep_merge(config, recipe_config)

    # Layer user config file
    if config_path:
        user_config = _load_yaml(config_path)
        _validate_version(user_config.get("config_version"))
        config = _deep_merge(config, user_config)

    # Layer CLI overrides (with dot-notation expansion)
    if cli_overrides:
        expanded = _expand_dot_notation(cli_overrides)
        config = _deep_merge(config, expanded)

    return config


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} is not a valid YAML dict")
    return data


def _load_recipe(name: str) -> dict[str, Any]:
    """Load a built-in recipe by name."""
    try:
        recipe_file = resources.files("syntheta.recipes").joinpath(f"{name}.yaml")
        data = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError(f"Recipe '{name}' is not a valid YAML dict")
        return data
    except (FileNotFoundError, TypeError) as e:
        raise ConfigError(
            f"Recipe '{name}' not found. Use 'syntheta recipes list' to see available recipes."
        ) from e


def _validate_version(version: str | None) -> None:
    """Check config version compatibility."""
    if version is None:
        return
    current = DEFAULTS["config_version"]
    if version != current:
        # For now, just warn — don't block
        import logging

        logging.getLogger("syntheta.config").warning(
            "Config version '%s' differs from current '%s'. Some fields may have changed.",
            version,
            current,
        )


def _expand_dot_notation(overrides: dict[str, Any]) -> dict[str, Any]:
    """Expand dot-notation keys into nested dicts.

    {"llm.model": "gpt-4o"} → {"llm": {"model": "gpt-4o"}}
    """
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
