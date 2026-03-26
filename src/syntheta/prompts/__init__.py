"""Prompt template system: load external .txt templates and render {{variable}} placeholders."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any

from syntheta.exceptions import PromptError, TemplateMissingVariableError

_TEMPLATE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def load_prompt(
    name: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Load a prompt template by dotted name.

    Args:
        name: Dotted template name, e.g. "generators.topic_tree".
              Maps to syntheta/prompts/generators/topic_tree.txt
        overrides: User-provided overrides. Keys are template names,
                   values are either file paths or inline template strings.

    Returns:
        The raw template string with {{variable}} placeholders.
    """
    # Check user overrides first
    if overrides and name in overrides:
        override = overrides[name]
        # If it looks like a file path, read it
        path = Path(override)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        # Otherwise treat as inline template
        return override

    # Load from bundled templates
    parts = name.split(".")
    if len(parts) != 2:
        raise PromptError(
            f"Invalid prompt name '{name}'. Expected format: 'category.template_name'"
        )

    category, template_name = parts
    try:
        template_file = (
            resources.files("syntheta.prompts").joinpath(category).joinpath(f"{template_name}.txt")
        )
        return template_file.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError) as e:
        raise PromptError(
            f"Prompt template '{name}' not found. "
            f"Expected at syntheta/prompts/{category}/{template_name}.txt"
        ) from e


def render_template(template: str, **kwargs: Any) -> str:
    """Replace {{variable}} placeholders in a template string.

    Args:
        template: Template string with {{variable}} placeholders.
        **kwargs: Values for each placeholder.

    Returns:
        Rendered string with placeholders replaced.

    Raises:
        TemplateMissingVariableError: If a placeholder has no corresponding value.
    """
    missing = []

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name not in kwargs:
            missing.append(var_name)
            return match.group(0)
        return str(kwargs[var_name])

    result = _TEMPLATE_PATTERN.sub(replacer, template)

    if missing:
        raise TemplateMissingVariableError(f"Missing template variables: {', '.join(missing)}")

    return result
