"""Tests for prompt template loading and rendering."""

import pytest

from syntheta.exceptions import PromptError, TemplateMissingVariableError
from syntheta.prompts import load_prompt, render_template


class TestRenderTemplate:
    def test_basic_substitution(self):
        result = render_template("Hello {{name}}, you are {{age}}!", name="Alice", age=30)
        assert result == "Hello Alice, you are 30!"

    def test_no_placeholders(self):
        result = render_template("No variables here.")
        assert result == "No variables here."

    def test_missing_variable_raises(self):
        with pytest.raises(TemplateMissingVariableError, match="missing_var"):
            render_template("Hello {{missing_var}}!")

    def test_multiple_same_variable(self):
        result = render_template("{{x}} and {{x}}", x="val")
        assert result == "val and val"

    def test_extra_kwargs_ignored(self):
        result = render_template("{{a}}", a="1", b="2")
        assert result == "1"


class TestLoadPrompt:
    def test_load_builtin_template(self):
        template = load_prompt("generators.topic_tree")
        assert "{{domain}}" in template
        assert "{{breadth}}" in template

    def test_load_response_generator(self):
        template = load_prompt("transformers.response_generator")
        assert "{{instruction}}" in template

    def test_invalid_name_raises(self):
        with pytest.raises(PromptError, match="Invalid prompt name"):
            load_prompt("invalid_no_dots")

    def test_nonexistent_template_raises(self):
        with pytest.raises(PromptError, match="not found"):
            load_prompt("generators.nonexistent")

    def test_override_inline(self):
        custom = "Custom template: {{var}}"
        result = load_prompt("generators.topic_tree", overrides={"generators.topic_tree": custom})
        assert result == custom

    def test_override_file(self, tmp_path):
        f = tmp_path / "custom.txt"
        f.write_text("File override: {{x}}")
        result = load_prompt("generators.topic_tree", overrides={"generators.topic_tree": str(f)})
        assert result == "File override: {{x}}"
