"""CLI recipes command group."""

from __future__ import annotations

from importlib import resources

import click


@click.group()
def recipes() -> None:
    """Manage built-in recipes."""


@recipes.command("list")
def list_recipes() -> None:
    """List available built-in recipes."""
    recipe_dir = resources.files("syntheta.recipes")
    recipe_files = [
        f.name for f in recipe_dir.iterdir() if hasattr(f, "name") and f.name.endswith(".yaml")
    ]
    click.echo("Available recipes:")
    for name in sorted(recipe_files):
        click.echo(f"  {name.replace('.yaml', '')}")


@recipes.command("show")
@click.argument("name")
def show_recipe(name: str) -> None:
    """Show the contents of a built-in recipe."""
    try:
        recipe_file = resources.files("syntheta.recipes").joinpath(f"{name}.yaml")
        click.echo(recipe_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError):
        click.echo(f"Recipe '{name}' not found.", err=True)
        raise SystemExit(1) from None


@recipes.command("export")
@click.argument("name")
@click.option("--output", required=True, help="Output file path")
def export_recipe(name: str, output: str) -> None:
    """Export a recipe to a local YAML file for customization."""
    try:
        recipe_file = resources.files("syntheta.recipes").joinpath(f"{name}.yaml")
        content = recipe_file.read_text(encoding="utf-8")
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"Recipe '{name}' exported to {output}")
    except (FileNotFoundError, TypeError):
        click.echo(f"Recipe '{name}' not found.", err=True)
        raise SystemExit(1) from None
