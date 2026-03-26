"""Syntheta CLI: main entry point."""

from __future__ import annotations

import click

from syntheta._version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="syntheta")
def cli() -> None:
    """Syntheta — generate synthetic LLM training datasets."""


# Import subcommands to register them
from syntheta.cli.convert import convert  # noqa: E402
from syntheta.cli.generate import generate  # noqa: E402
from syntheta.cli.inspect_cmd import inspect_cmd  # noqa: E402
from syntheta.cli.recipes import recipes  # noqa: E402
from syntheta.cli.validate import validate  # noqa: E402

cli.add_command(generate)
cli.add_command(inspect_cmd, name="inspect")
cli.add_command(validate)
cli.add_command(convert)
cli.add_command(recipes)
