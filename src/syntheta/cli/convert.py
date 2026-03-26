"""CLI convert command."""

from __future__ import annotations

import click

from syntheta.schema.dataset import SynthDataset


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--to", "fmt", type=str, required=True, help="Target format")
@click.option("--output", type=str, required=True, help="Output file path")
def convert(file_path: str, fmt: str, output: str) -> None:
    """Convert a dataset to a different format."""
    ds = SynthDataset.from_jsonl(file_path)
    count = ds.save(output, format=fmt)
    click.echo(f"Converted {count} samples to {fmt} format → {output}")
