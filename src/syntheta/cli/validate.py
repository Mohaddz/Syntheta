"""CLI validate command."""

from __future__ import annotations

import sys

import click

from syntheta.schema.dataset import SynthDataset

_FORMAT_CHECKS = {
    "sft": lambda s: (
        bool(s.instruction and s.response and len(s.response.strip()) >= 10),
        "missing instruction/response or response < 10 chars",
    ),
    "chat": lambda s: (
        bool(s.turns and len(s.turns) >= 2),
        "missing turns or fewer than 2 turns",
    ),
    "dpo": lambda s: (
        bool(s.instruction and s.chosen and s.rejected and s.chosen != s.rejected),
        "missing instruction/chosen/rejected or chosen == rejected",
    ),
    "rlvr": lambda s: (
        bool(s.instruction and s.info is not None),
        "missing instruction or info",
    ),
    "pretrain": lambda s: (
        bool(s.text and len(s.text.strip()) >= 50),
        "missing text or text < 50 chars",
    ),
}


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(list(_FORMAT_CHECKS)), required=True)
@click.option("--strict", is_flag=True, help="Enable strict quality checks")
def validate(file_path: str, fmt: str, strict: bool) -> None:
    """Validate a dataset against format requirements."""
    ds = SynthDataset.from_jsonl(file_path)

    check_fn = _FORMAT_CHECKS[fmt]
    passed = 0
    failures: list[str] = []

    for i, sample in enumerate(ds):
        ok, reason = check_fn(sample)
        if ok:
            passed += 1
        else:
            failures.append(f"  Row {i}: {reason} (id={sample.id})")

    click.echo(f"\nValidation: {file_path} (format: {fmt})")
    if not failures:
        click.echo(f"  ✓ {passed} / {len(ds)} samples passed")
        sys.exit(0)
    else:
        click.echo(f"  ✓ {passed} / {len(ds)} samples passed")
        click.echo(f"  ✗ {len(failures)} samples failed:")
        for f in failures[:10]:
            click.echo(f)
        if len(failures) > 10:
            click.echo(f"  ... and {len(failures) - 10} more")
        sys.exit(1)
